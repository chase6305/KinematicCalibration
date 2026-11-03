"""Post-process calibrated URDF geometry without changing its transforms."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_SPECIAL_ANGLE_STEP_RAD = np.pi / 2.0
_SPECIAL_ANGLE_SNAP_TOLERANCE_RAD = 1e-10
_SPECIAL_ANGLE_TIE_TOLERANCE_RAD = 1e-9
_FK_VERIFICATION_RANDOM_SAMPLES = 2048


def _vector(element, attribute: str, default: str) -> np.ndarray:
    if element is None:
        return np.fromstring(default, sep=" ")
    return np.fromstring(element.attrib.get(attribute, default), sep=" ")


def _nearest_equivalent_rpy(rpy, nominal_rpy) -> np.ndarray:
    """Choose a nominal-near XYZ branch with clean right-angle values."""
    canonical = Rotation.from_euler("xyz", rpy).as_euler("xyz")
    alternate = np.array(
        [canonical[0] + np.pi, np.pi - canonical[1], canonical[2] + np.pi]
    )
    reference = np.asarray(nominal_rpy, dtype=float)
    candidates = []
    for candidate in (canonical, alternate):
        candidate = candidate + 2.0 * np.pi * np.round(
            (reference - candidate) / (2.0 * np.pi)
        )
        candidates.append(candidate)

    nominal_distances = np.asarray(
        [np.linalg.norm(value - reference) for value in candidates]
    )
    nearest_distance = float(nominal_distances.min())
    nominal_near = [
        value
        for value, distance in zip(candidates, nominal_distances, strict=True)
        if distance <= nearest_distance + _SPECIAL_ANGLE_TIE_TOLERANCE_RAD
    ]

    def special_angle_distance(value):
        nearest = _SPECIAL_ANGLE_STEP_RAD * np.round(value / _SPECIAL_ANGLE_STEP_RAD)
        return np.linalg.norm(value - nearest)

    selected = min(nominal_near, key=special_angle_distance).copy()
    nearest_special = _SPECIAL_ANGLE_STEP_RAD * np.round(
        selected / _SPECIAL_ANGLE_STEP_RAD
    )
    snap_mask = np.abs(selected - nearest_special) <= _SPECIAL_ANGLE_SNAP_TOLERANCE_RAD
    selected[snap_mask] = nearest_special[snap_mask]
    return selected


def _transform(xyz, rpy) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def _fk(local_transforms, axes, joint_indices, joint_positions) -> np.ndarray:
    pose = np.eye(4)
    for local, axis, joint_index in zip(
        local_transforms, axes, joint_indices, strict=True
    ):
        pose = pose @ local
        if joint_index is not None:
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(
                axis * joint_positions[joint_index]
            ).as_matrix()
            pose = pose @ motion
    return pose


def _fk_prefixes(local_transforms, axes, joint_indices, joint_positions):
    """Return Base_T_Link after every local origin and joint motion."""
    pose = np.eye(4)
    output = []
    for local, axis, joint_index in zip(
        local_transforms, axes, joint_indices, strict=True
    ):
        pose = pose @ local
        if joint_index is not None:
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(
                axis * joint_positions[joint_index]
            ).as_matrix()
            pose = pose @ motion
        output.append(pose.copy())
    return output


def _compare_chains(
    reference_transforms,
    candidate_transforms,
    axes,
    joint_indices,
    joint_positions,
):
    """Compare local transforms and every FK prefix over deterministic samples."""
    local_errors = np.asarray(
        [
            _pose_residual(candidate, reference)
            for reference, candidate in zip(
                reference_transforms, candidate_transforms, strict=True
            )
        ]
    )
    joints = (
        None if joint_positions is None else np.asarray(joint_positions, dtype=float)
    )
    if joints is None:
        samples = np.zeros((1, sum(index is not None for index in joint_indices)))
        random_count = 0
    else:
        if joints.ndim != 2 or len(joints) == 0:
            raise ValueError("joint_positions must have shape [N, DOF]")
        rng = np.random.default_rng(0)
        random = rng.uniform(
            joints.min(axis=0),
            joints.max(axis=0),
            size=(_FK_VERIFICATION_RANDOM_SAMPLES, joints.shape[1]),
        )
        samples = np.concatenate([joints, random])
        random_count = len(random)

    position_max = 0.0
    orientation_max = 0.0
    end_position_max = 0.0
    end_orientation_max = 0.0
    for q in samples:
        reference_prefixes = _fk_prefixes(reference_transforms, axes, joint_indices, q)
        candidate_prefixes = _fk_prefixes(candidate_transforms, axes, joint_indices, q)
        for reference, candidate in zip(
            reference_prefixes, candidate_prefixes, strict=True
        ):
            error = _pose_residual(candidate, reference)
            position_max = max(position_max, float(np.linalg.norm(error[:3])))
            orientation_max = max(orientation_max, float(np.linalg.norm(error[3:])))
        end_error = _pose_residual(candidate_prefixes[-1], reference_prefixes[-1])
        end_position_max = max(end_position_max, float(np.linalg.norm(end_error[:3])))
        end_orientation_max = max(
            end_orientation_max, float(np.linalg.norm(end_error[3:]))
        )
    return {
        "local_position_max_m": float(
            np.linalg.norm(local_errors[:, :3], axis=1).max()
        ),
        "local_orientation_max_rad": float(
            np.linalg.norm(local_errors[:, 3:], axis=1).max()
        ),
        "fk_all_links_position_max_m": position_max,
        "fk_all_links_orientation_max_rad": orientation_max,
        "fk_end_effector_position_max_m": end_position_max,
        "fk_end_effector_orientation_max_rad": end_orientation_max,
        "dataset_samples": 0 if joints is None else int(len(joints)),
        "random_samples": random_count,
        "total_samples": int(len(samples)),
        "random_seed": 0 if random_count else None,
        "random_domain": ("per-joint observed dataset range" if random_count else None),
    }


def _pose_residual(actual, expected) -> np.ndarray:
    delta = np.linalg.inv(expected) @ actual
    return np.concatenate(
        [delta[:3, 3], Rotation.from_matrix(delta[:3, :3]).as_rotvec()]
    )


def _transform_delta(reference, candidate):
    """Describe reference_T_candidate in the reference frame."""
    delta = np.linalg.inv(reference) @ candidate
    translation = delta[:3, 3]
    rotation_vector = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    return {
        "translation_m": translation.tolist(),
        "rotation_vector_rad": rotation_vector.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_angle_rad": float(np.linalg.norm(rotation_vector)),
    }


def _allocate_near_nominal(
    raw_transforms,
    nominal_transforms,
    axes,
    joint_indices,
    joint_positions,
    *,
    position_tolerance_m,
    orientation_tolerance_rad,
    anchor_index=None,
):
    joints = np.asarray(joint_positions, dtype=float)
    if joints.ndim != 2 or len(joints) == 0:
        raise ValueError("joint_positions must have shape [N, DOF]")
    sample_positions = [*joints]
    if len(joints) > 1:
        rng = np.random.default_rng(0)
        sample_positions.extend(
            rng.uniform(
                joints.min(axis=0), joints.max(axis=0), size=(24, joints.shape[1])
            )
        )
    sample_positions = np.asarray(sample_positions)
    targets = [_fk(raw_transforms, axes, joint_indices, q) for q in sample_positions]

    initial_blocks = []
    for raw, nominal in zip(raw_transforms, nominal_transforms, strict=True):
        translation = nominal[:3, :3].T @ (raw[:3, 3] - nominal[:3, 3])
        rotation = Rotation.from_matrix(nominal[:3, :3].T @ raw[:3, :3]).as_rotvec()
        initial_blocks.append(np.concatenate([translation, rotation]))
    initial = np.concatenate(initial_blocks)

    def transforms(parameters):
        output = []
        for index, nominal in enumerate(nominal_transforms):
            block = parameters[6 * index : 6 * index + 6]
            value = np.eye(4)
            value[:3, :3] = (
                nominal[:3, :3] @ Rotation.from_rotvec(block[3:]).as_matrix()
            )
            value[:3, 3] = nominal[:3, 3] + nominal[:3, :3] @ block[:3]
            output.append(value)
        return output

    optimization_count = min(len(sample_positions), 32)
    optimization_indices = np.linspace(
        0, len(sample_positions) - 1, optimization_count, dtype=int
    )

    def residual(parameters):
        candidate = transforms(parameters)
        fk_residuals = []
        for index in optimization_indices:
            error = _pose_residual(
                _fk(candidate, axes, joint_indices, sample_positions[index]),
                targets[index],
            )
            fk_residuals.extend(error[:3] / 1e-8)
            fk_residuals.extend(error[3:] / 1e-8)
        # In exact gauge-null directions this selects the closest nominal URDF.
        prior = parameters / np.tile(
            np.array([0.001, 0.001, 0.001, 0.001745, 0.001745, 0.001745]),
            len(nominal_transforms),
        )
        return np.concatenate([fk_residuals, prior])

    if anchor_index is not None:
        if not 0 <= anchor_index < len(nominal_transforms):
            raise ValueError("anchor_index is outside the calibrated chain")
        free_mask = np.ones(initial.size, dtype=bool)
        free_mask[6 * anchor_index : 6 * anchor_index + 6] = False
        initial_free = initial[free_mask]

        def anchored_residual(free_parameters):
            parameters = np.zeros_like(initial)
            parameters[free_mask] = free_parameters
            return residual(parameters)

        optimization_residual = anchored_residual
    else:
        free_mask = None
        initial_free = initial
        optimization_residual = residual

    result = least_squares(
        optimization_residual,
        initial_free,
        method="trf",
        max_nfev=100,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    optimized = np.zeros_like(initial) if free_mask is not None else result.x
    if free_mask is not None:
        optimized[free_mask] = result.x
    candidate = transforms(optimized)
    errors = np.asarray(
        [
            _pose_residual(_fk(candidate, axes, joint_indices, q), target)
            for q, target in zip(sample_positions, targets, strict=True)
        ]
    )
    position_errors = np.linalg.norm(errors[:, :3], axis=1)
    orientation_errors = np.linalg.norm(errors[:, 3:], axis=1)
    initial_norm = float(np.linalg.norm(initial))
    final_norm = float(np.linalg.norm(optimized))
    equivalent = bool(
        position_errors.max() <= position_tolerance_m
        and orientation_errors.max() <= orientation_tolerance_rad
    )
    improved = bool(final_norm < initial_norm - 1e-12)
    accepted = equivalent and improved and bool(result.success)
    return (candidate if accepted else raw_transforms), {
        "attempted": True,
        "accepted": accepted,
        "optimizer_success": bool(result.success),
        "fallback_to_raw_solver": not accepted,
        "nominal_distance_initial": initial_norm,
        "nominal_distance_final": final_norm,
        "fk_position_max_m": float(position_errors.max()),
        "fk_orientation_max_rad": float(orientation_errors.max()),
        "fk_position_tolerance_m": position_tolerance_m,
        "fk_orientation_tolerance_rad": orientation_tolerance_rad,
        "verification_samples": int(len(sample_positions)),
        "verification_domain": "dataset plus deterministic observed-range samples",
        "anchor_applied": anchor_index is not None,
    }


def _allocate_svd_nominal_gauge(
    raw_transforms,
    nominal_transforms,
    axes,
    joint_indices,
    joint_positions,
    *,
    position_tolerance_m,
    orientation_tolerance_rad,
    relative_singular_value_threshold=1e-8,
):
    """Project nominal-relative parameters only along the FK regressor nullspace."""
    joints = np.asarray(joint_positions, dtype=float)
    if joints.ndim != 2 or len(joints) == 0:
        raise ValueError("joint_positions must have shape [N, DOF]")
    rng = np.random.default_rng(0)
    extra = rng.uniform(
        joints.min(axis=0), joints.max(axis=0), size=(24, joints.shape[1])
    )
    samples = np.concatenate([joints, extra])
    indices = np.linspace(0, len(samples) - 1, min(len(samples), 32), dtype=int)
    optimization_samples = samples[indices]
    targets = [_fk(raw_transforms, axes, joint_indices, q) for q in samples]

    scales = np.tile(
        np.array([0.001, 0.001, 0.001, 0.001745, 0.001745, 0.001745]),
        len(nominal_transforms),
    )
    blocks = []
    for raw, nominal in zip(raw_transforms, nominal_transforms, strict=True):
        translation = nominal[:3, :3].T @ (raw[:3, 3] - nominal[:3, 3])
        rotation = Rotation.from_matrix(nominal[:3, :3].T @ raw[:3, :3]).as_rotvec()
        blocks.append(np.concatenate([translation, rotation]))
    raw_scaled = np.concatenate(blocks) / scales

    def transforms(scaled_parameters):
        physical = scaled_parameters * scales
        output = []
        for index, nominal in enumerate(nominal_transforms):
            block = physical[6 * index : 6 * index + 6]
            value = np.eye(4)
            value[:3, :3] = (
                nominal[:3, :3] @ Rotation.from_rotvec(block[3:]).as_matrix()
            )
            value[:3, 3] = nominal[:3, 3] + nominal[:3, :3] @ block[:3]
            output.append(value)
        return output

    optimization_targets = [
        _fk(raw_transforms, axes, joint_indices, q) for q in optimization_samples
    ]

    def residual(scaled_parameters):
        candidate = transforms(scaled_parameters)
        return np.concatenate(
            [
                _pose_residual(_fk(candidate, axes, joint_indices, q), target)
                for q, target in zip(
                    optimization_samples, optimization_targets, strict=True
                )
            ]
        )

    step = 1e-5
    regressor = np.empty((len(optimization_samples) * 6, raw_scaled.size))
    for column in range(raw_scaled.size):
        offset = np.zeros_like(raw_scaled)
        offset[column] = step
        regressor[:, column] = (
            residual(raw_scaled + offset) - residual(raw_scaled - offset)
        ) / (2.0 * step)

    _, singular_values, vh = np.linalg.svd(regressor, full_matrices=False)
    cutoff = (
        relative_singular_value_threshold * singular_values[0]
        if len(singular_values)
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > cutoff))
    gauge_basis = vh[rank:].T
    gauge_dimension = int(gauge_basis.shape[1])
    gauge_component = (
        gauge_basis @ (gauge_basis.T @ raw_scaled)
        if gauge_dimension
        else np.zeros_like(raw_scaled)
    )
    initial_norm = float(np.linalg.norm(raw_scaled))
    best = raw_transforms
    best_report = None
    for exponent in range(11):
        fraction = 0.5**exponent
        projected = raw_scaled - fraction * gauge_component
        candidate = transforms(projected)
        errors = np.asarray(
            [
                _pose_residual(_fk(candidate, axes, joint_indices, q), target)
                for q, target in zip(samples, targets, strict=True)
            ]
        )
        position_max = float(np.linalg.norm(errors[:, :3], axis=1).max())
        orientation_max = float(np.linalg.norm(errors[:, 3:], axis=1).max())
        final_norm = float(np.linalg.norm(projected))
        if (
            position_max <= position_tolerance_m
            and orientation_max <= orientation_tolerance_rad
            and final_norm < initial_norm - 1e-12
        ):
            best = candidate
            best_report = (fraction, final_norm, position_max, orientation_max)
            break

    accepted = best_report is not None and gauge_dimension > 0
    fraction, final_norm, position_max, orientation_max = best_report or (
        0.0,
        initial_norm,
        0.0,
        0.0,
    )
    return best, {
        "attempted": True,
        "accepted": accepted,
        "fallback_to_raw_solver": not accepted,
        "parameterization": "scaled-local-se3-about-nominal-urdf",
        "equivalence_scope": "end_effector_only",
        "regressor_rank": rank,
        "parameter_count": int(raw_scaled.size),
        "gauge_dimension": gauge_dimension,
        "relative_singular_value_threshold": relative_singular_value_threshold,
        "singular_values": singular_values.tolist(),
        "translation_scale_m": 0.001,
        "rotation_scale_rad": 0.001745,
        "gauge_projection_fraction": fraction,
        "nominal_distance_initial": initial_norm,
        "nominal_distance_final": final_norm,
        "fk_position_max_m": position_max,
        "fk_orientation_max_rad": orientation_max,
        "fk_position_tolerance_m": position_tolerance_m,
        "fk_orientation_tolerance_rad": orientation_tolerance_rad,
        "verification_samples": int(len(samples)),
        "verification_domain": "dataset plus 24 deterministic observed-range samples",
    }


def postprocess_urdf_definitions(
    definitions,
    source_urdf: Path | str,
    joint_positions=None,
    *,
    position_tolerance_m: float = 1e-6,
    orientation_tolerance_rad: float = 1e-6,
    anchor_joint: str | None = None,
    enable_nominal_allocation: bool = False,
    enable_svd_gauge_allocation: bool = False,
):
    """Select stable RPY branches and optionally evaluate one nominal anchor.

    By default only equivalent RPY branches are normalized and raw solver
    transforms are retained. ``enable_nominal_allocation`` explicitly enables
    experimental gauge-null reallocation toward the nominal URDF.
    """
    root = ET.parse(source_urdf).getroot()
    nominal = {}
    source_joints = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        relation = f"{parent.attrib.get('link')}-{child.attrib.get('link')}"
        origin = joint.find("origin")
        nominal[relation] = {
            "rpy": _vector(origin, "rpy", "0 0 0"),
            "axis": _vector(joint.find("axis"), "xyz", "1 0 0"),
            "joint_name": joint.attrib.get("name"),
            "joint_type": joint.attrib.get("type"),
        }
        source_joints[relation] = joint

    prepared = []
    nominal_transforms = []
    solver_transforms = []
    raw_transforms = []
    chain_axes = []
    joint_indices = []
    revolute_index = 0
    anchor_index = None
    for definition in definitions:
        item = dict(definition)
        relation = item["name"]
        metadata = nominal.get(relation)
        if metadata is None:
            raise ValueError(f"calibrated relation {relation!r} is absent from URDF")
        solver_transforms.append(_transform(item["xyz"], item["rpy"]))
        item["rpy"] = _nearest_equivalent_rpy(item["rpy"], metadata["rpy"]).tolist()
        prepared.append(item)
        if anchor_joint in (relation, metadata["joint_name"]):
            if anchor_index is not None:
                raise ValueError(f"anchor joint {anchor_joint!r} is ambiguous")
            anchor_index = len(prepared) - 1
        raw_transforms.append(_transform(item["xyz"], item["rpy"]))
        origin = source_joints[relation].find("origin")
        nominal_transforms.append(
            _transform(
                _vector(origin, "xyz", "0 0 0"),
                _vector(origin, "rpy", "0 0 0"),
            )
        )
        chain_axes.append(metadata["axis"] / np.linalg.norm(metadata["axis"]))
        if metadata["joint_type"] in ("revolute", "continuous"):
            joint_indices.append(revolute_index)
            revolute_index += 1
        else:
            joint_indices.append(None)

    if enable_nominal_allocation and enable_svd_gauge_allocation:
        raise ValueError("nominal allocation modes are mutually exclusive")
    if anchor_joint is not None and not enable_nominal_allocation:
        raise ValueError("anchor_joint requires enable_nominal_allocation=True")
    if anchor_joint is not None and anchor_index is None:
        raise ValueError(
            f"anchor joint {anchor_joint!r} is absent from the calibrated chain"
        )

    allocation = {
        "attempted": False,
        "accepted": False,
        "fallback_to_raw_solver": True,
        "reason": "joint positions were not supplied",
    }
    selected_transforms = raw_transforms
    if enable_svd_gauge_allocation and joint_positions is not None:
        selected_transforms, allocation = _allocate_svd_nominal_gauge(
            raw_transforms,
            nominal_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )
        allocation["selection_mode"] = "svd_identifiable_nominal_gauge"
    elif enable_nominal_allocation and joint_positions is not None:
        selected_transforms, allocation = _allocate_near_nominal(
            raw_transforms,
            nominal_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            anchor_index=anchor_index,
        )
        allocation["selection_mode"] = (
            "explicit_anchor" if anchor_joint is not None else "unanchored"
        )
    elif not enable_nominal_allocation and not enable_svd_gauge_allocation:
        allocation["reason"] = "nominal allocation is disabled by default"
        allocation["selection_mode"] = "raw_solver"
    allocation["anchor_joint"] = anchor_joint
    allocation["anchor_constraint"] = (
        "xyz/rpy fixed to nominal URDF" if anchor_joint is not None else None
    )

    processed = []
    global_transform = np.eye(4)
    axes = []
    for item, local in zip(prepared, selected_transforms, strict=True):
        item = dict(item)
        relation = item["name"]
        metadata = nominal.get(relation)
        item["xyz"] = local[:3, 3].tolist()
        item["rpy"] = _nearest_equivalent_rpy(
            Rotation.from_matrix(local[:3, :3]).as_euler("xyz"),
            metadata["rpy"],
        ).tolist()
        global_transform = global_transform @ local
        if metadata["joint_type"] in ("revolute", "continuous"):
            direction = global_transform[:3, :3] @ metadata["axis"]
            direction /= np.linalg.norm(direction)
            axes.append(
                {
                    "joint_name": metadata["joint_name"],
                    "point": global_transform[:3, 3].copy(),
                    "direction": direction,
                }
            )
        processed.append(item)

    processed_transforms = [_transform(item["xyz"], item["rpy"]) for item in processed]
    raw_equivalence = _compare_chains(
        solver_transforms,
        processed_transforms,
        chain_axes,
        joint_indices,
        joint_positions,
    )
    nominal_comparison = _compare_chains(
        nominal_transforms,
        processed_transforms,
        chain_axes,
        joint_indices,
        joint_positions,
    )
    representation_position_tolerance_m = 1e-12
    representation_orientation_tolerance_rad = 1e-9
    if enable_svd_gauge_allocation:
        raw_equivalent = bool(
            raw_equivalence["fk_end_effector_position_max_m"] <= position_tolerance_m
            and raw_equivalence["fk_end_effector_orientation_max_rad"]
            <= orientation_tolerance_rad
        )
    else:
        raw_equivalent = bool(
            raw_equivalence["local_position_max_m"]
            <= representation_position_tolerance_m
            and raw_equivalence["local_orientation_max_rad"]
            <= representation_orientation_tolerance_rad
            and raw_equivalence["fk_all_links_position_max_m"]
            <= representation_position_tolerance_m
            and raw_equivalence["fk_all_links_orientation_max_rad"]
            <= representation_orientation_tolerance_rad
        )
    rpy_fallback = False
    if (
        not enable_nominal_allocation
        and not enable_svd_gauge_allocation
        and not raw_equivalent
    ):
        processed = [dict(definition) for definition in definitions]
        processed_transforms = solver_transforms
        raw_equivalence = _compare_chains(
            solver_transforms,
            processed_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
        )
        nominal_comparison = _compare_chains(
            nominal_transforms,
            processed_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
        )
        raw_equivalent = True
        rpy_fallback = True
    elif enable_svd_gauge_allocation and not raw_equivalent:
        processed = [dict(definition) for definition in definitions]
        processed_transforms = solver_transforms
        allocation["accepted"] = False
        allocation["fallback_to_raw_solver"] = True
        allocation["reason"] = "2048-sample end-effector FK verification failed"
        raw_equivalence = _compare_chains(
            solver_transforms,
            processed_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
        )
        nominal_comparison = _compare_chains(
            nominal_transforms,
            processed_transforms,
            chain_axes,
            joint_indices,
            joint_positions,
        )
        raw_equivalent = True

    joint_parameter_comparison = []
    parameter_rows = zip(
        prepared,
        nominal_transforms,
        solver_transforms,
        processed_transforms,
        strict=True,
    )
    for index, (
        item,
        nominal_transform,
        solver_transform,
        processed_transform,
    ) in enumerate(parameter_rows):
        metadata = nominal[item["name"]]
        joint_parameter_comparison.append(
            {
                "chain_index": index,
                "joint_name": metadata["joint_name"],
                "relation": item["name"],
                "nominal_to_raw_solver": _transform_delta(
                    nominal_transform, solver_transform
                ),
                "nominal_to_selected": _transform_delta(
                    nominal_transform, processed_transform
                ),
                "raw_solver_to_selected": _transform_delta(
                    solver_transform, processed_transform
                ),
            }
        )

    adjacent_parameter_coupling = []
    for first, second in zip(
        joint_parameter_comparison, joint_parameter_comparison[1:]
    ):
        adjacent_parameter_coupling.append(
            {
                "joint_pair": [first["joint_name"], second["joint_name"]],
                "note": (
                    "Adjacent grouping is descriptive; SVD gauge vectors are "
                    "global and do not have a unique pairwise attribution."
                ),
                "first_joint_raw_solver_to_selected": first["raw_solver_to_selected"],
                "second_joint_raw_solver_to_selected": second["raw_solver_to_selected"],
                "first_joint_nominal_to_selected": first["nominal_to_selected"],
                "second_joint_nominal_to_selected": second["nominal_to_selected"],
            }
        )

    relations = []
    parallel_tolerance_rad = 1e-3
    intersection_tolerance_m = 1e-5
    for first, second in zip(axes, axes[1:]):
        direction_cross = np.cross(first["direction"], second["direction"])
        cross_norm = float(np.linalg.norm(direction_cross))
        angle = float(
            np.arccos(
                np.clip(abs(np.dot(first["direction"], second["direction"])), 0, 1)
            )
        )
        displacement = second["point"] - first["point"]
        if angle <= parallel_tolerance_rad:
            relation_type = "parallel"
            distance = float(np.linalg.norm(np.cross(displacement, first["direction"])))
            gauge_dependent = ["origin_along_joint_axis", "frame_rotation_about_axis"]
        else:
            distance = float(abs(np.dot(displacement, direction_cross)) / cross_norm)
            relation_type = (
                "intersecting" if distance <= intersection_tolerance_m else "skew"
            )
            gauge_dependent = ["frame_rotation_about_axis"]
        relations.append(
            {
                "joint_pair": [first["joint_name"], second["joint_name"]],
                "relation": relation_type,
                "axis_angle_rad": angle,
                "axis_distance_m": distance,
                "gauge_dependent": gauge_dependent,
            }
        )

    report = {
        "allocation_policy": (
            "svd-identifiable-nominal-gauge"
            if enable_svd_gauge_allocation
            else (
                "nominal-constrained-se3-explicit-anchor"
                if anchor_joint is not None
                else (
                    "nominal-constrained-se3"
                    if enable_nominal_allocation
                    else "raw-solver-rpy-normalized"
                )
            )
        ),
        "transform_preserving": raw_equivalent,
        "physically_unique": False,
        "note": (
            "Raw solver transforms are retained by default; only an equivalent "
            "RPY branch nearest the nominal representation is selected. Nominal "
            "SE(3) allocation is experimental and must be explicitly enabled."
        ),
        "adjacent_axis_relations": relations,
        "allocation": allocation,
        "joint_parameter_comparison": joint_parameter_comparison,
        "adjacent_parameter_coupling": adjacent_parameter_coupling,
        "equivalence_verification": {
            "reference": "raw_solver",
            "verified": raw_equivalent,
            "scope": (
                "Base_T_EE only; intermediate link frames may change"
                if enable_svd_gauge_allocation
                else "every local transform and Base_T_Link prefix"
            ),
            "position_tolerance_m": (
                position_tolerance_m
                if enable_svd_gauge_allocation
                else representation_position_tolerance_m
            ),
            "orientation_tolerance_rad": (
                orientation_tolerance_rad
                if enable_svd_gauge_allocation
                else representation_orientation_tolerance_rad
            ),
            "rpy_normalization_fallback": rpy_fallback,
            **raw_equivalence,
        },
        "nominal_urdf_comparison": {
            "purpose": "calibration-change magnitude; not an equivalence gate",
            **nominal_comparison,
        },
        "selected_parameter_source": (
            "svd_identifiable_nominal_gauge"
            if enable_svd_gauge_allocation and allocation.get("accepted")
            else "nominal_constrained_allocation"
            if allocation.get("accepted")
            else "raw_solver"
        ),
    }
    return processed, report
