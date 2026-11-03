"""Model-based pose selection using the same POE observation regressor.

This module proposes joint configurations only. It acquires no measurements and
does not establish camera visibility, reachability, or collision safety.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends.protocol import KinematicsBackend


@dataclass(frozen=True)
class PoseSelection:
    joint_positions: np.ndarray
    candidate_indices: np.ndarray
    singular_values: np.ndarray
    rank: int
    parameter_count: int
    condition_number: float | None
    characteristic_length_m: float


@dataclass(frozen=True)
class PosePlan:
    """Saved joint configurations in visit order, with their original identities.

    ``visit_order`` maps each saved row to its selection-order row. Recover
    selection order with ``planned_joint_positions[np.argsort(visit_order)]``.
    Optional ``source_sample_indices`` maps every source row to its original
    sample ID; index it with ``candidate_indices`` to recover visit sample IDs.
    Loading a plan establishes neither independent measurements nor motion safety.
    """

    planned_joint_positions: np.ndarray
    candidate_indices: np.ndarray
    selected_candidate_indices: np.ndarray
    visit_order: np.ndarray
    segment_distances_rad: np.ndarray
    joint_names: tuple[str, ...]
    joint_limits: np.ndarray
    singular_values: np.ndarray
    metadata: dict
    source_sample_indices: np.ndarray | None = None


def _positive_scalar(value, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _sha256_metadata(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"pose plan {name} must be a SHA-256 hex digest")
    return value


def load_pose_plan(
    path: str | Path,
    *,
    backend: KinematicsBackend | None = None,
    source_urdf_sha256: str | None = None,
) -> PosePlan:
    """Read a schema-2 plan without pickle and check its saved identities.

    Rows retain their saved visit order. Optional model checks compare joint
    columns, limits and exposed frame names; a supplied URDF hash must match.
    Checks cover stored distances and diagnostics, without repeating selection,
    route optimization, collision checking or the observation regressor.
    """
    import json

    from kinematic_calib.dataset import validated_joint_names, validated_sample_indices

    from .routing import _configurations

    keys = (
        "planned_joint_positions",
        "candidate_indices",
        "selected_candidate_indices",
        "visit_order",
        "segment_distances_rad",
        "joint_names",
        "joint_limits",
        "singular_values",
        "metadata_json",
    )
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("pose plan must be an NPZ archive")
    with archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("pose plan is missing required arrays")
        arrays = {key: archive[key] for key in keys}
        if "source_sample_indices" in archive.files:
            arrays["source_sample_indices"] = archive["source_sample_indices"]
    raw_metadata = arrays.pop("metadata_json")
    if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
        raise ValueError("pose plan metadata_json must be a scalar string")
    metadata = json.loads(raw_metadata.item())
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 2
        or metadata.get("artifact_type") != "planned_joint_configurations"
    ):
        raise ValueError("unsupported pose plan schema or artifact type")

    def integer(name, minimum=1):
        value = metadata.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"pose plan {name} must be an integer >= {minimum}")
        return value

    positions = _configurations(
        arrays["planned_joint_positions"], name="planned_joint_positions"
    )
    count, dof = positions.shape
    if integer("selected_count") != count:
        raise ValueError("pose plan selected_count does not match saved rows")
    candidate_count = integer("candidate_count")
    integer("seed", 0)
    names = validated_joint_names(arrays["joint_names"], dof)
    if metadata.get("joint_names") != list(names):
        raise ValueError("pose plan metadata joint_names do not match saved columns")
    limits = _configurations(
        arrays["joint_limits"], name="joint_limits", shape=(dof, 2)
    )
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("pose plan joint_limits must be ordered")
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise ValueError("planned_joint_positions must respect joint limits")

    for key, upper in (
        ("candidate_indices", candidate_count),
        ("selected_candidate_indices", candidate_count),
        ("visit_order", count),
    ):
        values = arrays[key]
        if (
            values.dtype.kind not in "iu"
            or values.shape != (count,)
            or np.any(values < 0)
            or np.any(values >= upper)
            or len(np.unique(values)) != count
        ):
            raise ValueError(
                f"pose plan {key} must contain unique in-range row indices"
            )
    visit_order = arrays["visit_order"]
    if not np.array_equal(
        arrays["candidate_indices"], arrays["selected_candidate_indices"][visit_order]
    ):
        raise ValueError("pose plan candidate_indices do not match visit_order")

    source_sample_indices = None
    if "source_sample_indices" in arrays:
        source_sample_indices = validated_sample_indices(
            arrays["source_sample_indices"],
            candidate_count,
            label="source_sample_indices",
        )
    if "selection_method" in metadata:
        if metadata["selection_method"] not in ("all-input", "d-optimal"):
            raise ValueError("unsupported pose plan selection_method")
        if metadata["selection_method"] == "all-input" and (
            candidate_count != count
            or not np.array_equal(
                arrays["selected_candidate_indices"], np.arange(count)
            )
        ):
            raise ValueError(
                "all-input pose plan must retain every source row in order"
            )
    if (
        "generation_seed_used" in metadata
        and type(metadata["generation_seed_used"]) is not bool
    ):
        raise ValueError("pose plan generation_seed_used must be a boolean")
    provenance_keys = (
        "input_joints_sha256",
        "input_joints_format",
        "input_joint_identity_verified",
        "input_joint_names",
    )
    if (
        any(key in metadata for key in provenance_keys)
        or source_sample_indices is not None
        or metadata.get("generation_seed_used") is False
        or metadata.get("selection_method") == "all-input"
    ):
        if not all(key in metadata for key in provenance_keys):
            raise ValueError("pose plan is missing input joint provenance")
        _sha256_metadata(metadata["input_joints_sha256"], "input_joints_sha256")
        if metadata["input_joints_format"] not in ("npy", "npz"):
            raise ValueError("pose plan input_joints_format must be npy or npz")
        verified = metadata["input_joint_identity_verified"]
        if type(verified) is not bool:
            raise ValueError(
                "pose plan input_joint_identity_verified must be a boolean"
            )
        input_names = metadata["input_joint_names"]
        if input_names is not None:
            if validated_joint_names(input_names, dof) != names:
                raise ValueError(
                    "pose plan input_joint_names do not match model columns"
                )
        if verified != (input_names is not None):
            raise ValueError(
                "pose plan input joint identity verification is inconsistent"
            )
        if metadata["input_joints_format"] == "npy" and (
            input_names is not None or source_sample_indices is not None
        ):
            raise ValueError("NPY input cannot provide joint names or sample indices")
        if metadata.get("generation_seed_used") is not False:
            raise ValueError(
                "supplied input pose plan generation_seed_used must be false"
            )
        if metadata.get("selection_method") not in ("all-input", "d-optimal"):
            raise ValueError("supplied input pose plan requires selection_method")

    for key in (
        "measurements_acquired",
        "collision_checked",
        "visibility_checked",
        "trajectory_generated",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"pose plan {key} must be false")
    for key in ("base_link", "end_link"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"pose plan {key} must be a nonempty string")
    source_hash = _sha256_metadata(
        metadata.get("source_urdf_sha256"), "source_urdf_sha256"
    )
    if source_urdf_sha256 is not None and source_urdf_sha256 != source_hash:
        raise ValueError("pose plan source URDF SHA-256 does not match model")
    if backend is not None:
        if (
            backend.dof != dof
            or validated_joint_names(backend.joint_names, dof) != names
        ):
            raise ValueError("pose plan joint_names or DOF do not match backend")
        if not np.array_equal(backend.joint_limits, limits):
            raise ValueError("pose plan joint_limits do not match backend")
        for key in ("base_link", "end_link"):
            if hasattr(backend, key) and getattr(backend, key) != metadata[key]:
                raise ValueError(f"pose plan {key} does not match backend")

    if metadata.get("distance_metric") != "euclidean_joint_angles_without_wrapping":
        raise ValueError("unsupported pose plan distance_metric")
    if (
        not isinstance(metadata.get("visit_order_method"), str)
        or not metadata["visit_order_method"].strip()
    ):
        raise ValueError("pose plan visit_order_method must be a nonempty string")
    if type(metadata.get("return_to_start")) is not bool:
        raise ValueError("pose plan return_to_start must be a boolean")
    if "start_configuration_rad" not in metadata:
        raise ValueError("pose plan is missing start_configuration_rad")
    start = metadata["start_configuration_rad"]
    closed = metadata["return_to_start"]
    if start is None:
        if closed:
            raise ValueError("return_to_start requires start_configuration_rad")
        if visit_order[0] != 0:
            raise ValueError(
                "pose plan without a start must visit selection row zero first"
            )
    else:
        start = _configurations(start, name="start_configuration_rad", shape=(dof,))
        if np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
            raise ValueError("start_configuration_rad must respect joint limits")

    def segments(rows):
        if start is not None:
            rows = (
                np.vstack([start, rows, start]) if closed else np.vstack([start, rows])
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.hypot.reduce(np.diff(rows, axis=0), axis=1)
        if not np.isfinite(result).all():
            raise ValueError(
                "pose plan joint-space distances exceed the finite numeric range"
            )
        return result

    expected_segments = segments(positions)
    distances = _configurations(
        arrays["segment_distances_rad"],
        name="segment_distances_rad",
        shape=expected_segments.shape,
    )
    if np.any(distances < 0) or not np.allclose(
        distances, expected_segments, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("pose plan segment_distances_rad do not match saved rows")
    for key, values in (
        ("total_distance_rad", expected_segments),
        ("selection_order_distance_rad", segments(positions[np.argsort(visit_order)])),
    ):
        value = metadata.get(key)
        with np.errstate(over="ignore"):
            expected = float(values.sum())
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or value < 0
            or not np.isfinite(expected)
            or not np.isclose(value, expected, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(f"pose plan {key} does not match saved rows")

    parameter_count = integer("parameter_count")
    if parameter_count != 4 * dof + 6:
        raise ValueError("pose plan parameter_count does not match DOF")
    singular_values = _configurations(
        arrays["singular_values"],
        name="singular_values",
        shape=(min(6 * count, parameter_count),),
    )
    if np.any(singular_values < 0) or np.any(np.diff(singular_values) > 0):
        raise ValueError("pose plan singular_values must be nonnegative and descending")
    rank_tolerance = (
        singular_values[0] * max(6 * count, parameter_count) * np.finfo(float).eps
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    full_rank = rank == parameter_count
    if (
        integer("regressor_rank", 0) != rank
        or metadata.get("full_rank") is not full_rank
    ):
        raise ValueError("pose plan rank diagnostics do not match singular_values")
    if "condition_number" not in metadata:
        raise ValueError("pose plan is missing condition_number")
    condition = metadata["condition_number"]
    if full_rank:
        if (
            type(condition) not in (int, float)
            or not np.isfinite(condition)
            or not np.isclose(
                condition, singular_values[0] / singular_values[-1], rtol=1e-9
            )
        ):
            raise ValueError(
                "pose plan condition_number does not match singular_values"
            )
    elif condition is not None:
        raise ValueError("rank-deficient pose plan condition_number must be null")
    _positive_scalar(metadata.get("characteristic_length_m"), "characteristic_length_m")
    return PosePlan(
        planned_joint_positions=positions,
        candidate_indices=arrays["candidate_indices"],
        selected_candidate_indices=arrays["selected_candidate_indices"],
        visit_order=visit_order,
        segment_distances_rad=distances,
        joint_names=names,
        joint_limits=limits,
        singular_values=singular_values,
        metadata=metadata,
        source_sample_indices=source_sample_indices,
    )


def _candidate_configurations(backend, candidates):
    from .routing import _configurations

    configurations = _configurations(candidates, name="candidates")
    if configurations.shape[1] != backend.dof:
        raise ValueError("candidates must have shape [N,DOF] matching the backend")
    limits = np.asarray(backend.joint_limits)
    if (
        limits.dtype.kind not in "iuf"
        or limits.shape != (backend.dof, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("backend must provide ordered finite [DOF,2] joint limits")
    if np.any(configurations < limits[:, 0]) or np.any(configurations > limits[:, 1]):
        raise ValueError("candidate configurations must respect backend joint limits")
    return configurations


def _selection_count(count, candidate_count):
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or not 1 <= count <= candidate_count
    ):
        raise ValueError("count must be an integer between 1 and the candidate count")
    return int(count)


def _regressor_blocks(backend, configurations, characteristic_length_m):
    from spatialmath import SE3

    from robotkinecal import SerialRobotKineCal

    poe_parameters = getattr(backend, "poe_parameters", None)
    if not callable(poe_parameters):
        raise ValueError("pose selection requires a backend providing poe_parameters()")
    axes, home = poe_parameters()
    calibrator = SerialRobotKineCal.from_screw_axes(axes, home)
    # One predicted pose selects the 6D regressor parameterization. It is only
    # model initialization for planning; no predicted pose becomes a measurement.
    first_pose = calibrator.forward_kinematics_batch(configurations[:1])[0]
    calibrator.set_observations(configurations[:1], [SE3(first_pose, check=False)])
    regressor_blocks = calibrator.regressor_blocks(configurations)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        regressor_blocks[:, :3] /= characteristic_length_m
    if not np.isfinite(regressor_blocks).all():
        raise ValueError("characteristic length produces non-finite regressor rows")
    return regressor_blocks


def _pose_selection(configurations, selected_indices, blocks, characteristic_length_m):
    parameter_count = blocks.shape[2]
    selected_matrix = blocks[selected_indices].reshape(-1, parameter_count)
    singular_values = np.linalg.svd(selected_matrix, compute_uv=False)
    rank_tolerance = (
        singular_values[0]
        * max(selected_matrix.shape)
        * np.finfo(singular_values.dtype).eps
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == parameter_count
        else None
    )
    return PoseSelection(
        joint_positions=configurations[selected_indices].copy(),
        candidate_indices=selected_indices,
        singular_values=singular_values,
        rank=rank,
        parameter_count=parameter_count,
        condition_number=condition,
        characteristic_length_m=characteristic_length_m,
    )


def select_informative_poses(
    backend: KinematicsBackend,
    candidates: np.ndarray,
    *,
    count: int,
    characteristic_length_m: float = 1.0,
    regularization: float = 1e-6,
) -> PoseSelection:
    """Greedy D-optimal selection, scaled to dimensionless pose residuals.

    Translation rows are divided by ``characteristic_length_m``; rotation rows
    remain radians. The selection favors new observable directions, with
    Tikhonov regularization only for the greedy score. The reported rank and
    condition use the *unregularized* stacked selected regressor. Returned rows
    follow selection order; visit-order planning can reorder them separately.
    """
    configurations = _candidate_configurations(backend, candidates)
    count = _selection_count(count, len(configurations))
    characteristic_length_m = _positive_scalar(
        characteristic_length_m, "characteristic_length_m"
    )
    regularization = _positive_scalar(regularization, "regularization")
    regressor_blocks = _regressor_blocks(
        backend, configurations, characteristic_length_m
    )

    parameter_count = regressor_blocks.shape[2]
    parameter_identity = np.eye(parameter_count)
    information = regularization * parameter_identity
    with np.errstate(over="ignore"):
        information_inverse = parameter_identity / regularization
    if not np.isfinite(information_inverse).all():
        raise ValueError("regularization is too small for finite information scores")
    remaining = np.ones(len(configurations), dtype=bool)
    selected_indices = []
    for _ in range(count):
        candidate_indices = np.flatnonzero(remaining)
        available_blocks = regressor_blocks[candidate_indices]
        score_matrices = np.eye(6) + (
            available_blocks @ information_inverse @ available_blocks.transpose(0, 2, 1)
        )
        signs, scores = np.linalg.slogdet(score_matrices)
        if np.any(signs <= 0) or not np.isfinite(scores).all():
            raise ValueError("pose information score became numerically singular")
        selected_index = int(candidate_indices[np.argmax(scores)])
        selected_indices.append(selected_index)
        remaining[selected_index] = False
        selected_block = regressor_blocks[selected_index]
        information += selected_block.T @ selected_block
        # Re-factor the small information matrix each round instead of updating
        # its inverse recursively, which can accumulate numerical drift.
        information_inverse = np.linalg.solve(information, parameter_identity)
        information_inverse = (information_inverse + information_inverse.T) / 2

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    return _pose_selection(
        configurations, selected_indices, regressor_blocks, characteristic_length_m
    )


def save_pose_plan(
    output: str | Path,
    *,
    urdf: str | Path | None = None,
    end_link: str = "ee_link",
    base_link: str | None = None,
    count: int | None = None,
    candidates: int = 400,
    seed: int = 42,
    input_joints: str | Path | None = None,
    characteristic_length_m: float = 1.0,
    start_configuration: np.ndarray | None = None,
    return_to_start: bool = False,
    force: bool = False,
) -> dict:
    """Select calibration poses and save them in joint-space visit order.

    Without ``input_joints``, generate ``candidates`` proposals and select 40
    poses by default. Supplied NPY arrays or NPZ ``joint_positions`` contain
    radians in selected URDF joint order; optional NPZ ``joint_names`` must
    match that order exactly. Unnamed inputs retain unverified column identity.
    NPZ ``sample_indices`` are preserved as source identities; observations
    are never read. All supplied rows are visited unless ``count`` requests
    a smaller D-optimal subset. Keeping all rows skips greedy selection.

    With no explicit start, the first selected pose starts the open route.
    ``return_to_start`` requires an explicit start configuration. Distances
    are Euclidean joint-angle distances in radians, without angle wrapping.
    This produces a visit order, with no collision checks or timed trajectory.
    """
    import hashlib
    import io
    import json

    from kinematic_calib.dataset import validated_joint_names, validated_sample_indices
    from kinematic_calib.io_utils import atomic_savez_compressed

    from .backends.numpy import NumpyBackend
    from .experiment import example_urdf
    from .routing import plan_visit_order

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(candidates, int)
        or isinstance(candidates, bool)
        or candidates < 1
    ):
        raise ValueError("candidates must be a positive integer")
    output = Path(output)
    input_path = Path(input_joints) if input_joints is not None else None
    if input_path is not None and (
        output.resolve() == input_path.resolve()
        or (output.exists() and input_path.exists() and output.samefile(input_path))
    ):
        raise ValueError("plan output cannot replace its input joint configurations")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError("plan output must be a regular file")
    if output.exists() and not force:
        raise FileExistsError(
            f"pose plan already exists: {output}; use --force to replace"
        )
    source = Path(urdf) if urdf is not None else example_urdf()
    if output.resolve() == source.resolve():
        raise ValueError("plan output cannot replace its source URDF")
    backend = NumpyBackend.from_urdf(source, end_link=end_link, base_link=base_link)
    limits = backend.joint_limits
    if start_configuration is not None:
        start = np.asarray(start_configuration)
        if (
            start.dtype.kind not in "iuf"
            or start.shape != (backend.dof,)
            or not np.isfinite(start).all()
        ):
            raise ValueError(
                f"start_configuration must be a finite real ({backend.dof},) vector"
            )
        if np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
            raise ValueError("start_configuration must respect backend joint limits")
    if not isinstance(return_to_start, bool):
        raise ValueError("return_to_start must be a boolean")
    if return_to_start and start_configuration is None:
        raise ValueError("return_to_start requires start_configuration")
    source_sample_indices = None
    provenance = {}
    if input_path is None:
        lower, upper = np.maximum(limits[:, 0], -np.pi), np.minimum(limits[:, 1], np.pi)
        empty = lower >= upper
        lower[empty], upper[empty] = limits[empty, 0], limits[empty, 1]
        proposals = np.random.default_rng(seed).uniform(
            lower, upper, (candidates, backend.dof)
        )
        count = 40 if count is None else count
    else:
        # Hash the exact bytes parsed, retaining portable provenance without
        # requiring the original file when this plan is loaded elsewhere.
        input_bytes = input_path.read_bytes()
        archive = np.load(io.BytesIO(input_bytes), allow_pickle=False)
        input_names = None
        input_format = "npy"
        if isinstance(archive, np.lib.npyio.NpzFile):
            input_format = "npz"
            with archive:
                if "joint_positions" not in archive.files:
                    raise ValueError("input joint NPZ is missing joint_positions")
                proposals = _candidate_configurations(
                    backend, archive["joint_positions"]
                )
                if "joint_names" in archive.files:
                    input_names = validated_joint_names(
                        archive["joint_names"], backend.dof
                    )
                    if input_names != tuple(backend.joint_names):
                        raise ValueError(
                            "input joint_names order does not match model; "
                            "columns are never automatically reordered"
                        )
                if "sample_indices" in archive.files:
                    source_sample_indices = validated_sample_indices(
                        archive["sample_indices"], len(proposals)
                    )
        else:
            proposals = _candidate_configurations(backend, archive)
        candidates = len(proposals)
        count = candidates if count is None else count
        provenance = {
            "input_joints_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "input_joints_format": input_format,
            "input_joint_identity_verified": input_names is not None,
            "input_joint_names": list(input_names) if input_names is not None else None,
        }
    count = _selection_count(count, candidates)
    selection_method = "d-optimal"
    if input_path is not None and count == candidates:
        selection_method = "all-input"
        characteristic_length_m = _positive_scalar(
            characteristic_length_m, "characteristic_length_m"
        )
        blocks = _regressor_blocks(backend, proposals, characteristic_length_m)
        selected = _pose_selection(
            proposals,
            np.arange(candidates, dtype=np.int64),
            blocks,
            characteristic_length_m,
        )
    else:
        selected = select_informative_poses(
            backend,
            proposals,
            count=count,
            characteristic_length_m=characteristic_length_m,
        )
    route = plan_visit_order(
        selected.joint_positions,
        start_configuration=start_configuration,
        return_to_start=return_to_start,
    )
    report = {
        "schema_version": 2,
        "artifact_type": "planned_joint_configurations",
        "measurements_acquired": False,
        "collision_checked": False,
        "visibility_checked": False,
        "trajectory_generated": False,
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "joint_names": list(backend.joint_names),
        "base_link": backend.base_link,
        "end_link": end_link,
        "seed": seed,
        "generation_seed_used": input_path is None,
        "selection_method": selection_method,
        "candidate_count": candidates,
        "selected_count": len(selected.joint_positions),
        "regressor_rank": selected.rank,
        "parameter_count": selected.parameter_count,
        "full_rank": selected.rank == selected.parameter_count,
        "condition_number": selected.condition_number,
        "characteristic_length_m": selected.characteristic_length_m,
        "visit_order_method": "nearest-neighbor-2opt",
        "distance_metric": "euclidean_joint_angles_without_wrapping",
        "start_configuration_rad": (
            None
            if start_configuration is None
            else np.asarray(start_configuration, dtype=float).tolist()
        ),
        "return_to_start": return_to_start,
        "total_distance_rad": route.total_distance_rad,
        "selection_order_distance_rad": route.initial_distance_rad,
        **provenance,
    }
    source_arrays = (
        {"source_sample_indices": source_sample_indices}
        if source_sample_indices is not None
        else {}
    )
    atomic_savez_compressed(
        output,
        planned_joint_positions=selected.joint_positions[route.indices],
        candidate_indices=selected.candidate_indices[route.indices],
        selected_candidate_indices=selected.candidate_indices,
        visit_order=route.indices,
        segment_distances_rad=route.segment_distances_rad,
        joint_names=np.asarray(backend.joint_names),
        joint_limits=limits,
        singular_values=selected.singular_values,
        metadata_json=np.asarray(json.dumps(report, allow_nan=False)),
        **source_arrays,
    )
    return report
