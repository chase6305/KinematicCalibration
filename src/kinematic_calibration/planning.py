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


def _positive_scalar(value, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


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
    from spatialmath import SE3

    from robotkinecal import SerialRobotKineCal

    raw_candidates = np.asarray(candidates)
    if (
        raw_candidates.dtype.kind not in "iuf"
        or raw_candidates.ndim != 2
        or raw_candidates.shape[1] != backend.dof
        or not len(raw_candidates)
        or not np.isfinite(raw_candidates).all()
    ):
        raise ValueError("candidates must be finite real [N,DOF] joint configurations")
    configurations = raw_candidates.astype(float, copy=False)
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or not 1 <= count <= len(configurations)
    ):
        raise ValueError("count must be an integer between 1 and the candidate count")
    characteristic_length_m = _positive_scalar(
        characteristic_length_m, "characteristic_length_m"
    )
    regularization = _positive_scalar(regularization, "regularization")
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
        score_matrices = (
            np.eye(6)
            + available_blocks
            @ information_inverse
            @ available_blocks.transpose(0, 2, 1)
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
    selected_matrix = regressor_blocks[selected_indices].reshape(-1, parameter_count)
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


def save_pose_plan(
    output: str | Path,
    *,
    urdf: str | Path | None = None,
    end_link: str = "ee_link",
    base_link: str | None = None,
    count: int = 40,
    candidates: int = 400,
    seed: int = 42,
    characteristic_length_m: float = 1.0,
    start_configuration: np.ndarray | None = None,
    return_to_start: bool = False,
    force: bool = False,
) -> dict:
    """Select calibration poses and save them in joint-space visit order.

    With no explicit start, the first selected pose starts the open route.
    ``return_to_start`` requires an explicit start configuration. Distances
    are Euclidean joint-angle distances in radians, without angle wrapping.
    This produces a visit order, with no collision checks or timed trajectory.
    """
    import hashlib
    import json

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
    lower, upper = np.maximum(limits[:, 0], -np.pi), np.minimum(limits[:, 1], np.pi)
    empty = lower >= upper
    lower[empty], upper[empty] = limits[empty, 0], limits[empty, 1]
    proposals = np.random.default_rng(seed).uniform(
        lower, upper, (candidates, backend.dof)
    )
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
    }
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
    )
    return report
