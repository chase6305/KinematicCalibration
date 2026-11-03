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
    condition use the *unregularized* stacked selected regressor.
    """
    from robotkinecal import SerialRobotKineCal

    from .experiment import predict_poe

    raw = np.asarray(candidates)
    if (
        raw.dtype.kind not in "iuf"
        or raw.ndim != 2
        or raw.shape[1] != backend.dof
        or not len(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("candidates must be finite real [N,DOF] joint configurations")
    q = raw.astype(float, copy=False)
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or not 1 <= count <= len(q)
    ):
        raise ValueError("count must be an integer between 1 and the candidate count")
    if (
        not np.isfinite([characteristic_length_m, regularization]).all()
        or min(characteristic_length_m, regularization) <= 0
    ):
        raise ValueError(
            "characteristic length and regularization must be finite and positive"
        )
    limits = np.asarray(backend.joint_limits)
    if (
        limits.shape != (backend.dof, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("backend must provide ordered finite [DOF,2] joint limits")
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("candidate configurations must respect backend joint limits")
    if not hasattr(backend, "poe_parameters"):
        raise ValueError("pose selection requires a backend providing poe_parameters()")
    axes, home = backend.poe_parameters()
    calibrator = SerialRobotKineCal.from_screw_axes(axes, home)
    # Predicted poses here initialize the analytical model's regressor dimensions.
    # These are planning quantities and are never exported as observed data.
    from spatialmath import SE3

    calibrator.set_observations(
        q, [SE3(p, check=False) for p in predict_poe(axes, home, q)]
    )
    blocks = calibrator.regressor_blocks(q)
    blocks[:, :3] /= characteristic_length_m
    parameters = blocks.shape[2]
    information_inverse = np.eye(parameters) / regularization
    remaining = np.ones(len(q), dtype=bool)
    selected = []
    for _ in range(count):
        indices = np.flatnonzero(remaining)
        available = blocks[indices]
        inner = np.eye(6) + available @ information_inverse @ available.transpose(
            0, 2, 1
        )
        signs, scores = np.linalg.slogdet(inner)
        if np.any(signs <= 0) or not np.isfinite(scores).all():
            raise ValueError("pose information score became numerically singular")
        index = int(indices[np.argmax(scores)])
        selected.append(index)
        remaining[index] = False
        # Re-factor the small information matrix rather than accumulating inverse
        # update drift when candidate rows have very different magnitudes.
        stacked = blocks[selected].reshape(-1, parameters)
        information_inverse = np.linalg.solve(
            stacked.T @ stacked + regularization * np.eye(parameters),
            np.eye(parameters),
        )
        information_inverse = (information_inverse + information_inverse.T) / 2
    selected = np.asarray(selected, dtype=np.int64)
    matrix = blocks[selected].reshape(-1, parameters)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(singular[0] / singular[-1]) if rank == parameters else None
    return PoseSelection(
        q[selected].copy(),
        selected,
        singular,
        rank,
        parameters,
        condition,
        characteristic_length_m,
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
    force: bool = False,
) -> dict:
    """Save proposed configurations, explicitly distinct from measured datasets."""
    import hashlib
    import json

    from kinematic_calib.io_utils import atomic_savez_compressed

    from .backends.numpy import NumpyBackend
    from .experiment import example_urdf

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
    report = {
        "schema_version": 1,
        "artifact_type": "planned_joint_configurations",
        "measurements_acquired": False,
        "collision_checked": False,
        "visibility_checked": False,
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "joint_names": list(backend.joint_names),
        "base_link": backend.base_link,
        "end_link": end_link,
        "seed": seed,
        "candidate_count": candidates,
        "selected_count": count,
        "regressor_rank": selected.rank,
        "parameter_count": selected.parameter_count,
        "full_rank": selected.rank == selected.parameter_count,
        "condition_number": selected.condition_number,
        "characteristic_length_m": characteristic_length_m,
    }
    atomic_savez_compressed(
        output,
        planned_joint_positions=selected.joint_positions,
        candidate_indices=selected.candidate_indices,
        singular_values=selected.singular_values,
        metadata_json=np.asarray(json.dumps(report, allow_nan=False)),
    )
    return report
