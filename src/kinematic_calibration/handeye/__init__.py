"""Hand-eye adapters with explicit transform direction and lazy dependencies.

``A_T_B`` maps points in B into A. All translations are metres. The fixed
camera/mounted-board constraint is::

    base_T_ee @ ee_T_board == base_T_camera @ camera_T_board

OpenCV and the HandEyeCalibrationToolkit are optional backends. Importing this
module does not load either dependency or start a simulation engine.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class HandEyeResult:
    base_T_camera: np.ndarray
    ee_T_board: np.ndarray
    position_error_mean_m: float
    position_error_max_m: float
    orientation_error_mean_rad: float
    orientation_error_max_rad: float
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    backend: str
    method: str


def _pose_values(value, name: str, *, batch: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    poses = np.asarray(raw, dtype=np.float64)
    if batch:
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
            raise ValueError(f"{name} must have shape [N, 4, 4] with N >= 3")
    elif poses.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4]")
    values = poses if batch else poses[None]
    rotations = values[:, :3, :3]
    if (
        not np.all(np.isfinite(values))
        or not np.allclose(values[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(
            rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-6
        )
        or not np.allclose(np.linalg.det(rotations), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(f"{name} must contain valid rigid transforms")
    return poses.copy()


def _observations(base_T_ee, camera_T_board):
    robot = _pose_values(base_T_ee, "base_T_ee", batch=True)
    board = _pose_values(camera_T_board, "camera_T_board", batch=True)
    if len(robot) != len(board):
        raise ValueError("base_T_ee and camera_T_board sample counts must match")
    return robot, board


def evaluate_eye_to_hand(
    base_T_ee, camera_T_board, base_T_camera, ee_T_board
) -> dict[str, float | np.ndarray]:
    """Evaluate absolute closure error; this is not an independent truth error."""
    robot, board = _observations(base_T_ee, camera_T_board)
    camera = _pose_values(base_T_camera, "base_T_camera")
    mount = _pose_values(ee_T_board, "ee_T_board")
    deltas = np.linalg.inv(robot @ mount) @ camera @ board
    position_errors = np.linalg.norm(deltas[:, :3, 3], axis=1)
    orientation_errors = Rotation.from_matrix(deltas[:, :3, :3]).magnitude()
    return {
        "position_error_mean_m": float(np.mean(position_errors)),
        "position_error_max_m": float(np.max(position_errors)),
        "orientation_error_mean_rad": float(np.mean(orientation_errors)),
        "orientation_error_max_rad": float(np.max(orientation_errors)),
        "position_errors_m": position_errors,
        "orientation_errors_rad": orientation_errors,
    }


def _check_rotation_excitation(robot):
    relative = robot[0, :3, :3].T @ robot[1:, :3, :3]
    vectors = Rotation.from_matrix(relative).as_rotvec()
    singular = np.linalg.svd(vectors, compute_uv=False)
    if singular[0] < 1e-3 or singular[1] < 1e-3 or singular[1] / singular[0] < 1e-2:
        raise ValueError(
            "hand-eye observations need at least two independent rotation directions"
        )


def solve_toolkit_eye_to_hand(
    base_T_ee, camera_T_board, *, solver: Callable | None = None
) -> HandEyeResult:
    """Fit full board poses with the optional toolkit's public ``Calibrate``.

    The toolkit returns ``Base_T_Camera``. The constant full board mount is
    reconstructed from *all* implied mounts using a chordal SO(3) mean, then
    the physical closure is reevaluated on every input sample. This full-pose
    path requires a board with observable orientation, such as a multi-marker
    CCTag or an asymmetric circle grid. It does not use ``CalibrateAxyb``, whose
    observation model estimates only a target point.
    """
    robot, board = _observations(base_T_ee, camera_T_board)
    _check_rotation_excitation(robot)
    if solver is None:
        try:
            calibration = import_module("handeye_calibration").calibration
        except (ImportError, OSError) as exc:
            raise ImportError(
                "HandEyeCalibrationToolkit native bindings are unavailable. "
                "Initialize third_party/HandEyeCalibrationToolkit and follow "
                "its build/install instructions, or select backend='opencv'."
            ) from exc
        solver = calibration.Calibrate
    report = solver(list(board), list(robot), False)
    if int(report.status) != 0:
        raise RuntimeError(
            f"HandEyeCalibrationToolkit solve failed with status {report.status}"
        )
    # Procedural Calibrate returns pose_calibrate; the class API names it
    # camera_pose. Accept both public report forms for toolkit compatibility.
    pose = getattr(report, "pose_calibrate", None)
    if pose is None:
        pose = getattr(report, "camera_pose", None)
    camera = _pose_values(pose, "toolkit Base_T_Camera")
    implied_mounts = np.linalg.inv(robot) @ camera @ board
    u, _, vt = np.linalg.svd(np.mean(implied_mounts[:, :3, :3], axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    mount = np.eye(4)
    mount[:3, :3] = u @ correction @ vt
    mount[:3, 3] = np.mean(implied_mounts[:, :3, 3], axis=0)
    metrics = evaluate_eye_to_hand(robot, board, camera, mount)
    return HandEyeResult(camera, mount, backend="toolkit", method="axxb", **metrics)


def solve_eye_to_hand(
    base_T_ee, camera_T_board, *, backend: str = "opencv", method: str | None = None
) -> HandEyeResult:
    """Solve a fixed-camera/full-board-pose hand-eye problem.

    ``method`` is ``shah`` or ``li`` for OpenCV and ``axxb`` for the toolkit.
    No backend silently falls back to a different solver.
    """
    if backend == "toolkit":
        if method not in (None, "axxb"):
            raise ValueError("toolkit backend supports only method='axxb'")
        return solve_toolkit_eye_to_hand(base_T_ee, camera_T_board)
    if backend != "opencv":
        raise ValueError(f"unsupported hand-eye backend: {backend}")
    robot, board = _observations(base_T_ee, camera_T_board)
    try:
        opencv = import_module(".opencv", __name__)
    except ImportError as exc:
        raise ImportError(
            "OpenCV backend requires the vision extra: pip install '.[vision]'"
        ) from exc
    result = opencv.solve_opencv_eye_to_hand(robot, board, method=method or "shah")
    metrics = evaluate_eye_to_hand(
        robot, board, result.base_T_camera, result.ee_T_board
    )
    return HandEyeResult(
        result.base_T_camera.copy(),
        result.ee_T_board.copy(),
        backend="opencv",
        method=result.method,
        **metrics,
    )


def camera_observations_to_base_ee(
    camera_T_board, base_T_camera, ee_T_board
) -> np.ndarray:
    """Convert measured board poses into kinematic end-effector observations.

    ``Base_T_EE = Base_T_Camera @ Camera_T_Board @ inverse(EE_T_Board)``.
    The external transforms must be independently calibrated for a meaningful
    evaluation of robot geometry.
    """
    board = _pose_values(camera_T_board, "camera_T_board", batch=True)
    camera = _pose_values(base_T_camera, "base_T_camera")
    mount = _pose_values(ee_T_board, "ee_T_board")
    return camera @ board @ np.linalg.inv(mount)


__all__ = [
    "HandEyeResult",
    "solve_eye_to_hand",
    "solve_toolkit_eye_to_hand",
    "evaluate_eye_to_hand",
    "camera_observations_to_base_ee",
]
