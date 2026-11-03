"""OpenCV robot-world/hand-eye solver ported from auto_calibrate_sim.

This module is loaded only when the OpenCV backend is requested.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from kinematic_calib import io_utils

OPENCV_ROBOT_WORLD_METHODS = {
    "shah": cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "li": cv.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}
MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD = 1e-3
MIN_ROTATION_DIRECTION_RATIO = 1e-2
MIN_ROTATION_SPAN_RAD = 1e-3


def opencv_calibration_implementation_sha256() -> str:
    """Fingerprint the OpenCV solver and artifact-contract implementation."""
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for path in (
        directory / "opencv.py",
        directory.parent / "handeye_cli.py",
        Path(io_utils.__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class OpenCVEyeToHandResult:
    """Eye-to-hand solution using explicit ``A X = Z B`` frame notation."""

    base_T_camera: np.ndarray
    ee_T_board: np.ndarray
    position_error_mean_m: float
    position_error_max_m: float
    orientation_error_mean_rad: float
    orientation_error_max_rad: float
    motion_rotation_rank: int
    max_rotation_from_first_rad: float
    rotation_singular_value_1_rad: float
    rotation_singular_value_2_rad: float
    rotation_direction_ratio: float
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    method: str


@dataclass(frozen=True)
class OpenCVObservationArchive:
    """Validated synchronized observations for offline OpenCV solving."""

    base_T_ee: np.ndarray
    camera_T_board: np.ndarray
    candidate_indices: np.ndarray
    translation_candidate_count: int
    rotation_candidate_count: int


def handeye_observations_sha256(
    base_T_ee: np.ndarray,
    camera_T_board: np.ndarray,
    candidate_indices: np.ndarray,
) -> str:
    """Return a stable content fingerprint independent of NPZ compression."""
    digest = hashlib.sha256()
    digest.update(np.asarray(base_T_ee, dtype="<f8").tobytes())
    digest.update(np.asarray(camera_T_board, dtype="<f8").tobytes())
    digest.update(np.asarray(candidate_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_opencv_observations(path: Path | str) -> OpenCVObservationArchive:
    """Load and validate a schema-tagged eye-to-hand observation NPZ."""
    with io_utils.open_npz(path) as archive:
        required = {
            "schema_version",
            "archive_type",
            "base_T_ee",
            "camera_T_board",
            "candidate_indices",
            "translation_candidate_count",
            "rotation_candidate_count",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"hand-eye archive is missing fields: {sorted(missing)}")
        if (
            archive["schema_version"].shape != ()
            or archive["schema_version"].item() != 1
        ):
            raise ValueError("unsupported hand-eye archive schema version")
        if str(archive["archive_type"].item()) != "eye-to-hand-observations":
            raise ValueError("unsupported hand-eye archive type")
        base_T_ee = _validate_pose_batch("base_T_ee", archive["base_T_ee"])
        camera_T_board = _validate_pose_batch(
            "camera_T_board", archive["camera_T_board"]
        )
        raw_indices = np.asarray(archive["candidate_indices"])
        translation_count = archive["translation_candidate_count"].item()
        rotation_count = archive["rotation_candidate_count"].item()
    if len(base_T_ee) != len(camera_T_board):
        raise ValueError("hand-eye archive pose sample counts do not match")
    if raw_indices.shape != (len(base_T_ee),):
        raise ValueError("candidate_indices must have shape [N]")
    if (
        raw_indices.dtype.kind not in "iuf"
        or not np.all(np.isfinite(raw_indices))
        or np.any(raw_indices < 0)
        or np.any(raw_indices >= np.iinfo(np.int64).max)
    ):
        raise ValueError("candidate_indices must contain finite non-negative integers")
    indices = raw_indices.astype(np.int64)
    if not np.array_equal(raw_indices, indices):
        raise ValueError("candidate_indices must contain integers")
    for name, value in (
        ("translation_candidate_count", translation_count),
        ("rotation_candidate_count", rotation_count),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    candidate_count = int(translation_count + rotation_count)
    if (
        np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices >= candidate_count)
    ):
        raise ValueError(
            "candidate_indices are invalid for the declared candidate counts"
        )
    return OpenCVObservationArchive(
        base_T_ee=base_T_ee,
        camera_T_board=camera_T_board,
        candidate_indices=indices,
        translation_candidate_count=int(translation_count),
        rotation_candidate_count=int(rotation_count),
    )


def _validate_pose_batch(name: str, poses: np.ndarray) -> np.ndarray:
    raw = np.asarray(poses)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) < 3:
        raise ValueError(f"{name} must have shape [N, 4, 4] with N >= 3")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(values[:, 3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-8):
        raise ValueError(f"{name} contains invalid homogeneous transforms")
    rotations = values[:, :3, :3]
    if not np.allclose(
        rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-6
    ) or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"{name} contains invalid rotations")
    return values


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64)
    value[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return value


def evaluate_eye_to_hand_solution(
    base_T_ee: np.ndarray,
    camera_T_board: np.ndarray,
    base_T_camera: np.ndarray,
    ee_T_board: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Evaluate the absolute ``A X = Z B`` closure residuals."""
    ee_poses = _validate_pose_batch("base_T_ee", base_T_ee)
    board_poses = _validate_pose_batch("camera_T_board", camera_T_board)
    if len(ee_poses) != len(board_poses):
        raise ValueError("base_T_ee and camera_T_board sample counts must match")
    solutions = []
    for name, value in (
        ("Base_T_Camera", base_T_camera),
        ("EE_T_Board", ee_T_board),
    ):
        transform = np.asarray(value, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must be a 4x4 transform")
        rotation = transform[:3, :3]
        if (
            not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-6)
        ):
            raise ValueError(f"{name} is not a valid rigid transform")
        solutions.append(transform)
    base_T_camera_value, ee_T_board_value = solutions
    # The robot and camera sides both express Base_T_Board.
    lhs = ee_poses @ ee_T_board_value
    rhs = base_T_camera_value @ board_poses
    delta = np.linalg.inv(lhs) @ rhs
    # Preserve the per-vector norm reduction order and worst-sample tie behavior.
    position_errors_array = np.asarray(
        [float(np.linalg.norm(item[:3, 3])) for item in delta]
    )
    cosine = np.clip(
        (np.trace(delta[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    orientation_errors_array = np.arccos(cosine)
    return {
        "position_error_mean_m": float(np.mean(position_errors_array)),
        "position_error_max_m": float(np.max(position_errors_array)),
        "orientation_error_mean_rad": float(np.mean(orientation_errors_array)),
        "orientation_error_max_rad": float(np.max(orientation_errors_array)),
        "position_errors_m": position_errors_array,
        "orientation_errors_rad": orientation_errors_array,
    }


def measure_handeye_motion(base_T_ee: np.ndarray) -> dict[str, float | int]:
    """Measure translational and rotational excitation of EE observations."""
    poses = _validate_pose_batch("base_T_ee", base_T_ee)
    relative_rotation_vectors = []
    reference_rotation = poses[0, :3, :3]
    for pose in poses[1:]:
        relative_rotation = reference_rotation.T @ pose[:3, :3]
        rotation_vector, _ = cv.Rodrigues(relative_rotation)
        relative_rotation_vectors.append(rotation_vector.reshape(3))
    relative_rotation_vectors = np.asarray(relative_rotation_vectors)
    singular_values = np.linalg.svd(relative_rotation_vectors, compute_uv=False)
    largest = float(singular_values[0])
    second = float(singular_values[1])
    translations = poses[:, :3, 3]
    translation_span = translations.max(axis=0) - translations.min(axis=0)
    return {
        "motion_rotation_rank": int(np.linalg.matrix_rank(relative_rotation_vectors)),
        "rotation_singular_value_1_rad": largest,
        "rotation_singular_value_2_rad": second,
        "rotation_direction_ratio": second / largest if largest > 0.0 else 0.0,
        "max_rotation_from_first_rad": float(
            np.max(np.linalg.norm(relative_rotation_vectors, axis=1))
        ),
        "translation_span_x_m": float(translation_span[0]),
        "translation_span_y_m": float(translation_span[1]),
        "translation_span_z_m": float(translation_span[2]),
        "translation_span_norm_m": float(np.linalg.norm(translation_span)),
    }


def solve_opencv_eye_to_hand(
    base_T_ee: np.ndarray,
    camera_T_board: np.ndarray,
    *,
    method: str = "shah",
) -> OpenCVEyeToHandResult:
    """Solve ``Base_T_EE * EE_T_Board = Base_T_Camera * Camera_T_Board``."""
    ee_poses = _validate_pose_batch("base_T_ee", base_T_ee)
    board_poses = _validate_pose_batch("camera_T_board", camera_T_board)
    if len(ee_poses) != len(board_poses):
        raise ValueError("base_T_ee and camera_T_board sample counts must match")
    try:
        method_code = OPENCV_ROBOT_WORLD_METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unsupported OpenCV robot-world method: {method}") from exc
    excitation = measure_handeye_motion(ee_poses)
    motion_rotation_rank = int(excitation["motion_rotation_rank"])
    max_rotation_from_first_rad = float(excitation["max_rotation_from_first_rad"])
    if motion_rotation_rank < 2 or max_rotation_from_first_rad < MIN_ROTATION_SPAN_RAD:
        raise ValueError(
            "OpenCV hand-eye observations need at least two independent rotation "
            "directions"
        )
    if (
        excitation["rotation_singular_value_2_rad"]
        < MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD
        or excitation["rotation_direction_ratio"] < MIN_ROTATION_DIRECTION_RATIO
    ):
        raise ValueError(
            "OpenCV hand-eye observations are nearly single-axis; increase "
            "the second independent rotation direction"
        )
    try:
        r_ee_T_board, t_ee_T_board, r_base_T_camera, t_base_T_camera = (
            cv.calibrateRobotWorldHandEye(
                [pose[:3, :3] for pose in ee_poses],
                [pose[:3, 3] for pose in ee_poses],
                [pose[:3, :3] for pose in board_poses],
                [pose[:3, 3] for pose in board_poses],
                method=method_code,
            )
        )
    except cv.error as exc:
        raise ValueError(f"OpenCV robot-world/hand-eye solve failed: {exc}") from exc

    ee_T_board = _transform(r_ee_T_board, t_ee_T_board)
    base_T_camera = _transform(r_base_T_camera, t_base_T_camera)
    metrics = evaluate_eye_to_hand_solution(
        ee_poses, board_poses, base_T_camera, ee_T_board
    )
    return OpenCVEyeToHandResult(
        base_T_camera=base_T_camera,
        ee_T_board=ee_T_board,
        position_error_mean_m=metrics["position_error_mean_m"],
        position_error_max_m=metrics["position_error_max_m"],
        orientation_error_mean_rad=metrics["orientation_error_mean_rad"],
        orientation_error_max_rad=metrics["orientation_error_max_rad"],
        motion_rotation_rank=motion_rotation_rank,
        max_rotation_from_first_rad=max_rotation_from_first_rad,
        rotation_singular_value_1_rad=float(
            excitation["rotation_singular_value_1_rad"]
        ),
        rotation_singular_value_2_rad=float(
            excitation["rotation_singular_value_2_rad"]
        ),
        rotation_direction_ratio=float(excitation["rotation_direction_ratio"]),
        position_errors_m=np.asarray(metrics["position_errors_m"]),
        orientation_errors_rad=np.asarray(metrics["orientation_errors_rad"]),
        method=method,
    )
