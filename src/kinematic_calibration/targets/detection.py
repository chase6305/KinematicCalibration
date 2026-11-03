"""Image-to-board observations with optional OpenCV and toolkit detectors."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from . import BoardConfig, BoardType, get_board_config


@dataclass(frozen=True)
class BoardDetection:
    """A board observation in the optical camera frame, with metric translation."""

    camera_T_board: np.ndarray
    image_points: np.ndarray | None
    reprojection_rms_px: float | None
    backend: str
    native_report: object | None = None
    pose_diagnostics: dict | None = None


def _camera_parameters(intrinsic, distortion):
    intrinsic = np.asarray(intrinsic)
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion)
    if intrinsic.dtype.kind not in "iuf" or distortion.dtype.kind not in "iuf":
        raise ValueError("camera parameters must contain real numeric values")
    intrinsic = intrinsic.astype(np.float64)
    distortion = distortion.astype(np.float64).reshape(-1)
    if (
        intrinsic.shape != (3, 3)
        or not np.all(np.isfinite(intrinsic))
        or not np.all(np.isfinite(distortion))
        or distortion.size not in (0, 4, 5, 8, 12, 14)
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-12)
        or abs(intrinsic[0, 1]) > 1e-12
        or abs(intrinsic[1, 0]) > 1e-12
    ):
        raise ValueError("invalid pinhole intrinsic matrix or distortion coefficients")
    return np.ascontiguousarray(intrinsic), np.ascontiguousarray(distortion)


def _image(image):
    image = np.asarray(image)
    if (
        image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or not image.size
        or (image.ndim == 3 and image.shape[2] not in (3, 4))
    ):
        raise ValueError("image must be nonempty uint8 grayscale, RGB or RGBA")
    return np.ascontiguousarray(image)


def pose_from_correspondences(
    board: BoardConfig,
    image_points,
    intrinsic,
    distortion=None,
    *,
    max_reprojection_rms_px: float = 2.0,
) -> BoardDetection:
    """Fit ``Camera_T_Board`` from ordered board/pixel correspondences.

    This path can consume externally detected CCTag centers from an existing
    detector. Ordering must match ``board.points``; point identity is supplied
    by the caller. A single circular marker cannot produce a full board pose.
    """
    if board.size_x * board.size_y < 4:
        raise ValueError("full board pose requires at least four control points")
    if not np.isfinite(max_reprojection_rms_px) or max_reprojection_rms_px <= 0:
        raise ValueError("max_reprojection_rms_px must be positive and finite")
    if np.linalg.matrix_rank(board.points - board.points.mean(axis=0)) < 2:
        raise ValueError("full board pose requires noncollinear metric control points")
    pixels = np.asarray(image_points)
    if (
        pixels.shape != (len(board.points), 2)
        or pixels.dtype.kind not in "iuf"
        or not np.all(np.isfinite(pixels))
    ):
        raise ValueError(
            "image_points must be finite real [board point count, 2] pixels"
        )
    if np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2:
        raise ValueError("full board pose requires noncollinear image control points")
    intrinsic, distortion = _camera_parameters(intrinsic, distortion)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    cv = import_module("cv2")
    try:
        ok, rotvec, translation = cv.solvePnP(
            board.points, pixels, intrinsic, distortion, flags=cv.SOLVEPNP_ITERATIVE
        )
    except cv.error as exc:
        raise ValueError(f"board pose estimation failed: {exc}") from exc
    if not ok:
        raise ValueError("board pose estimation did not converge")
    rotation, _ = cv.Rodrigues(rotvec)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation[:, 0]
    camera_points = board.points @ rotation.T + translation[:, 0]
    if not np.all(np.isfinite(pose)) or np.any(camera_points[:, 2] <= 0):
        raise ValueError("board pose has invalid or nonpositive optical depth")
    projected, _ = cv.projectPoints(
        board.points, rotvec, translation, intrinsic, distortion
    )
    rms = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)))
    )
    if rms > max_reprojection_rms_px:
        raise ValueError(
            f"board reprojection RMS {rms:.4g} px exceeds {max_reprojection_rms_px:.4g} px"
        )
    diagnostics = _planar_pose_diagnostics(board, pixels, intrinsic, distortion)
    return BoardDetection(
        pose, pixels.copy(), rms, "opencv-pnp", pose_diagnostics=diagnostics
    )


def detect_opencv_circles(
    image,
    intrinsic,
    distortion=None,
    *,
    board: BoardConfig | None = None,
    invert_gray: bool = True,
    max_reprojection_rms_px: float = 2.0,
) -> BoardDetection | None:
    """Detect the asymmetric-circle board from RGB/grayscale, then run PnP.

    Returns ``None`` when no complete grid is detected. The supplied legacy
    artwork has white circles on black, so grayscale inversion defaults to
    true. Detection alone does not establish visibility or collision freedom.
    """
    board = board or get_board_config(BoardType.OPENCV_ACIRCLE)
    if board.board_type != BoardType.OPENCV_ACIRCLE:
        raise ValueError("OpenCV circles detector requires an asymmetric-circle board")
    image = _image(image)
    cv = import_module("cv2")
    gray = (
        image
        if image.ndim == 2
        else cv.cvtColor(
            image, cv.COLOR_RGB2GRAY if image.shape[2] == 3 else cv.COLOR_RGBA2GRAY
        )
    )
    if invert_gray:
        gray = cv.bitwise_not(gray)
    found, centers = cv.findCirclesGrid(
        gray,
        board.toolkit_size,
        flags=cv.CALIB_CB_ASYMMETRIC_GRID | cv.CALIB_CB_CLUSTERING,
    )
    if not found:
        return None
    return pose_from_correspondences(
        board,
        centers.reshape(-1, 2),
        intrinsic,
        distortion,
        max_reprojection_rms_px=max_reprojection_rms_px,
    )


def detect_toolkit_board(
    image,
    intrinsic,
    distortion=None,
    *,
    board: BoardConfig | None = None,
    depth_m=None,
    allow_legacy_profile: bool = False,
    invert_gray: bool = False,
    max_board_tilt_rad: float = np.pi / 3,
) -> BoardDetection | None:
    """Use the optional native toolkit to detect a full grid board pose.

    Legacy CCTag profiles are rejected by default. Set ``allow_legacy_profile``
    only when using a detector build verified against the retained physical
    board. The current toolkit includes its own versioned printable templates;
    a matching modern board configuration should set ``legacy_artwork=False``.
    """
    board = board or get_board_config()
    if board.board_type == BoardType.CCTAG_SINGLE:
        raise ValueError(
            "a single CCTag does not observe full orientation; use the toolkit point workflow with depth"
        )
    if (
        board.board_type == BoardType.CCTAG_MULTI
        and board.legacy_artwork
        and not allow_legacy_profile
    ):
        raise ValueError(
            "legacy CCTag ring profile requires an explicitly verified detector; pass allow_legacy_profile=True only for a compatible detector build"
        )
    if not np.isfinite(max_board_tilt_rad) or not 0 < max_board_tilt_rad <= np.pi:
        raise ValueError("max_board_tilt_rad must be in (0, pi]")
    image = _image(image)
    intrinsic, distortion = _camera_parameters(intrinsic, distortion)
    if distortion.size != 5:
        raise ValueError(
            "the toolkit detector requires exactly five distortion coefficients"
        )
    if depth_m is not None:
        depth = np.asarray(depth_m)
        if (
            depth.shape != image.shape[:2]
            or depth.dtype.kind not in "fiu"
            or not np.all(np.isfinite(depth))
            or np.any(depth < 0)
        ):
            raise ValueError(
                "depth_m must be finite nonnegative metres with the image shape"
            )
        with np.errstate(over="ignore"):
            depth_m = np.ascontiguousarray(depth, dtype=np.float32)
        if not np.all(np.isfinite(depth_m)):
            raise ValueError("depth_m is outside the float32 range")
    try:
        calibration = import_module("handeye_calibration").calibration
    except (ImportError, OSError) as exc:
        raise ImportError(
            "build/install third_party/HandEyeCalibrationToolkit native bindings before selecting its detector"
        ) from exc
    sx, sy = board.toolkit_spacing
    options = dict(
        board_size_x=board.size_x,
        board_size_y=board.size_y,
        board_space_x=sx,
        board_space_y=sy,
        angle_threshold=float(max_board_tilt_rad),
        inv_gray=bool(invert_gray),
    )
    if hasattr(calibration, "DetectGridBoardPose"):
        kind = (
            calibration.BoardType.CCTAG
            if board.board_type == BoardType.CCTAG_MULTI
            else calibration.BoardType.OPENCV_ACIRCLE
        )
        report = calibration.DetectGridBoardPose(
            image,
            depth_m,
            intrinsic,
            distortion,
            kind,
            **options,
        )
    else:
        detector = (
            calibration.EllipseDetectBoardPose
            if board.board_type == BoardType.CCTAG_MULTI
            else calibration.DetectBoardPose
        )
        report = detector(image, depth_m, intrinsic, distortion, **options)
    if not bool(getattr(report, "success", report.status == 0)):
        return None
    # Reuse the strict adapter validation without importing OpenCV.
    from ..handeye import _pose_values

    pose = _pose_values(report.board_pose, "Camera_T_Board")
    if np.any((board.points @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        raise ValueError("toolkit detected board with nonpositive optical depth")
    pixels, rms = _toolkit_reprojection(board, report, pose, intrinsic, distortion)
    diagnostics = (
        _planar_pose_diagnostics(board, pixels, intrinsic, distortion)
        if pixels is not None
        else None
    )
    return BoardDetection(pose, pixels, rms, "toolkit", report, diagnostics)


__all__ = [
    "BoardDetection",
    "pose_from_correspondences",
    "detect_opencv_circles",
    "detect_toolkit_board",
]


def _toolkit_reprojection(board, report, pose, intrinsic, distortion):
    """Recompute pixel RMS from native correspondences without importing cv2.

    Old builds may expose only a pose. Return unavailable diagnostics in that
    case; callers with a strict pixel gate must reject that sample.
    """
    if not hasattr(report, "object_points") or not hasattr(report, "grid_list"):
        return None, None
    objects = np.asarray(report.object_points, dtype=np.float64)
    pixels = np.asarray(report.grid_list, dtype=np.float64)
    if objects.shape != board.points.shape or pixels.shape != (len(objects), 2):
        raise ValueError(
            "toolkit correspondences do not match the complete configured board"
        )
    if not np.all(np.isfinite(objects)) or not np.all(np.isfinite(pixels)):
        raise ValueError("toolkit correspondences contain nonfinite values")
    # CCTag marker IDs need not be consecutive or sorted. Match the metric
    # coordinates, which are explicitly paired with image pixels by the API.
    distances = np.linalg.norm(board.points[:, None] - objects[None], axis=2)
    order = np.argmin(distances, axis=1)
    if len(np.unique(order)) != len(order) or np.any(
        distances[np.arange(len(order)), order] > 1e-6
    ):
        raise ValueError(
            "toolkit object points disagree with configured board geometry"
        )
    pixels = pixels[order]
    camera_points = board.points @ pose[:3, :3].T + pose[:3, 3]
    normalized = camera_points[:, :2] / camera_points[:, 2:]
    x, y = normalized.T
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = distortion
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    distorted = np.column_stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )
    )
    projected = distorted * [intrinsic[0, 0], intrinsic[1, 1]] + intrinsic[:2, 2]
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    if not np.isfinite(rms):
        raise ValueError("toolkit projection produced a nonfinite pixel RMS")
    return pixels.copy(), rms


def _planar_pose_diagnostics(board, pixels, intrinsic, distortion):
    """Expose the two-solution planar PnP ambiguity as a diagnostic.

    A low reprojection RMS alone cannot certify a plane's orientation. IPPE
    supplies alternate physically valid poses. Similar reprojection errors
    with separated rotations indicate weak orientation evidence. The ratio
    and angular cutoffs below are declared heuristics, not confidence bounds.
    """
    thresholds = {
        "max_alternative_to_best_rms_ratio": 1.5,
        "min_candidate_rotation_separation_rad": 0.05,
    }
    try:
        cv = import_module("cv2")
    except ImportError:
        return {"evaluated": False, "reason": "OpenCV IPPE is unavailable"}
    try:
        report = cv.solvePnPGeneric(
            board.points,
            np.ascontiguousarray(pixels, dtype=np.float64),
            intrinsic,
            distortion,
            flags=cv.SOLVEPNP_IPPE,
        )
    except cv.error as exc:
        return {"evaluated": False, "reason": f"OpenCV IPPE failed: {exc}"}
    candidates = []
    for rotation_vector, translation in zip(report[1], report[2], strict=True):
        rotation, _ = cv.Rodrigues(rotation_vector)
        translation = translation.reshape(3)
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            continue
        camera_points = board.points @ rotation.T + translation
        if np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv.projectPoints(
            board.points,
            rotation_vector,
            translation,
            intrinsic,
            distortion,
        )
        rms = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)))
        )
        if np.isfinite(rms):
            candidates.append((rms, rotation, translation))
    candidates.sort(key=lambda candidate: candidate[0])
    if len(candidates) < 2:
        return {
            "evaluated": False,
            "reason": "IPPE did not return two finite positive-depth poses",
            "candidate_count": len(candidates),
        }
    best, alternative = candidates[:2]
    relative = best[1].T @ alternative[1]
    angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    shift = float(np.linalg.norm(best[2] - alternative[2]))
    ratio = (
        alternative[0] / best[0]
        if best[0] > 1e-12
        else (1.0 if alternative[0] <= 1e-12 else None)
    )
    ambiguous = (
        ratio is not None
        and ratio <= thresholds["max_alternative_to_best_rms_ratio"]
        and angle >= thresholds["min_candidate_rotation_separation_rad"]
    )
    return {
        "evaluated": True,
        "method": "planar-ippe-candidates",
        "candidate_count": len(candidates),
        "ambiguous": bool(ambiguous),
        "best_candidate_rms_px": best[0],
        "alternative_candidate_rms_px": alternative[0],
        "alternative_to_best_rms_ratio": ratio,
        "candidate_rotation_separation_rad": angle,
        "candidate_translation_separation_m": shift,
        "heuristic_thresholds": thresholds,
        "interpretation": "candidate ambiguity diagnostic, not a calibrated confidence interval",
    }
