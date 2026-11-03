from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calibration.targets import detection, get_board_config

cv2 = pytest.importorskip("cv2")


def test_pnp_recovers_board_pose_from_projected_cctag_centers():
    board = get_board_config()
    k = np.array([[800, 0, 320], [0, 820, 240], [0, 0, 1]], float)
    vector = np.array([0.2, -0.1, 0.05])
    translation = np.array([0.02, -0.03, 0.9])
    pixels, _ = cv2.projectPoints(board.points, vector, translation, k, np.zeros(5))
    result = detection.pose_from_correspondences(board, pixels[:, 0], k)
    np.testing.assert_allclose(
        result.camera_T_board[:3, :3],
        Rotation.from_rotvec(vector).as_matrix(),
        atol=1e-7,
    )
    np.testing.assert_allclose(result.camera_T_board[:3, 3], translation, atol=1e-8)
    assert result.reprojection_rms_px < 1e-6
    pixels[1, 0] += [15, -10]
    with pytest.raises(ValueError, match="reprojection RMS"):
        detection.pose_from_correspondences(board, pixels[:, 0], k)


def test_opencv_detects_white_circle_grid_with_rgb_and_inversion():
    board = get_board_config("opencv-acircle")
    # Exact front-facing metric projection, with board margins retained.
    image = np.zeros((720, 900, 3), np.uint8)
    for x, y, _ in board.points:
        center = tuple(
            np.rint(([x, y] + np.array(board.origin_from_top_left_m)) * 3000).astype(
                int
            )
        )
        cv2.circle(image, center, 24, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    k = np.array([[1500, 0, 450], [0, 1500, 360], [0, 0, 1]], float)
    result = detection.detect_opencv_circles(image, k)
    assert result is not None
    assert result.reprojection_rms_px < 0.1
    assert result.camera_T_board[2, 3] == pytest.approx(0.5, abs=1e-4)
    assert detection.detect_opencv_circles(np.zeros_like(image), k) is None


def test_legacy_profile_and_single_orientation_rejected_before_native_import():
    with pytest.raises(ValueError, match="legacy CCTag"):
        detection.detect_toolkit_board(np.zeros((10, 10), np.uint8), np.eye(3))
    with pytest.raises(ValueError, match="single CCTag"):
        detection.detect_toolkit_board(
            np.zeros((10, 10), np.uint8),
            np.eye(3),
            board=get_board_config("cctag-single"),
        )


def test_native_detection_passes_twice_physical_row_pitch(monkeypatch):
    calls = []
    pose = np.eye(4)
    pose[2, 3] = 1.0

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(status=0, board_pose=pose)

    calibration = SimpleNamespace(EllipseDetectBoardPose=native)
    monkeypatch.setattr(
        detection,
        "import_module",
        lambda name: cv2 if name == "cv2" else SimpleNamespace(calibration=calibration),
    )
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
    result = detection.detect_toolkit_board(
        np.zeros((480, 640, 3), np.uint8), k, allow_legacy_profile=True
    )
    assert result.backend == "toolkit"
    assert calls[0][1]["board_space_y"] == 0.13
    np.testing.assert_allclose(result.camera_T_board, pose)


def test_toolkit_reprojection_uses_correspondences_and_reorders_by_geometry(
    monkeypatch,
):
    board = get_board_config()
    pose = np.eye(4)
    pose[:3, 3] = [0.02, -0.03, 1.0]
    k = np.array([[800, 0, 320], [0, 820, 240], [0, 0, 1]], float)
    distortion = np.array([0.05, -0.02, 0.001, -0.003, 0.01])
    projected, _ = cv2.projectPoints(
        board.points, np.zeros(3), pose[:3, 3], k, distortion
    )
    pixels = projected[:, 0]
    shuffled = np.array([2, 0, 5, 4, 1, 3])

    def native(*args, **kwargs):
        assert args[4] == "cctag"
        return SimpleNamespace(
            status=0,
            board_pose=pose,
            object_points=board.points[shuffled],
            grid_list=pixels[shuffled],
        )

    calibration = SimpleNamespace(
        DetectGridBoardPose=native, BoardType=SimpleNamespace(CCTAG="cctag")
    )
    monkeypatch.setattr(
        detection,
        "import_module",
        lambda name: cv2 if name == "cv2" else SimpleNamespace(calibration=calibration),
    )
    result = detection.detect_toolkit_board(
        np.zeros((480, 640, 3), np.uint8), k, distortion, allow_legacy_profile=True
    )
    np.testing.assert_allclose(result.image_points, pixels)
    assert result.reprojection_rms_px < 1e-10


def test_optional_native_toolkit_circle_detection_exposes_real_pixel_rms():
    pytest.importorskip("handeye_calibration")
    board = get_board_config("opencv-acircle")
    image = np.zeros((720, 900, 3), np.uint8)
    for x, y, _ in board.points:
        center = tuple(
            np.rint(([x, y] + np.array(board.origin_from_top_left_m)) * 3000).astype(
                int
            )
        )
        cv2.circle(image, center, 24, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    k = np.array([[1500, 0, 450], [0, 1500, 360], [0, 0, 1]], float)
    result = detection.detect_toolkit_board(image, k, board=board, invert_gray=True)
    assert result is not None
    assert result.image_points.shape == (77, 2)
    assert result.reprojection_rms_px < 0.2
    assert result.camera_T_board[2, 3] == pytest.approx(0.5, abs=1e-3)


def test_collinear_board_points_rejected_instead_of_accepting_arbitrary_orientation():
    from dataclasses import replace

    board = replace(get_board_config(), size_x=6, size_y=1, pitch_x_m=0.03)
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
    pixels, _ = cv2.projectPoints(
        board.points,
        np.array([0.3, 0.2, 0.1]),
        np.array([0.02, -0.03, 1.0]),
        k,
        np.zeros(5),
    )
    with pytest.raises(ValueError, match="noncollinear metric"):
        detection.pose_from_correspondences(board, pixels[:, 0], k)


def test_nearly_frontal_far_board_reports_pose_ambiguity_despite_low_rms():
    board = get_board_config()
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
    vector, translation = np.array([0.01, 0.01, 0.0]), np.array([0.0, 0.0, 3.0])
    pixels, _ = cv2.projectPoints(board.points, vector, translation, k, np.zeros(5))
    # A concrete pixel-noise realization passes a pixel-only gate despite
    # two substantially different plausible plane orientations.
    noisy = pixels[:, 0] + np.array(
        [
            [-0.336573954323161, -0.066977005997154],
            [0.032550613021001, 0.117244466271856],
            [0.142245315958571, 0.158669447039985],
            [-0.069745014449688, -0.092470358532913],
            [0.171595176251431, -0.038260864977291],
            [-0.255137264667584, -0.226657442800697],
        ]
    )
    result = detection.pose_from_correspondences(board, noisy, k)
    diagnostics = result.pose_diagnostics
    assert result.reprojection_rms_px < 0.5
    assert diagnostics["evaluated"]
    assert diagnostics["candidate_rotation_separation_rad"] > 0.05
    assert diagnostics["ambiguous"]
