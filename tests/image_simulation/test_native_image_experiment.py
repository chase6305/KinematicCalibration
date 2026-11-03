"""Optional pinned-toolkit checks using the actual version-3 CCTag artwork."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.acquisition import import_captures
from kinematic_calibration.cli import main
from kinematic_calibration.image_simulation import (
    ImageNoise,
    PlanarBoardRenderer,
    load_image_experiment,
)
from kinematic_calibration.targets import get_board_config
from kinematic_calibration.targets.detection import detect_toolkit_board

cv2 = pytest.importorskip("cv2")
pytest.importorskip("handeye_calibration")


def test_optional_native_cctag_v3_recovers_pose_from_rendered_png(tmp_path):
    board = get_board_config("cctag-multi-v3")
    renderer = PlanarBoardRenderer(
        board,
        noise=ImageNoise(0, 0, 2),
        registration_backend="toolkit",
    )
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_rotvec([0.3, -0.25, 0.1]).as_matrix()
    expected[:3, 3] = [-0.0975, -0.065, 0.9]
    image_path = tmp_path / "cctag-v3.png"
    image = renderer.render(expected)
    assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    decoded = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    detection = detect_toolkit_board(
        decoded, renderer.camera.intrinsic, board=board, max_board_tilt_rad=1.2
    )
    assert detection is not None
    assert detection.image_points.shape == (6, 2)
    np.testing.assert_array_equal(detection.native_report.marker_ids, np.arange(6))
    assert detection.reprojection_rms_px < 0.25
    assert not detection.pose_diagnostics["ambiguous"]
    assert np.linalg.norm(detection.camera_T_board[:3, 3] - expected[:3, 3]) < 0.003
    assert (
        Rotation.from_matrix(
            detection.camera_T_board[:3, :3].T @ expected[:3, :3]
        ).magnitude()
        < 0.01
    )


def test_optional_native_cctag_v3_cli_fits_and_reloads_independent_images(tmp_path):
    output = tmp_path / "v3-experiment"
    assert (
        main(
            [
                "image-demo",
                "--output",
                str(output),
                "--board",
                "cctag-multi-v3",
                "--detector-backend",
                "toolkit",
                "--samples",
                "12",
                "--validation-samples",
                "6",
                "--max-candidates",
                "2000",
            ]
        )
        == 0
    )
    report, validation, _ = load_image_experiment(output)
    assert report["accepted"]
    assert report["measurement_model"] == "rendered-image-detection"
    assert report["detector_backend"] == "toolkit"
    assert report["board_type"] == "cctag-multi-v3"
    assert report["regressor_rank"] == report["parameter_count"] == 30
    assert report["texture_registration"]["method"] == (
        "static-native-cctag-control-point-registration"
    )
    assert (
        report["calibrated_vs_truth"]["position_rmse_m"]
        < report["nominal_vs_truth"]["position_rmse_m"] / 2
    )
    assert len(validation.joint_positions) == 6
    with np.load(output / "train.npz") as archive:
        training_ids = archive["sample_indices"]
    assert not np.intersect1d(training_ids, validation.sample_indices).size
    manifest = json.loads((output / "captures/train/manifest.json").read_text())
    assert all(
        "image" in row and "image_points" not in row for row in manifest["samples"]
    )
    imported, import_report = import_captures(
        output / "captures/train/manifest.json", tmp_path / "reimported.npz"
    )
    original = KinematicDataset.load(output / "train-camera.npz")
    assert import_report["all_samples_accepted"]
    np.testing.assert_allclose(
        imported.observed_poses, original.observed_poses, atol=1e-10
    )
    np.testing.assert_array_equal(imported.sample_indices, original.sample_indices)
