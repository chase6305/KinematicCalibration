import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration import image_simulation
from kinematic_calibration.acquisition import import_captures
from kinematic_calibration.image_simulation import (
    ImageNoise,
    PlanarBoardRenderer,
    run_image_experiment,
)
from kinematic_calibration.targets import get_board_config
from kinematic_calibration.targets.detection import detect_opencv_circles


def test_actual_warped_png_is_detected_with_metric_pose():
    renderer = PlanarBoardRenderer(
        get_board_config("opencv-acircle"), noise=ImageNoise(0, 0, 1)
    )
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    pose[:3, 3] = [-0.12, -0.1, 0.9]
    image = renderer.render(pose)
    detection = detect_opencv_circles(image, renderer.camera.intrinsic)
    assert detection is not None
    assert image.dtype == np.uint8
    assert image.shape == (960, 1280, 3)
    np.testing.assert_allclose(detection.camera_T_board[:3, 3], pose[:3, 3], atol=0.001)
    assert (
        Rotation.from_matrix(
            detection.camera_T_board[:3, :3].T @ pose[:3, :3]
        ).magnitude()
        < 0.005
    )
    assert detection.reprojection_rms_px < 0.2
    assert renderer.registration["rms_source_px"] < 0.1
    pose[2, 3] = -0.9
    with pytest.raises(ValueError, match="not renderable"):
        renderer.render(pose)


@pytest.fixture(scope="module")
def completed_experiment(tmp_path_factory):
    output = tmp_path_factory.mktemp("image-trial") / "experiment"
    report = run_image_experiment(
        output,
        samples=24,
        validation_samples=10,
        noise=ImageNoise(0.5, 0.3, 1),
        seed=42,
    )
    return output, report


def test_image_measurements_improve_independent_truth_and_have_full_rank(
    completed_experiment,
):
    output, report = completed_experiment
    assert report["accepted"]
    assert report["regressor_rank"] == report["parameter_count"] == 30
    assert report["measurement_model"] == "rendered-image-detection"
    assert report["measurement_vs_truth"]["position_rmse_m"] > 1e-6
    assert (
        report["calibrated_vs_truth"]["position_rmse_m"]
        < report["nominal_vs_truth"]["position_rmse_m"] / 10
    )
    train = KinematicDataset.load(output / "train.npz")
    validation = KinematicDataset.load(output / "validation.npz")
    assert not np.intersect1d(train.sample_indices, validation.sample_indices).size
    assert train.joint_names == validation.joint_names == tuple(report["joint_names"])
    assert report["joint_column_identity"]["verified"]
    assert len(list((output / "captures/train/frames").glob("*.png"))) >= 24
    assert len(list((output / "captures/validation/frames").glob("*.png"))) >= 10


def test_saved_pngs_reimport_to_same_real_board_observations(
    completed_experiment, tmp_path
):
    output, _ = completed_experiment
    reimported, report = import_captures(
        output / "captures/train/manifest.json", tmp_path / "reimport.npz"
    )
    original = KinematicDataset.load(output / "train-camera.npz")
    np.testing.assert_allclose(
        reimported.observed_poses, original.observed_poses, atol=1e-10
    )
    np.testing.assert_array_equal(reimported.joint_positions, original.joint_positions)
    assert report["all_samples_accepted"]
    # The archive binds all PNGs and measurements, not just aggregate metrics.
    manifest = json.loads((output / "manifest.json").read_text())
    for filename, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest


def test_no_image_detection_never_falls_back_to_ideal_fk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        image_simulation, "detect_opencv_circles", lambda *args, **kwargs: None
    )
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="collected 0/6"):
        run_image_experiment(
            output,
            samples=6,
            validation_samples=5,
            max_candidates=80,
            noise=ImageNoise(0, 0, 1),
        )
    assert not output.exists()


def test_cctag_requires_explicit_external_centers_mode(tmp_path):
    with pytest.raises(ValueError, match="external-centers"):
        run_image_experiment(
            tmp_path / "invalid", board=get_board_config("cctag-multi")
        )
    output = tmp_path / "centers"
    report = run_image_experiment(
        output,
        board=get_board_config("cctag-multi"),
        measurement_mode="external-centers",
        samples=20,
        validation_samples=8,
        noise=ImageNoise(0, 0, 1),
    )
    assert report["measurement_model"] == "simulated-external-centers"
    manifest = json.loads((output / "captures/train/manifest.json").read_text())
    assert "image_points" in manifest["samples"][0]
    assert "image" not in manifest["samples"][0]
    assert list((output / "captures/train/frames").glob("*.png"))


def test_existing_nonempty_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-file.txt"
    marker.write_text("retain this")
    with pytest.raises(FileExistsError):
        run_image_experiment(output, samples=6, validation_samples=5)
    assert marker.read_text() == "retain this"


def _copied_experiment(completed_experiment, tmp_path):
    import shutil

    destination = tmp_path / "moved-bundle"
    shutil.copytree(completed_experiment[0], destination)
    return destination


def _refresh_hash_manifest(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = hashlib.sha256(
            (path / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_complete_image_experiment_is_portable_and_recomputes_viewer_predictions(
    completed_experiment, tmp_path
):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    report, dataset, model = load_image_experiment(path)
    assert report["accepted"]
    for name in ("nominal_predictions", "calibrated_predictions", "truth_predictions"):
        assert model[name].shape == dataset.observed_poses.shape
    assert (
        np.max(
            np.linalg.norm(
                model["calibrated_predictions"][:, :3, 3]
                - model["truth_predictions"][:, :3, 3],
                axis=1,
            )
        )
        < 0.002
    )


def test_image_loader_rejects_changed_png_hash(completed_experiment, tmp_path):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    image = next((path / "captures/validation/frames").glob("*.png"))
    with image.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_image_experiment(path)


def test_image_loader_rejects_rehashed_valid_model_with_inconsistent_metrics(
    completed_experiment, tmp_path
):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["home"][:3, 3] += [0.01, 0.02, -0.03]
    np.savez_compressed(path / "model.npz", **arrays)
    _refresh_hash_manifest(path)
    with pytest.raises(ValueError, match="does not match saved predictions"):
        load_image_experiment(path)


def test_image_loader_rejects_rehashed_capture_row_order(
    completed_experiment, tmp_path
):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    capture_path = path / "captures/validation/manifest.json"
    capture = json.loads(capture_path.read_text())
    capture["samples"].reverse()
    capture_path.write_text(json.dumps(capture))
    _refresh_hash_manifest(path)
    with pytest.raises(ValueError, match="sample IDs/order"):
        load_image_experiment(path)


def test_image_loader_rejects_manifest_paths_outside_bundle(
    completed_experiment, tmp_path
):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["../outside.png"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical relative path"):
        load_image_experiment(path)


def test_image_loader_rejects_capture_preview_traversal_after_rehash(
    completed_experiment, tmp_path
):
    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    capture_path = path / "captures/validation/manifest.json"
    capture = json.loads(capture_path.read_text())
    capture["samples"][0]["image"] = "../../nominal.urdf"
    capture_path.write_text(json.dumps(capture))
    _refresh_hash_manifest(path)
    with pytest.raises(ValueError, match="canonical relative path"):
        load_image_experiment(path)


def test_image_loader_redetects_saved_pixels_even_if_hashes_are_refreshed(
    completed_experiment, tmp_path
):
    import cv2

    from kinematic_calibration.image_simulation import load_image_experiment

    path = _copied_experiment(completed_experiment, tmp_path)
    collection_path = path / "captures/validation/collection.json"
    collection = json.loads(collection_path.read_text())
    attempt = next(row for row in collection["rendered_attempts"] if row["accepted"])
    image = path / "captures/validation" / attempt["image"]
    cv2.imwrite(str(image), np.zeros((960, 1280, 3), np.uint8))
    attempt["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    collection_path.write_text(json.dumps(collection))
    _refresh_hash_manifest(path)
    with pytest.raises(ValueError, match="no longer yields a board detection"):
        load_image_experiment(path)
