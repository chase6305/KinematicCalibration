import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset, ObservationFrame
from kinematic_calibration.acquisition import CaptureImportError, import_captures
from kinematic_calibration.targets import get_board_config
from kinematic_calibration.targets.detection import BoardDetection

cv2 = pytest.importorskip("cv2")


def _manifest(tmp_path):
    board = get_board_config("cctag-multi")
    camera = np.array([[800, 0, 320], [0, 820, 240], [0, 0, 1]], float)
    distortion = np.array([0.01, -0.02, 0.001, -0.002, 0.005])
    poses = []
    samples = []
    for index, sample_id in enumerate([11, 23, 35]):
        vector = np.array([0.2 + 0.01 * index, -0.1, 0.05])
        translation = np.array([0.02 + 0.01 * index, -0.03, 0.9 + 0.02 * index])
        pixels, _ = cv2.projectPoints(
            board.points, vector, translation, camera, distortion
        )
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
        pose[:3, 3] = translation
        poses.append(pose)
        samples.append(
            {
                "sample_id": sample_id,
                "joint_positions": [0.1 + 0.2 * index, -0.2, 0.3, -0.4, 0.5, -0.6],
                "joint_timestamp_s": 1000.0 + index,
                "image_timestamp_s": 1000.005 + index,
                "point_ids": list(range(len(board.points))),
                "image_points": pixels[:, 0].tolist(),
            }
        )
    value = {
        "schema_version": 1,
        "camera_matrix": camera.tolist(),
        "distortion": distortion.tolist(),
        "board_type": "cctag-multi",
        "observation_frame": "camera_T_target",
        "joint_names": [f"joint_{index}" for index in range(6)],
        "samples": samples,
    }
    path = tmp_path / "captures.json"
    path.write_text(json.dumps(value))
    return path, value, np.asarray(poses)


def _write(path, value):
    path.write_text(json.dumps(value))


def test_external_centers_recover_measured_board_poses_and_encoder_identity(tmp_path):
    path, manifest, expected = _manifest(tmp_path)
    output = tmp_path / "measurements.npz"
    dataset, report = import_captures(path, output)

    assert dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET
    np.testing.assert_allclose(dataset.observed_poses, expected, atol=1e-7)
    np.testing.assert_allclose(
        dataset.joint_positions, [row["joint_positions"] for row in manifest["samples"]]
    )
    assert dataset.sample_indices.tolist() == [11, 23, 35]
    assert report["all_samples_accepted"]
    assert report["dataset_written"]
    assert report["rejected_samples"] == 0
    assert all(row["detected"] and not row["rejected"] for row in report["frames"])
    assert max(row["reprojection_rms_px"] for row in report["frames"]) < 1e-6
    assert report["dataset_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = KinematicDataset.load(output)
    np.testing.assert_array_equal(saved.sample_indices, dataset.sample_indices)
    assert json.loads(output.with_suffix(".report.json").read_text()) == report


def test_strict_batch_reports_every_frame_without_dropping_rejections(tmp_path):
    path, manifest, _ = _manifest(tmp_path)
    manifest["samples"][1]["image_timestamp_s"] += 0.1
    manifest["samples"][2]["image_points"][1][0] += 30
    _write(path, manifest)
    output = tmp_path / "measurements.npz"

    with pytest.raises(CaptureImportError) as caught:
        import_captures(path, output)
    report = caught.value.report
    assert not output.exists()
    assert caught.value.report_path == output.with_suffix(".rejected.json")
    assert report["total_samples"] == 3
    assert report["accepted_samples"] == 1
    assert report["rejected_samples"] == 2
    assert not report["dataset_written"]
    assert not report["frames"][1]["detection_attempted"]
    assert "timestamp" in report["frames"][1]["rejection_reasons"][0]
    assert report["frames"][2]["detected"]
    assert report["frames"][2]["reprojection_rms_px"] > 2


def test_partial_import_requires_explicit_choice_and_retains_sample_ids(tmp_path):
    path, manifest, expected = _manifest(tmp_path)
    manifest["samples"][1]["image_timestamp_s"] += 0.1
    _write(path, manifest)
    dataset, report = import_captures(
        path, tmp_path / "accepted.npz", allow_partial=True
    )
    assert dataset.sample_indices.tolist() == [11, 35]
    np.testing.assert_allclose(dataset.observed_poses, expected[[0, 2]], atol=1e-7)
    assert report["partial"]
    assert report["allow_partial"]
    assert not report["all_samples_accepted"]
    assert report["frames"][2]["dataset_row"] == 1


def test_strict_rejection_preserves_a_previously_accepted_dataset_and_report(tmp_path):
    path, manifest, _ = _manifest(tmp_path)
    output = tmp_path / "measurements.npz"
    import_captures(path, output)
    previous_data = output.read_bytes()
    previous_report = output.with_suffix(".report.json").read_bytes()
    manifest["samples"][1]["image_timestamp_s"] += 0.1
    _write(path, manifest)
    with pytest.raises(CaptureImportError):
        import_captures(path, output, force=True)
    assert output.read_bytes() == previous_data
    assert output.with_suffix(".report.json").read_bytes() == previous_report


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda m: m["samples"][1].update(sample_id=11), "duplicate sample_id"),
        (lambda m: m["samples"][0].update(sample_id=True), "sample_id"),
        (lambda m: m["samples"][0].update(sample_id=-1), "sample_id"),
        (lambda m: m["samples"][0].update(sample_id="11"), "sample_id"),
        (lambda m: m["samples"][0].update(sample_id=2**64), "sample_id"),
        (lambda m: m.update(observation_frame="base_T_ee"), "observation_frame"),
        (
            lambda m: m["samples"][0].update(observation_frame="base_T_ee"),
            "observation_frame",
        ),
        (
            lambda m: m["samples"][0].update(board_type="opencv-acircle"),
            "board identity",
        ),
        (lambda m: m["samples"][0].update(point_ids=[0, 1, 2, 3, 4, 4]), "point_ids"),
        (lambda m: m["samples"][0].update(joint_positions=[1, 2]), "column count"),
        (
            lambda m: m["samples"][0].update(joint_positions=[float("nan")] * 6),
            "joint_positions",
        ),
        (
            lambda m: m["samples"][0].update(joint_timestamp_s=float("nan")),
            "timestamp_s",
        ),
        (lambda m: m["samples"][0].update(image_timestamp_s=True), "timestamp_s"),
        (lambda m: m["samples"][0].update(image="also.png"), "exactly one"),
        (lambda m: m.update(joint_names=["duplicate"] * 6), "joint_names"),
        (lambda m: m.update(joint_unit="deg"), "units"),
        (lambda m: m.update(camera_matrix=np.eye(2).tolist()), "intrinsic"),
        (lambda m: m.update(camera_convention="opengl"), "camera_convention"),
    ],
)
def test_identity_or_structural_errors_reject_before_any_output(
    tmp_path, change, message
):
    path, manifest, _ = _manifest(tmp_path)
    change(manifest)
    _write(path, manifest)
    with pytest.raises(ValueError, match=message):
        import_captures(path, tmp_path / "measurements.npz")
    assert [entry.name for entry in tmp_path.iterdir()] == ["captures.json"]


def test_actual_circle_grid_image_is_decoded_detected_and_measured(tmp_path):
    board = get_board_config("opencv-acircle")
    image = np.zeros((720, 900, 3), np.uint8)
    for x, y, _ in board.points:
        pixel = tuple(
            np.rint(([x, y] + np.asarray(board.origin_from_top_left_m)) * 3000).astype(
                int
            )
        )
        cv2.circle(image, pixel, 24, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    image_path = tmp_path / "captured-grid.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = {
        "camera_matrix": [[1500, 0, 450], [0, 1500, 360], [0, 0, 1]],
        "distortion": [0, 0, 0, 0, 0],
        "board_type": "opencv-acircle",
        "joint_names": ["joint_1", "joint_2"],
        "samples": [
            {
                "sample_id": 41,
                "joint_positions": [0.2, -0.4],
                "image": image_path.name,
                "joint_timestamp_s": 10.0,
                "image_timestamp_s": 10.005,
            }
        ],
    }
    path = tmp_path / "captures.json"
    _write(path, manifest)
    dataset, report = import_captures(path, tmp_path / "measured.npz")
    assert report["frames"][0]["source"] == "image"
    assert report["frames"][0]["detected"]
    assert (
        report["frames"][0]["image_sha256"]
        == hashlib.sha256(image_path.read_bytes()).hexdigest()
    )
    assert report["frames"][0]["reprojection_rms_px"] < 0.1
    assert dataset.observed_poses[0, 2, 3] == pytest.approx(0.5, abs=1e-4)


def test_missing_image_and_no_detections_have_explicit_rejection_reports(tmp_path):
    path, manifest, _ = _manifest(tmp_path)
    for index, sample in enumerate(manifest["samples"]):
        sample.pop("image_points")
        sample["image"] = f"missing-{index}.png"
    _write(path, manifest)
    with pytest.raises(CaptureImportError) as caught:
        import_captures(path, tmp_path / "empty.npz", allow_partial=True)
    assert caught.value.report["accepted_samples"] == 0
    assert caught.value.report["rejected_samples"] == 3
    assert all(row["rejection_reasons"] for row in caught.value.report["frames"])
    assert not (tmp_path / "empty.npz").exists()


def test_toolkit_pose_without_reprojection_evidence_is_explicitly_rejected(
    tmp_path, monkeypatch
):
    import kinematic_calibration.acquisition as acquisition

    path, manifest, _ = _manifest(tmp_path)
    manifest["samples"] = manifest["samples"][:1]
    sample = manifest["samples"][0]
    sample.pop("image_points")
    sample["image"] = "captured.png"
    _write(path, manifest)
    pose = np.eye(4)
    pose[2, 3] = 1
    monkeypatch.setattr(
        acquisition, "_read_image", lambda _: (np.zeros((10, 10), np.uint8), "hash")
    )
    monkeypatch.setattr(
        acquisition,
        "detect_toolkit_board",
        lambda *a, **kw: BoardDetection(pose, None, None, "toolkit"),
    )
    with pytest.raises(CaptureImportError) as caught:
        import_captures(path, tmp_path / "measured.npz", detector="toolkit")
    row = caught.value.report["frames"][0]
    assert row["detected"] and row["rejected"]
    assert row["reprojection_rms_px"] is None
    assert "no reprojection error" in row["rejection_reasons"][0]


def test_source_images_cannot_be_reused_for_multiple_joint_readings(tmp_path):
    path, manifest, _ = _manifest(tmp_path)
    for sample in manifest["samples"]:
        sample.pop("image_points")
        sample["image"] = "same.png"
    _write(path, manifest)
    with pytest.raises(ValueError, match="reuses an image"):
        import_captures(path, tmp_path / "measured.npz")


def test_output_conflicts_are_detected_before_any_existing_data_is_modified(tmp_path):
    path, _, _ = _manifest(tmp_path)
    output = tmp_path / "measured.npz"
    output.write_bytes(b"previous valid data")
    with pytest.raises(FileExistsError):
        import_captures(path, output)
    assert output.read_bytes() == b"previous valid data"
    output.with_suffix(".report.json").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        import_captures(path, output, force=True)
    assert output.read_bytes() == b"previous valid data"


def test_pair_publish_failure_restores_existing_dataset_and_report(
    tmp_path, monkeypatch
):
    import kinematic_calibration.acquisition as acquisition

    path, _, _ = _manifest(tmp_path)
    output = tmp_path / "measured.npz"
    import_captures(path, output)
    original = {
        name: name.read_bytes() for name in (output, output.with_suffix(".report.json"))
    }
    original_replace = acquisition.os.replace
    armed = True

    def fail_once(source, destination):
        nonlocal armed
        if armed and destination == output.with_suffix(".report.json"):
            armed = False
            raise OSError("injected report installation failure")
        return original_replace(source, destination)

    monkeypatch.setattr(acquisition.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected report installation failure"):
        import_captures(path, output, force=True)
    assert {name: name.read_bytes() for name in original} == original


def test_ambiguous_planar_pose_is_reported_and_rejected_even_with_low_rms(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import kinematic_calibration.acquisition as acquisition

    path, _, expected = _manifest(tmp_path)
    diagnostics = {
        "evaluated": True,
        "ambiguous": True,
        "best_candidate_rms_px": 0.1,
        "alternative_candidate_rms_px": 0.11,
    }
    monkeypatch.setattr(
        acquisition,
        "pose_from_correspondences",
        lambda *a, **kw: SimpleNamespace(
            camera_T_board=expected[0],
            backend="opencv-pnp",
            reprojection_rms_px=0.1,
            pose_diagnostics=diagnostics,
        ),
    )
    with pytest.raises(CaptureImportError) as caught:
        import_captures(path, tmp_path / "ambiguous.npz")
    assert caught.value.report["rejected_samples"] == 3
    row = caught.value.report["frames"][0]
    assert row["detected"] and row["rejected"]
    assert row["reprojection_rms_px"] == 0.1
    assert row["pose_diagnostics"] == diagnostics
    assert "ambiguous" in row["rejection_reasons"][0]


def test_import_requires_and_preserves_declared_joint_column_names(tmp_path):
    path, manifest, _ = _manifest(tmp_path)
    output = tmp_path / "named.npz"
    dataset, _ = import_captures(path, output)
    assert dataset.joint_names == tuple(manifest["joint_names"])
    assert KinematicDataset.load(output).joint_names == dataset.joint_names
    manifest.pop("joint_names")
    _write(path, manifest)
    with pytest.raises(ValueError, match="joint_names"):
        import_captures(path, tmp_path / "missing-names.npz")
    assert not (tmp_path / "missing-names.npz").exists()
