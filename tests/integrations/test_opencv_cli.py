import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")

from kinematic_calibration.handeye_cli import main


def make_pose(rotation_vector, translation) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    pose[:3, 3] = translation
    return pose


def test_offline_opencv_cli_solves_archived_observations(
    tmp_path: Path, capsys
) -> None:
    base_T_camera = make_pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    ee_T_board = make_pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    base_T_ee = np.stack(
        [
            make_pose(
                [0.08 * index, -0.05 * index**2, 0.04 * (index + 1)],
                [0.1 * np.sin(index), 0.08 * np.cos(index), 0.35 + 0.03 * index],
            )
            for index in range(1, 10)
        ]
    )
    camera_T_board = np.stack(
        [np.linalg.inv(base_T_camera) @ pose @ ee_T_board for pose in base_T_ee]
    )
    samples = tmp_path / "samples.npz"
    np.savez_compressed(
        samples,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=base_T_ee,
        camera_T_board=camera_T_board,
        candidate_indices=np.arange(9),
        translation_candidate_count=np.asarray(6),
        rotation_candidate_count=np.asarray(3),
    )
    output = tmp_path / "result.json"

    assert (
        main(
            [
                "solve",
                "--samples",
                str(samples),
                "--output",
                str(output),
                "--method",
                "li",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["accepted"] is True
    assert report["method"] == "li"
    assert report["rotation_singular_value_2_rad"] > 0.0
    assert report["rotation_direction_ratio"] > 0.01
    assert len(report["observations_npz_sha256"]) == 64
    assert len(report["sample_errors"]) == 9
    assert report["position_error_worst_candidate_index"] in range(9)
    np.testing.assert_allclose(report["Base_T_Camera"], base_T_camera, atol=1e-7)
    assert (
        main(
            [
                "verify",
                "--samples",
                str(samples),
                "--result",
                str(output),
            ]
        )
        == 0
    )
    base_output = tmp_path / "base_T_camera.npy"
    target_output = tmp_path / "ee_T_target.npy"
    assert (
        main(
            [
                "export",
                "--samples",
                str(samples),
                "--result",
                str(output),
                "--base-t-camera",
                str(base_output),
                "--ee-t-target",
                str(target_output),
            ]
        )
        == 0
    )
    np.testing.assert_allclose(
        np.load(base_output, allow_pickle=False), base_T_camera, atol=1e-7
    )
    np.testing.assert_allclose(
        np.load(target_output, allow_pickle=False), ee_T_board, atol=1e-7
    )
    assert (
        main(
            [
                "export",
                "--samples",
                str(samples),
                "--result",
                str(output),
                "--base-t-camera",
                str(base_output),
                "--ee-t-target",
                str(target_output),
            ]
        )
        == 2
    )
    report["position_error_max_m"] = 1.0
    output.write_text(json.dumps(report))
    assert (
        main(
            [
                "verify",
                "--samples",
                str(samples),
                "--result",
                str(output),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "export",
                "--samples",
                str(samples),
                "--result",
                str(output),
                "--base-t-camera",
                str(base_output),
                "--ee-t-target",
                str(target_output),
                "--force",
            ]
        )
        == 2
    )
    comparison = tmp_path / "comparison.json"
    assert (
        main(
            [
                "compare",
                "--samples",
                str(samples),
                "--output",
                str(comparison),
            ]
        )
        == 0
    )
    comparison_report = json.loads(comparison.read_text())
    assert comparison_report["accepted"] is True
    assert comparison_report["recommended_method"] in {"shah", "li"}
    assert (
        comparison_report["method_differences"]["Base_T_Camera"]["translation_m"] < 1e-7
    )
    capsys.readouterr()
    assert main(["inspect", "--samples", str(samples)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["report_type"] == "opencv-eye-to-hand-observation-inspection"
    assert inspection["accepted"] is True
    influence = tmp_path / "influence.json"
    assert (
        main(
            [
                "influence",
                "--samples",
                str(samples),
                "--output",
                str(influence),
                "--method",
                "shah",
            ]
        )
        == 0
    )
    influence_report = json.loads(influence.read_text())
    assert len(influence_report["sample_influence"]) == 9
    assert influence_report["most_influential_candidate_index"] in range(9)
    bootstrap = tmp_path / "bootstrap.json"
    assert (
        main(
            [
                "bootstrap",
                "--samples",
                str(samples),
                "--output",
                str(bootstrap),
                "--replicates",
                "20",
                "--seed",
                "7",
                "--min-success-rate",
                "0.5",
            ]
        )
        == 0
    )
    bootstrap_report = json.loads(bootstrap.read_text())
    assert bootstrap_report["accepted"] is True
    assert bootstrap_report["replicates"] == 20
    assert bootstrap_report["stability_statistics"]["translation_p95_m"] < 1e-6


def test_offline_opencv_cli_rejects_unknown_archive(tmp_path: Path) -> None:
    samples = tmp_path / "samples.npz"
    np.savez_compressed(samples, schema_version=np.asarray(1))

    assert (
        main(
            [
                "solve",
                "--samples",
                str(samples),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 2
    )


def test_opencv_inspect_rejects_single_axis_motion(tmp_path: Path, capsys) -> None:
    poses = np.stack(
        [
            make_pose([0.0, 0.0, 0.1 * index], [0.1 * index, 0.0, 0.0])
            for index in range(4)
        ]
    )
    samples = tmp_path / "samples.npz"
    np.savez_compressed(
        samples,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=poses,
        camera_T_board=poses,
        candidate_indices=np.arange(4),
        translation_candidate_count=np.asarray(2),
        rotation_candidate_count=np.asarray(2),
    )

    assert main(["inspect", "--samples", str(samples)]) == 1
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["accepted"] is False
    assert "motion rotation rank is below threshold" in inspection["quality_failures"]
