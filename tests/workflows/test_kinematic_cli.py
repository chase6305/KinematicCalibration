import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kinematic_calib.cli import (
    _dataset_sample_sha256,
    _evaluate_solution_quality,
    _implementation_sha256,
    _json_dumps_fixed,
    _jsonable,
    _load_transform,
    _merge_dataset_quality_gate,
    _refuse_existing_outputs,
    _solver_dataset_quality,
    _urdf_joint_names_by_child,
    _validate_quality_threshold_args,
    build_parser,
    main,
)
from kinematic_calib.dataset import KinematicDataset, ObservationFrame


def test_kinematic_cli_requires_explicit_model_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "solve",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output",
            "result.npz",
            "--model-mode",
            "robot-body",
        ]
    )

    assert args.model_mode == "robot-body"


def test_self_calibrating_mode_does_not_require_external_transforms() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "solve",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output",
            "result.npz",
            "--model-mode",
            "self-calibrating",
        ]
    )

    assert args.model_mode == "self-calibrating"
    assert args.base_t_camera is None
    assert args.ee_t_target is None


def test_benchmark_cli_accepts_measurement_noise_configuration() -> None:
    args = build_parser().parse_args(
        [
            "benchmark",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output",
            "benchmark.json",
            "--position-noise-std-m",
            "0.002",
            "--orientation-noise-std-rad",
            "0.005",
            "--joint-noise-std-rad",
            "0.0005",
            "--outlier-ratio",
            "0.05",
            "--outlier-position-std-m",
            "0.02",
            "--outlier-orientation-std-rad",
            "0.05",
        ]
    )

    assert args.position_noise_std_m == 0.002
    assert args.orientation_noise_std_rad == 0.005
    assert args.joint_noise_std_rad == 0.0005
    assert args.outlier_ratio == 0.05


def test_urdf_joint_names_are_mapped_from_child_links(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><link name="base"/><link name="link1"/>'
        '<joint name="Joint1" type="revolute"><parent link="base"/>'
        '<child link="link1"/><axis xyz="0 0 1"/></joint></robot>'
    )

    assert _urdf_joint_names_by_child(urdf) == {"link1": "Joint1"}


def test_zero_offset_parser_supports_self_calibrating_mode() -> None:
    args = build_parser().parse_args(
        [
            "zero-offset",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--model-mode",
            "self-calibrating",
            "--output",
            "offsets.json",
        ]
    )

    assert args.command == "zero-offset"
    assert args.base_t_camera is None
    assert args.ee_t_target is None


def test_validate_urdf_truth_parser_defaults_to_300_samples() -> None:
    args = build_parser().parse_args(
        [
            "validate-urdf-truth",
            "--nominal-urdf",
            "nominal.urdf",
            "--calibrated-urdf",
            "calibrated.urdf",
            "--ground-truth-urdf",
            "truth.urdf",
            "--dataset",
            "poses.npz",
            "--output",
            "validation.json",
        ]
    )

    assert args.samples == 300
    assert args.seed == 1234
    assert args.sampling_domain == "joint-limits"


def test_unified_calibrate_parser_defaults_to_both_and_automatic_handeye() -> None:
    args = build_parser().parse_args(
        [
            "calibrate",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output-dir",
            "outputs",
            "--handeye-samples",
            "handeye.npz",
        ]
    )

    assert args.command == "calibrate"
    assert args.calibration_type == "both"
    assert args.no_use_handeye_extrinsics is False


def test_unified_calibrate_minimal_input_uses_concise_defaults() -> None:
    args = build_parser().parse_args(
        ["calibrate", "--dataset", "dataset/processed/calibration_poses.npz"]
    )

    assert args.urdf is None
    assert args.ee_name == "ee_link"
    assert args.output_dir is None
    assert args.output_detail == "concise"
    assert args.skip_handeye is False


def test_unified_calibrate_without_handeye_uses_self_calibrating_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_modes = []
    selected_urdfs = []

    def fake_solve(args):
        selected_modes.append(("geometry", args.model_mode))
        selected_urdfs.append(("geometry", args.urdf))
        return 0

    def fake_zero_offset(args):
        selected_modes.append(("zero-offset", args.model_mode))
        selected_urdfs.append(("zero-offset", args.urdf))
        return 0

    monkeypatch.setattr("kinematic_calib.cli._solve", fake_solve)
    monkeypatch.setattr("kinematic_calib.cli._zero_offset", fake_zero_offset)
    monkeypatch.setattr("kinematic_calib.cli._verify_result", lambda args: 0)
    monkeypatch.setattr("kinematic_calib.cli._export_urdf", lambda args: 0)

    output_dir = tmp_path / "output"
    KinematicDataset(
        np.zeros((2, 1)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
    ).save(tmp_path / "poses.npz")
    (tmp_path / "robot.urdf").touch()
    assert (
        main(
            [
                "calibrate",
                "--dataset",
                str(tmp_path / "poses.npz"),
                "--urdf",
                str(tmp_path / "robot.urdf"),
                "--ee-name",
                "tool0",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert selected_modes == [
        ("geometry", "self-calibrating"),
        ("zero-offset", "self-calibrating"),
    ]
    assert selected_urdfs == [
        ("geometry", tmp_path / "robot.urdf"),
        ("zero-offset", output_dir / "robot_self_calibrated.urdf"),
    ]
    manifest = json.loads((output_dir / "calibration_manifest.json").read_text())
    assert manifest["selected_model_mode"] == "self-calibrating"
    assert manifest["handeye"]["extrinsics_used"] is False
    assert manifest["stages"]["zero_offset"]["source_urdf"] == str(
        output_dir / "robot_self_calibrated.urdf"
    )


def test_unified_calibrate_passes_accepted_handeye_to_all_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cv2", reason="OpenCV hand-eye adapter is optional")
    import kinematic_calibration.handeye_cli as opencv_cli

    selected = []
    identity = np.eye(4)
    handeye_samples = tmp_path / "handeye.npz"
    handeye_samples.touch()
    KinematicDataset(
        np.zeros((2, 1)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
    ).save(tmp_path / "poses.npz")
    (tmp_path / "robot.urdf").touch()

    monkeypatch.setattr(opencv_cli, "_solve", lambda args: 0)
    monkeypatch.setattr(
        opencv_cli,
        "_load_verified_result",
        lambda samples, result: (
            {"Base_T_Camera": identity.tolist(), "EE_T_Board": identity.tolist()},
            {},
            True,
        ),
    )

    def fake_stage(args):
        selected.append(
            (args.model_mode, args.base_t_camera.name, args.ee_t_target.name)
        )
        return 0

    monkeypatch.setattr("kinematic_calib.cli._solve", fake_stage)
    monkeypatch.setattr("kinematic_calib.cli._zero_offset", fake_stage)
    monkeypatch.setattr("kinematic_calib.cli._verify_result", lambda args: 0)
    monkeypatch.setattr("kinematic_calib.cli._export_urdf", lambda args: 0)

    output_dir = tmp_path / "output"
    assert (
        main(
            [
                "calibrate",
                "--dataset",
                str(tmp_path / "poses.npz"),
                "--urdf",
                str(tmp_path / "robot.urdf"),
                "--ee-name",
                "tool0",
                "--output-dir",
                str(output_dir),
                "--handeye-samples",
                str(handeye_samples),
            ]
        )
        == 0
    )

    assert selected == [
        ("robot-body", "base_T_camera.npy", "ee_T_target.npy"),
        ("robot-body", "base_T_camera.npy", "ee_T_target.npy"),
    ]
    assert np.array_equal(np.load(output_dir / "base_T_camera.npy"), identity)
    assert np.array_equal(np.load(output_dir / "ee_T_target.npy"), identity)


def test_unified_geometry_only_ignores_stale_zero_offset_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "zero_offsets.json").write_text(
        json.dumps({"accepted": True, "joints": []})
    )
    urdf = tmp_path / "robot.urdf"
    urdf.touch()
    KinematicDataset(
        np.zeros((2, 1)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
    ).save(tmp_path / "poses.npz")

    def fake_solve(args):
        args.output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "position_error_mean_m": 0.0,
                    "position_error_max_m": 0.0,
                    "orientation_error_mean_rad": 0.0,
                    "orientation_error_max_rad": 0.0,
                    "urdf_joint_definitions": [],
                }
            )
        )
        return 0

    monkeypatch.setattr("kinematic_calib.cli._solve", fake_solve)
    monkeypatch.setattr("kinematic_calib.cli._verify_result", lambda args: 0)
    monkeypatch.setattr("kinematic_calib.cli._export_urdf", lambda args: 0)

    assert (
        main(
            [
                "calibrate",
                "--dataset",
                str(tmp_path / "poses.npz"),
                "--urdf",
                str(urdf),
                "--output-dir",
                str(output_dir),
                "--calibration-type",
                "geometry",
                "--skip-handeye",
                "--force",
            ]
        )
        == 0
    )

    result = json.loads((output_dir / "final_result.json").read_text())
    assert "zero_offset_calibration" not in result


def test_transform_loader_rejects_non_rigid_matrix(tmp_path: Path) -> None:
    path = tmp_path / "transform.npy"
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    np.save(path, invalid)

    with pytest.raises(ValueError, match="not orthonormal"):
        _load_transform(path, "test_transform")


def test_quality_gate_rejects_large_residual() -> None:
    accepted, failures = _evaluate_solution_quality(
        converged=True,
        full_rank=True,
        position_error_max_m=0.2,
        orientation_error_max_rad=0.01,
        condition_number=100.0,
        max_position_error=0.01,
        max_orientation_error=0.05,
        max_condition_number=1e6,
    )

    assert not accepted
    assert any("position error" in failure for failure in failures)


@pytest.mark.parametrize(
    ("position_error", "orientation_error"),
    [(np.nan, 0.001), (0.001, np.inf)],
)
def test_quality_gate_rejects_non_finite_residual(
    position_error: float, orientation_error: float
) -> None:
    accepted, failures = _evaluate_solution_quality(
        converged=True,
        full_rank=True,
        position_error_max_m=position_error,
        orientation_error_max_rad=orientation_error,
        condition_number=100.0,
        max_position_error=0.01,
        max_orientation_error=0.05,
        max_condition_number=1e6,
    )

    assert not accepted
    assert any("not finite" in failure for failure in failures)


def test_jsonable_replaces_nested_non_finite_values() -> None:
    converted = _jsonable({"values": np.array([1.0, np.nan, np.inf])})

    assert converted == {"values": [1.0, None, None]}


def test_fixed_json_renderer_avoids_scientific_notation() -> None:
    rendered = _json_dumps_fixed(
        {"offsets": [-1.1285269311123652e-05, 4.311389675774905e-06]}
    )

    assert "e-" not in rendered.lower()
    assert "-0.00001128526931112" in rendered
    assert json.loads(rendered)["offsets"][1] == pytest.approx(4.311389675774905e-06)


def test_solver_dataset_quality_can_reject_low_diversity() -> None:
    dataset = KinematicDataset(
        np.zeros((5, 1)),
        np.repeat(np.eye(4)[None], 5, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
    )
    args = build_parser().parse_args(
        [
            "solve",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output",
            "result.npz",
            "--model-mode",
            "self-calibrating",
        ]
    )

    coverage, warnings = _solver_dataset_quality(dataset, args, "solve")

    assert coverage["unique_sample_count"] == 1
    assert len(warnings) == 4
    args.fail_on_low_diversity = True
    with pytest.raises(ValueError, match="diversity requirements failed"):
        _solver_dataset_quality(dataset, args, "solve")


def test_dataset_quality_warnings_prevent_result_acceptance() -> None:
    accepted, failures = _merge_dataset_quality_gate(
        True,
        [],
        ["target rotation coverage is below threshold"],
    )

    assert accepted is False
    assert failures == [
        "dataset diversity: target rotation coverage is below threshold"
    ]


def test_solver_option_preflight_rejects_invalid_values() -> None:
    args = build_parser().parse_args(
        [
            "solve",
            "--dataset",
            "poses.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "tool0",
            "--output",
            "result.npz",
            "--model-mode",
            "self-calibrating",
            "--max-condition-number",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="positive and finite"):
        _validate_quality_threshold_args(args, "solve")


def test_template_rejects_non_npz_output(tmp_path: Path) -> None:
    assert main(["template", "--output", str(tmp_path / "template.data")]) == 2


def test_material_output_guard_requires_force(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("existing")

    with pytest.raises(ValueError, match="use --force"):
        _refuse_existing_outputs((output,), False, "test")

    _refuse_existing_outputs((output,), True, "test")


def test_template_command_creates_valid_dataset(tmp_path: Path) -> None:
    output = tmp_path / "template.npz"

    assert main(["template", "--output", str(output), "--samples", "8"]) == 0

    dataset = KinematicDataset.load(output)
    assert dataset.joint_positions.shape == (8, 6)
    assert dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET


def test_inspect_command_writes_coverage_and_hash(tmp_path: Path) -> None:
    dataset_path = tmp_path / "template.npz"
    output_path = tmp_path / "inspection.json"
    assert main(["template", "--output", str(dataset_path)]) == 0

    assert (
        main(
            [
                "inspect",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text())
    assert report["samples"] == 20
    assert report["dataset_schema_version"] == 1
    assert report["legacy_dataset_schema"] is False
    assert len(report["dataset_sha256"]) == 64
    assert len(report["quality_warnings"]) == 4
    assert report["unique_sample_count"] == 1
    assert report["duplicate_sample_count"] == 19


def test_inspect_can_fail_on_low_diversity(tmp_path: Path) -> None:
    dataset_path = tmp_path / "template.npz"
    assert main(["template", "--output", str(dataset_path)]) == 0

    assert (
        main(
            [
                "inspect",
                "--dataset",
                str(dataset_path),
                "--fail-on-low-diversity",
            ]
        )
        == 1
    )


def test_split_dataset_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    joints = np.arange(60, dtype=float).reshape(10, 6)
    poses = np.repeat(np.eye(4)[None], 10, axis=0)
    source_indices = np.arange(100, 110)
    KinematicDataset(
        joints,
        poses,
        ObservationFrame.CAMERA_T_TARGET,
        sample_indices=source_indices,
    ).save(source)

    def run_split(prefix: str) -> dict:
        train = tmp_path / f"{prefix}_train.npz"
        validation = tmp_path / f"{prefix}_validation.npz"
        manifest = tmp_path / f"{prefix}_manifest.json"
        assert (
            main(
                [
                    "split-dataset",
                    "--dataset",
                    str(source),
                    "--train-output",
                    str(train),
                    "--validation-output",
                    str(validation),
                    "--manifest",
                    str(manifest),
                    "--seed",
                    "42",
                    "--validation-count",
                    "2",
                    "--search-attempts",
                    "3",
                ]
            )
            == 0
        )
        return json.loads(manifest.read_text())

    first = run_split("first")
    second = run_split("second")
    assert first["train_indices"] == second["train_indices"]
    assert first["validation_indices"] == second["validation_indices"]
    assert set(first["train_indices"]).isdisjoint(first["validation_indices"])
    assert sorted(first["train_indices"] + first["validation_indices"]) == list(
        range(10)
    )
    assert first["validation_samples"] == 2
    assert first["seed_requested"] == 42
    assert first["search_attempts_requested"] == 3
    assert 1 <= first["search_attempts_evaluated"] <= 3
    assert first["quality_warnings"]
    assert "coverage_thresholds" in first
    assert first["validation_quality_warnings"]
    assert first["validation_source_sample_indices"] == [
        int(source_indices[index]) for index in first["validation_indices"]
    ]


def test_split_dataset_can_fail_before_writing_low_diversity_subsets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    joints = np.arange(10, dtype=float)[:, None] * 0.1
    KinematicDataset(
        joints,
        np.repeat(np.eye(4)[None], 10, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
    ).save(source)
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    manifest = tmp_path / "split.json"

    assert (
        main(
            [
                "split-dataset",
                "--dataset",
                str(source),
                "--train-output",
                str(train),
                "--validation-output",
                str(validation),
                "--manifest",
                str(manifest),
                "--validation-count",
                "5",
                "--fail-on-low-diversity",
            ]
        )
        == 2
    )
    assert not train.exists()
    assert not validation.exists()
    assert not manifest.exists()


def test_deduplicate_dataset_preserves_first_occurrences(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[2:, 0, 3] = 0.2
    dataset = KinematicDataset(
        np.array([[0.0], [0.0], [0.2], [0.2]]),
        poses,
        ObservationFrame.CAMERA_T_TARGET,
        sample_indices=np.array([5, 8, 13, 21]),
    )
    dataset.save(source)
    output = tmp_path / "clean.npz"
    manifest_path = tmp_path / "deduplicate.json"

    assert (
        main(
            [
                "deduplicate-dataset",
                "--dataset",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    cleaned = KinematicDataset.load(output)
    np.testing.assert_array_equal(cleaned.sample_indices, [5, 13])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["removed_positions"] == [1, 3]
    assert manifest["removed_source_sample_indices"] == [8, 21]
    assert manifest["duplicate_of_positions"] == [0, 2]
    assert manifest["output_coverage"]["duplicate_sample_count"] == 0
    verify_arguments = [
        "verify-deduplication",
        "--manifest",
        str(manifest_path),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    assert main(verify_arguments) == 0
    manifest["removed_positions"] = [1]
    manifest_path.write_text(json.dumps(manifest))
    assert main(verify_arguments) == 2

    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    split_manifest = tmp_path / "split.json"
    assert (
        main(
            [
                "split-dataset",
                "--dataset",
                str(source),
                "--train-output",
                str(train),
                "--validation-output",
                str(validation),
                "--manifest",
                str(split_manifest),
                "--validation-count",
                "1",
            ]
        )
        == 2
    )


def test_verify_split_checks_partition_and_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    joints = np.arange(8, dtype=float)[:, None] * 0.1
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, 0, 3] = np.arange(8) * 0.01
    KinematicDataset(joints, poses, ObservationFrame.CAMERA_T_TARGET).save(source)
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    manifest = tmp_path / "split.json"
    assert (
        main(
            [
                "split-dataset",
                "--dataset",
                str(source),
                "--train-output",
                str(train),
                "--validation-output",
                str(validation),
                "--manifest",
                str(manifest),
                "--validation-count",
                "2",
            ]
        )
        == 0
    )
    arguments = [
        "verify-split",
        "--manifest",
        str(manifest),
        "--source",
        str(source),
        "--train",
        str(train),
        "--validation",
        str(validation),
    ]
    assert main(arguments) == 0
    split_report = json.loads(manifest.read_text())
    original_seed = split_report["seed"]
    split_report["seed"] = original_seed + 100
    manifest.write_text(json.dumps(split_report))
    assert main(arguments) == 2
    split_report["seed"] = original_seed
    manifest.write_text(json.dumps(split_report))
    assert main(arguments) == 0
    damaged = KinematicDataset.load(validation)
    damaged_poses = damaged.observed_poses.copy()
    damaged_poses[0, 0, 3] += 0.001
    KinematicDataset(
        damaged.joint_positions,
        damaged_poses,
        damaged.observation_frame,
        damaged.sample_indices,
    ).save(validation)
    assert main(arguments) == 2


def test_upgrade_dataset_adds_schema_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "legacy.npz"
    np.savez_compressed(
        source,
        joint_positions=np.zeros((3, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 3, axis=0),
        observation_frame=np.asarray("camera_T_target"),
    )
    output = tmp_path / "upgraded.npz"
    manifest = tmp_path / "upgrade.json"

    assert (
        main(
            [
                "upgrade-dataset",
                "--dataset",
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )

    with np.load(output, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == 1
        np.testing.assert_array_equal(archive["sample_indices"], [0, 1, 2])
    report = json.loads(manifest.read_text())
    assert report["source_schema_version"] is None
    assert report["output_schema_version"] == 1
    assert report["source_dataset_sha256"] != report["output_dataset_sha256"]


def test_verify_result_checks_all_artifact_hashes(tmp_path: Path) -> None:
    dataset = tmp_path / "poses.npz"
    urdf = tmp_path / "robot.urdf"
    parameters = tmp_path / "result.npz"
    source_dataset = KinematicDataset(
        np.zeros((2, 1)),
        np.stack([np.eye(4), np.eye(4)]),
        ObservationFrame.CAMERA_T_TARGET,
        sample_indices=np.asarray([4, 9]),
    )
    source_dataset.save(dataset)
    urdf.write_bytes(b"urdf")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    implementation = _implementation_sha256()
    np.savez_compressed(
        parameters,
        schema_version=np.asarray(1),
        parameter_type=np.asarray("kinematic-poe"),
        model_mode=np.asarray("self-calibrating"),
        source_dataset_sha256=np.asarray(digest(dataset)),
        source_urdf_sha256=np.asarray(digest(urdf)),
        implementation_sha256=np.asarray(implementation),
        screw_axes=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        zero_configuration_pose=np.eye(4),
        observation_frame=np.asarray("camera_T_target"),
        source_observation_frame=np.asarray("camera_T_target"),
        source_sample_sha256=_dataset_sample_sha256(source_dataset),
        source_sample_indices=source_dataset.sample_indices,
    )

    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted": True,
                "calibration_type": "kinematic-geometry",
                "model_mode": "self-calibrating",
                "observation_frame": "camera_T_target",
                "source_observation_frame": "camera_T_target",
                "dof": 1,
                "dataset_sha256": digest(dataset),
                "urdf_sha256": digest(urdf),
                "parameters_npz_sha256": digest(parameters),
                "provenance": {"implementation_sha256": implementation},
            }
        )
    )
    assert (
        main(
            [
                "verify-result",
                "--result",
                str(result),
                "--dataset",
                str(dataset),
                "--urdf",
                str(urdf),
                "--parameters",
                str(parameters),
            ]
        )
        == 0
    )


def test_verify_result_rejects_hash_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "poses.npz"
    urdf = tmp_path / "robot.urdf"
    dataset.write_bytes(b"dataset")
    urdf.write_bytes(b"urdf")
    result = tmp_path / "offsets.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "dataset_sha256": "0" * 64,
                "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            }
        )
    )
    assert (
        main(
            [
                "verify-result",
                "--result",
                str(result),
                "--dataset",
                str(dataset),
                "--urdf",
                str(urdf),
            ]
        )
        == 2
    )


def test_verify_result_checks_robot_body_external_transforms(tmp_path: Path) -> None:
    dataset = tmp_path / "poses.npz"
    urdf = tmp_path / "robot.urdf"
    base_t_camera = tmp_path / "base_T_camera.npy"
    ee_t_target = tmp_path / "ee_T_target.npy"
    dataset.write_bytes(b"dataset")
    urdf.write_bytes(b"urdf")
    np.save(base_t_camera, np.eye(4))
    np.save(ee_t_target, np.eye(4))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted": True,
                "calibration_type": "kinematic-geometry",
                "model_mode": "robot-body",
                "dataset_sha256": digest(dataset),
                "urdf_sha256": digest(urdf),
                "base_T_camera_sha256": digest(base_t_camera),
                "ee_T_target_sha256": digest(ee_t_target),
            }
        )
    )
    arguments = [
        "verify-result",
        "--result",
        str(result),
        "--dataset",
        str(dataset),
        "--urdf",
        str(urdf),
        "--base-t-camera",
        str(base_t_camera),
        "--ee-t-target",
        str(ee_t_target),
    ]
    assert main(arguments) == 0
    np.save(ee_t_target, np.diag([1.0, 1.0, 1.0, 2.0]))
    assert main(arguments) == 2


def test_verify_validation_checks_complete_artifact_set(tmp_path: Path) -> None:
    parameters = tmp_path / "parameters.npz"
    dataset = tmp_path / "validation.npz"
    parameters.write_bytes(b"parameters")
    dataset.write_bytes(b"validation")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = tmp_path / "solve.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "parameters_npz_sha256": digest(parameters),
            }
        )
    )
    report = tmp_path / "validation.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_type": "kinematic-independent-validation",
                "accepted": True,
                "parameter_acceptance_verified": True,
                "independent_validation": True,
                "implementation_matches_parameters": True,
                "current_implementation_sha256": _implementation_sha256(),
                "parameters_npz_sha256": digest(parameters),
                "validation_dataset_sha256": digest(dataset),
                "calibration_result_sha256": digest(result),
                "quality_failures": [],
            }
        )
    )
    arguments = [
        "verify-validation",
        "--report",
        str(report),
        "--parameters",
        str(parameters),
        "--dataset",
        str(dataset),
        "--result",
        str(result),
    ]
    assert main(arguments) == 0
    dataset.write_bytes(b"tampered")
    assert main(arguments) == 2


def test_validate_result_rejects_training_dataset_by_default(tmp_path: Path) -> None:
    dataset = tmp_path / "poses.npz"
    KinematicDataset(
        joint_positions=np.array([[0.0], [0.0]]),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=ObservationFrame.CAMERA_T_TARGET,
    ).save(dataset)
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    parameters = tmp_path / "parameters.npz"
    np.savez_compressed(
        parameters,
        screw_axes=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        zero_configuration_pose=np.eye(4),
        model_mode=np.asarray("self-calibrating"),
        source_dataset_sha256=np.asarray(dataset_hash),
    )

    arguments = [
        "validate-result",
        "--parameters",
        str(parameters),
        "--dataset",
        str(dataset),
        "--allow-unverified-parameters",
    ]
    assert main(arguments) == 2
    report_path = tmp_path / "consistency.json"
    assert (
        main(
            [
                *arguments,
                "--allow-training-data",
                "--output",
                str(report_path),
            ]
        )
        == 1
    )
    report = json.loads(report_path.read_text())
    assert report["numerically_accepted"] is True
    assert report["accepted"] is False
    assert report["parameter_acceptance_verified"] is False
    assert report["independent_validation"] is False
    assert len(report["deployment_failures"]) == 3


def test_validate_result_rejects_partial_training_sample_overlap(
    tmp_path: Path,
) -> None:
    training = KinematicDataset(
        joint_positions=np.array([[0.0], [0.1]]),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=ObservationFrame.CAMERA_T_TARGET,
    )
    validation = KinematicDataset(
        joint_positions=np.array([[0.1], [0.2]]),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=ObservationFrame.CAMERA_T_TARGET,
    )
    training_path = tmp_path / "training.npz"
    validation_path = tmp_path / "validation.npz"
    training.save(training_path)
    validation.save(validation_path)
    parameters = tmp_path / "parameters.npz"
    np.savez_compressed(
        parameters,
        screw_axes=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        zero_configuration_pose=np.eye(4),
        model_mode=np.asarray("self-calibrating"),
        source_dataset_sha256=np.asarray(
            hashlib.sha256(training_path.read_bytes()).hexdigest()
        ),
        source_sample_sha256=_dataset_sample_sha256(training),
        source_sample_indices=training.sample_indices,
    )
    arguments = [
        "validate-result",
        "--parameters",
        str(parameters),
        "--dataset",
        str(validation_path),
        "--allow-unverified-parameters",
    ]
    assert main(arguments) == 2
    output = tmp_path / "overlap.json"
    assert main([*arguments, "--allow-training-data", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["independent_validation"] is False
    assert report["overlapping_training_sample_count"] == 1
    assert report["overlapping_training_samples"] == [
        {
            "sha256": _dataset_sample_sha256(training)[1],
            "training_positions": [1],
            "training_source_sample_indices": [1],
            "validation_position": 0,
            "validation_source_sample_index": 0,
        }
    ]


def test_validate_result_rejects_too_few_independent_samples(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "validation.npz"
    KinematicDataset(
        joint_positions=np.zeros((2, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=ObservationFrame.CAMERA_T_TARGET,
    ).save(dataset)
    implementation = _implementation_sha256()
    parameters = tmp_path / "parameters.npz"
    np.savez_compressed(
        parameters,
        schema_version=np.asarray(1),
        parameter_type=np.asarray("kinematic-poe"),
        screw_axes=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        zero_configuration_pose=np.eye(4),
        model_mode=np.asarray("self-calibrating"),
        source_dataset_sha256=np.asarray("a" * 64),
        source_sample_sha256=np.asarray(["b" * 64]),
        source_sample_indices=np.asarray([0]),
        implementation_sha256=np.asarray(implementation),
    )
    result = parameters.with_suffix(".json")
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_type": "kinematic-geometry",
                "accepted": True,
                "model_mode": "self-calibrating",
                "dataset_sha256": "a" * 64,
                "parameters_npz_sha256": hashlib.sha256(
                    parameters.read_bytes()
                ).hexdigest(),
                "provenance": {"implementation_sha256": implementation},
            }
        )
    )
    output = tmp_path / "validation.json"

    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["independent_validation"] is True
    assert report["numerically_accepted"] is False
    assert report["accepted"] is False
    assert report["quality_thresholds"]["min_validation_samples"] == 5
    assert any("at least 5" in item for item in report["numerical_quality_failures"])
    assert len(report["validation_coverage_failures"]) == 4


def test_validate_result_rejects_unknown_parameter_schema(tmp_path: Path) -> None:
    parameters = tmp_path / "parameters.npz"
    np.savez_compressed(
        parameters,
        schema_version=np.asarray(999),
        screw_axes=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        zero_configuration_pose=np.eye(4),
        model_mode=np.asarray("self-calibrating"),
    )


@pytest.mark.parametrize(
    ("accepted", "parameters_hash"),
    [(False, "unused"), (True, "0" * 64)],
)
def test_validate_result_rejects_unaccepted_or_mismatched_companion(
    tmp_path: Path, accepted: bool, parameters_hash: str
) -> None:
    parameters = tmp_path / "parameters.npz"
    np.savez_compressed(parameters, placeholder=np.asarray(1))
    companion = parameters.with_suffix(".json")
    companion.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "parameters_npz_sha256": parameters_hash,
            }
        )
    )

    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(tmp_path / "unused.npz"),
            ]
        )
        == 2
    )

    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(tmp_path / "unused.npz"),
                "--allow-unverified-parameters",
            ]
        )
        == 2
    )


def test_verify_collection_checks_candidate_partition(tmp_path: Path) -> None:
    dataset_path = tmp_path / "poses.npz"
    dataset = KinematicDataset(
        joint_positions=np.zeros((3, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 3, axis=0),
        observation_frame=ObservationFrame.CAMERA_T_TARGET,
        sample_indices=np.array([0, 2, 3]),
    )
    dataset.save(dataset_path)
    report_path = tmp_path / "poses.collection.json"
    report = {
        "schema_version": 1,
        "report_type": "ptpr-kinematic-collection",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "candidate_count": 4,
        "translation_candidate_count": 2,
        "rotation_candidate_count": 2,
        "successful_detection_count": 3,
        "failed_detection_count": 1,
        "detection_success_rate": 0.75,
        "successful_candidate_indices": [0, 2, 3],
        "failed_candidate_indices": [1],
        "dataset_coverage": dataset.get_summary(),
    }
    report_path.write_text(json.dumps(report))
    arguments = [
        "verify-collection",
        "--dataset",
        str(dataset_path),
        "--report",
        str(report_path),
    ]

    assert main(arguments) == 0
    report["failed_candidate_indices"] = [2]
    report_path.write_text(json.dumps(report))
    assert main(arguments) == 2


def test_compare_zero_offsets_reports_between_run_statistics(tmp_path: Path) -> None:
    result_paths = []
    for run, offset in enumerate((0.01, 0.012)):
        path = tmp_path / f"offsets_{run}.json"
        path.write_text(
            json.dumps(
                {
                    "calibration_type": "joint-zero-offset-only",
                    "accepted": True,
                    "model_mode": "robot-body",
                    "ee_name": "tool0",
                    "dof": 2,
                    "urdf_sha256": "a" * 64,
                    "dataset_sha256": str(run) * 64,
                    "offset_convention": "q_effective = q_measured + zero_offset_rad",
                    "estimated_joint_indices": [0, 1],
                    "gauge_fixed_joint_indices": [],
                    "joints": [
                        {
                            "zero_based_index": 0,
                            "joint_name": "Joint1",
                            "zero_offset_rad": offset,
                        },
                        {
                            "zero_based_index": 1,
                            "joint_name": "Joint2",
                            "zero_offset_rad": -0.02,
                        },
                    ],
                }
            )
        )
        result_paths.append(path)
    output = tmp_path / "repeatability.json"

    assert (
        main(
            [
                "compare-zero-offsets",
                "--result",
                str(result_paths[0]),
                "--result",
                str(result_paths[1]),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    assert report["accepted"] is True
    assert report["joints"][0]["mean_offset_rad"] == pytest.approx(0.011)
    assert report["joints"][0]["between_run_std_rad"] == pytest.approx(np.sqrt(2e-6))
