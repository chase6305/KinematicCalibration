import numpy as np
import pytest

from kinematic_calib.dataset import camera_target_to_base_ee
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import (
    example_urdf,
    load_experiment,
    run_experiment,
)
from kinematic_calibration.simulation import MeasurementNoise, sample_measurements


def test_noiseless_calibration_generalizes_to_unseen_joints(tmp_path):
    report = run_experiment(
        tmp_path,
        samples=25,
        validation_samples=12,
        noise=MeasurementNoise(0, 0),
        seed=8,
    )
    before = report["nominal_vs_measurements"]["position_rmse_m"]
    after = report["calibrated_vs_measurements"]["position_rmse_m"]
    assert after < 1e-6
    assert after < before / 1000
    assert report["regressor_rank"] == 30
    _, dataset, model = load_experiment(tmp_path)
    assert len(dataset.joint_positions) == 12
    assert model["screw_axes"].shape == (6, 6)
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path)


def test_camera_observations_recover_robot_body_poses():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    base_t_camera, ee_t_board = np.eye(4), np.eye(4)
    base_t_camera[:3, 3] = [1, 2, 3]
    ee_t_board[:3, 3] = [0.1, 0.2, -0.1]
    observed = sample_measurements(
        robot,
        count=8,
        noise=MeasurementNoise(0, 0),
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_board,
    )
    recovered = camera_target_to_base_ee(
        observed.observed_poses, base_t_camera, ee_t_board
    )
    expected = np.array([robot.forward(q) for q in observed.joint_positions])
    np.testing.assert_allclose(recovered, expected, atol=1e-14)


@pytest.mark.parametrize("position,orientation", [(np.nan, 0), (0, np.inf), (-1, 0)])
def test_noise_rejects_invalid_standard_deviations(position, orientation):
    with pytest.raises(ValueError):
        MeasurementNoise(position, orientation)


@pytest.fixture(scope="module")
def experiment_snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("verified-experiment")
    run_experiment(
        path, samples=20, validation_samples=8, seed=17, noise=MeasurementNoise(0, 0)
    )
    return path


def _copy_experiment(source, destination):
    import shutil

    shutil.copytree(source, destination)
    return destination


def _refresh_manifest(path):
    """Exercise semantic validation after deliberate, internally rehashed edits."""
    import hashlib
    import json

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = hashlib.sha256(
            (path / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_experiment_has_disjoint_ids_and_bound_source_identity(experiment_snapshot):
    import hashlib

    from kinematic_calib.dataset import KinematicDataset

    report, validation, _ = load_experiment(experiment_snapshot)
    train = KinematicDataset.load(experiment_snapshot / "train.npz")
    assert not set(train.sample_indices) & set(validation.sample_indices)
    assert (
        report["source_urdf_sha256"]
        == hashlib.sha256(
            (experiment_snapshot / "nominal.urdf").read_bytes()
        ).hexdigest()
    )
    assert report["parameter_count"] == 30
    assert report["full_rank"]
    assert report["condition_number"] >= 1
    assert report["accepted"]


@pytest.mark.parametrize(
    "artifact",
    ["train.npz", "validation.npz", "model.npz", "nominal.urdf", "metrics.json"],
)
def test_load_rejects_modified_artifact_hash(experiment_snapshot, tmp_path, artifact):
    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    with (path / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_experiment(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("screw_axes", np.zeros((5, 6)), "unit angular|DOF"),
        ("home", np.full((4, 4), np.nan), "finite"),
        (
            "calibrated_predictions",
            np.repeat(np.eye(4)[None], 7, axis=0),
            "sample count",
        ),
        ("joint_names", np.asarray(["duplicate"] * 6), "joint identities"),
        ("base_link", np.asarray("unrelated"), "base_link"),
        ("regressor_singular_values", np.full(30, np.nan), "singular values"),
    ],
)
def test_load_rejects_semantically_invalid_model(
    experiment_snapshot, tmp_path, field, value, message
):
    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    model[field] = value
    np.savez_compressed(path / "model.npz", **model)
    _refresh_manifest(path)
    with pytest.raises(ValueError, match=message):
        load_experiment(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("validation_samples", True, "validation_samples"),
        ("dof", 5, "joint dimensions"),
        ("regressor_rank", 29, "regressor diagnostics"),
        ("condition_number", 1.0, "regressor diagnostics"),
        ("converged", "true", "converged"),
        ("accepted", False, "acceptance diagnostics"),
        ("truth_seed", False, "truth_seed"),
    ],
)
def test_load_rejects_inconsistent_report(
    experiment_snapshot, tmp_path, field, value, message
):
    import json

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    report_path = path / "metrics.json"
    report = json.loads(report_path.read_text())
    report[field] = value
    report_path.write_text(json.dumps(report))
    _refresh_manifest(path)
    with pytest.raises(ValueError, match=message):
        load_experiment(path)


def test_load_checks_predictions_and_metrics_against_model(
    experiment_snapshot, tmp_path
):
    import json

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    report_path = path / "metrics.json"
    report = json.loads(report_path.read_text())
    report["calibrated_vs_measurements"]["position_rmse_m"] = 1.0
    report_path.write_text(json.dumps(report))
    _refresh_manifest(path)
    with pytest.raises(ValueError, match="does not match saved predictions"):
        load_experiment(path)


def test_load_rejects_reused_sample_ids(experiment_snapshot, tmp_path):
    from kinematic_calib.dataset import KinematicDataset

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    validation = KinematicDataset.load(path / "validation.npz")
    KinematicDataset(
        validation.joint_positions,
        validation.observed_poses,
        validation.observation_frame,
    ).save(path / "validation.npz")
    _refresh_manifest(path)
    with pytest.raises(ValueError, match="sample identities must be disjoint"):
        load_experiment(path)


@pytest.mark.parametrize(
    "arguments",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2.5},
        {"samples": False},
        {"samples": 4},
        {"validation_samples": 4},
        {"validation_samples": 8.5},
        {"max_iterations": 0},
        {"force": "yes"},
    ],
)
def test_invalid_controls_do_not_create_partial_experiment(tmp_path, arguments):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        run_experiment(output, **arguments)
    assert not output.exists()


@pytest.mark.parametrize("reserved", ["nominal.urdf", "metrics.json", "manifest.json"])
def test_force_rejects_reserved_directory_collision_before_fit(tmp_path, reserved):
    (tmp_path / reserved).mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        run_experiment(tmp_path, force=True)
    assert list(tmp_path.iterdir()) == [tmp_path / reserved]


def test_force_rejects_symlink_artifacts(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_bytes(example_urdf().read_bytes())
    (tmp_path / "nominal.urdf").symlink_to(source)
    with pytest.raises(ValueError, match="not a regular file"):
        run_experiment(tmp_path, force=True)
    assert (tmp_path / "nominal.urdf").is_symlink()


def test_source_model_inside_output_is_archived_without_overwriting_input(tmp_path):
    source = tmp_path / "nominal.urdf"
    source_bytes = example_urdf().read_bytes()
    source.write_bytes(source_bytes)
    run_experiment(tmp_path, urdf=source, samples=20, validation_samples=5, force=True)
    assert source.read_bytes() == source_bytes
    load_experiment(tmp_path)


def test_source_cannot_use_another_reserved_output_name(tmp_path):
    source = tmp_path / "model.npz"
    source_bytes = example_urdf().read_bytes()
    source.write_bytes(source_bytes)
    with pytest.raises(ValueError, match="source URDF conflicts"):
        run_experiment(tmp_path, urdf=source, force=True)
    assert source.read_bytes() == source_bytes


def test_publish_failure_restores_previous_experiment(
    experiment_snapshot, tmp_path, monkeypatch
):
    import kinematic_calibration.experiment as experiment

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    (path / "notes.txt").write_text("keep unrelated files")
    before = {entry.name: entry.read_bytes() for entry in path.iterdir()}
    original_replace = experiment.os.replace
    armed = True

    def fail_one_install(source, destination):
        nonlocal armed
        if armed and destination == path / "model.npz":
            armed = False
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(experiment.os, "replace", fail_one_install)
    with pytest.raises(OSError, match="injected publish failure"):
        run_experiment(path, samples=20, validation_samples=8, seed=19, force=True)
    assert {entry.name: entry.read_bytes() for entry in path.iterdir()} == before
    load_experiment(path)


def test_experiment_remains_loadable_after_directory_move(
    experiment_snapshot, tmp_path
):
    import shutil

    path = _copy_experiment(experiment_snapshot, tmp_path / "before")
    moved = tmp_path / "after"
    shutil.move(path, moved)
    report, dataset, model = load_experiment(moved)
    assert report["validation_samples"] == len(dataset.joint_positions)
    assert model["nominal_predictions"].shape == (len(dataset.joint_positions), 4, 4)


def test_nonconverged_fit_is_saved_as_rejected_diagnostics(tmp_path):
    report = run_experiment(
        tmp_path, samples=20, validation_samples=5, max_iterations=1
    )
    assert not report["accepted"]
    assert not report["converged"]
    assert "solver did not converge" in report["quality_failures"]
    loaded_report, _, _ = load_experiment(tmp_path)
    assert loaded_report["accepted"] is False


def test_load_rejects_endpoint_predictions_unrelated_to_model(
    experiment_snapshot, tmp_path
):
    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    model["calibrated_predictions"][:, 0, 3] += 0.1
    np.savez_compressed(path / "model.npz", **model)
    _refresh_manifest(path)
    with pytest.raises(ValueError, match="do not match saved POE"):
        load_experiment(path)


def test_load_rejects_joint_states_outside_model_limits(experiment_snapshot, tmp_path):
    from kinematic_calib.dataset import KinematicDataset

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    train = KinematicDataset.load(path / "train.npz")
    joints = train.joint_positions.copy()
    joints[0, 0] = 100
    KinematicDataset(
        joints,
        train.observed_poses,
        train.observation_frame,
        train.sample_indices,
        train.joint_names,
    ).save(path / "train.npz")
    _refresh_manifest(path)
    with pytest.raises(ValueError, match="joint states exceed"):
        load_experiment(path)


def test_publish_interruption_restores_previous_experiment(
    experiment_snapshot, tmp_path, monkeypatch
):
    import kinematic_calibration.experiment as experiment

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    before = {entry.name: entry.read_bytes() for entry in path.iterdir()}
    original_replace = experiment.os.replace
    armed = True

    def interrupt_one_install(source, destination):
        nonlocal armed
        if armed and destination == path / "model.npz":
            armed = False
            raise KeyboardInterrupt()
        return original_replace(source, destination)

    monkeypatch.setattr(experiment.os, "replace", interrupt_one_install)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(path, samples=20, validation_samples=8, seed=19, force=True)
    assert {entry.name: entry.read_bytes() for entry in path.iterdir()} == before
    load_experiment(path)


def test_new_simulation_and_experiment_datasets_have_exact_model_joint_names(
    experiment_snapshot,
):
    from kinematic_calib.dataset import KinematicDataset

    report, validation, model = load_experiment(experiment_snapshot)
    training = KinematicDataset.load(experiment_snapshot / "train.npz")
    assert training.joint_names == validation.joint_names == tuple(model["joint_names"])
    assert report["joint_identity_verified"]
    assert report["joint_identity"]["training"]["verified"]
    assert report["joint_identity"]["validation"]["verified"]


def test_previous_schema_two_experiment_with_unnamed_v1_data_loads_unverified(
    experiment_snapshot, tmp_path, caplog
):
    import json

    path = _copy_experiment(experiment_snapshot, tmp_path / "legacy-experiment")
    for filename in ("train.npz", "validation.npz"):
        with np.load(path / filename, allow_pickle=False) as archive:
            arrays = {
                key: archive[key] for key in archive.files if key != "joint_names"
            }
        arrays["schema_version"] = np.asarray(1)
        np.savez_compressed(path / filename, **arrays)
    report_path = path / "metrics.json"
    report = json.loads(report_path.read_text())
    report.pop("joint_identity_verified")
    report.pop("joint_identity")
    report_path.write_text(json.dumps(report))
    _refresh_manifest(path)
    original_report_bytes = report_path.read_bytes()
    report, validation, _ = load_experiment(path)
    assert not report["joint_identity_verified"]
    assert validation.joint_names is None
    assert "unverified" in caplog.text
    assert report_path.read_bytes() == original_report_bytes


def test_experiment_rejects_validation_joint_name_permutation(
    experiment_snapshot, tmp_path
):
    from kinematic_calib.dataset import KinematicDataset

    path = _copy_experiment(experiment_snapshot, tmp_path / "experiment")
    validation = KinematicDataset.load(path / "validation.npz")
    KinematicDataset(
        validation.joint_positions[:, ::-1],
        validation.observed_poses,
        validation.observation_frame,
        validation.sample_indices,
        validation.joint_names[::-1],
    ).save(path / "validation.npz")
    _refresh_manifest(path)
    with pytest.raises(ValueError, match="joint_names order"):
        load_experiment(path)
