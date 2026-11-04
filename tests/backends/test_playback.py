import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.artifacts import load_experiment_bundle
from kinematic_calibration.backends import playback as playback_backend
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.backends.playback import ExperimentPlayback
from kinematic_calibration.experiment import (
    example_urdf,
    pose_metrics,
    predict_poe,
    run_experiment,
)


@pytest.fixture(scope="module")
def playback(tmp_path_factory):
    path = tmp_path_factory.mktemp("playback")
    run_experiment(path, samples=25, validation_samples=12)
    return ExperimentPlayback(path)


def test_replay_matches_verified_predictions_and_observation_identity(playback):
    assert playback.split == "validation"
    state = playback.sample(3)
    for name in ("nominal", "truth", "calibrated"):
        np.testing.assert_allclose(
            state.poses[name], playback.model[f"{name}_predictions"][3], atol=1e-12
        )
    np.testing.assert_array_equal(
        state.measured_pose, playback.dataset.observed_poses[3]
    )
    assert state.sample_index == playback.dataset.sample_indices[3]
    assert state.error_reference == "held-out measurement"


def test_training_replays_archived_rows_and_recomputes_its_metrics(
    playback, monkeypatch
):
    report, validation, model = load_experiment_bundle(playback.path)
    before_report = json.dumps(report, sort_keys=True)
    before_predictions = {
        name: model[f"{name}_predictions"].copy()
        for name in ("nominal", "truth", "calibrated")
    }
    monkeypatch.setattr(
        playback_backend,
        "load_experiment_bundle",
        lambda path: (report, validation, model),
    )
    training = ExperimentPlayback(playback.path, split="training")
    archived = KinematicDataset.load(playback.path / "train.npz")
    assert training.split == training.report["data_role"] == "training"
    assert training.model is not model
    assert training.report is not report
    np.testing.assert_array_equal(
        training.dataset.joint_positions, archived.joint_positions
    )
    np.testing.assert_array_equal(
        training.dataset.sample_indices, archived.sample_indices
    )
    assert len(training.dataset.joint_positions) != len(validation.joint_positions)
    for name, axes, home in (
        ("nominal", model["nominal_axes"], model["nominal_home"]),
        ("truth", model["truth_axes"], model["truth_home"]),
        ("calibrated", model["screw_axes"], model["home"]),
    ):
        expected = predict_poe(axes, home, archived.joint_positions)
        np.testing.assert_allclose(
            training.model[f"{name}_predictions"], expected, atol=1e-12
        )
        np.testing.assert_array_equal(
            model[f"{name}_predictions"], before_predictions[name]
        )
        for index in (0, 3, len(archived.joint_positions) - 1):
            state = training.sample(index)
            assert state.sample_index == archived.sample_indices[index]
            assert state.error_reference == "training measurement"
            np.testing.assert_array_equal(
                state.joint_positions, archived.joint_positions[index]
            )
            np.testing.assert_array_equal(
                state.measured_pose, archived.observed_poses[index]
            )
            np.testing.assert_allclose(state.poses[name], expected[index], atol=1e-12)
    for name, metric in (
        ("nominal", "nominal_vs_measurements"),
        ("calibrated", "calibrated_vs_measurements"),
    ):
        expected = pose_metrics(
            training.model[f"{name}_predictions"], archived.observed_poses
        )
        assert training.report[metric] == pytest.approx(expected)
    state = training.sample(3)
    assert state.position_error_m == pytest.approx(
        np.linalg.norm(
            state.poses["calibrated"][:3, 3] - archived.observed_poses[3, :3, 3]
        )
    )
    manual = training.manual(archived.joint_positions[3])
    assert manual.error_reference == "synthetic truth"
    assert manual.measured_pose is None
    assert json.dumps(report, sort_keys=True) == before_report


@pytest.mark.parametrize("split", [None, "train", "held-out", "", 1, [], True])
def test_split_is_rejected_before_loading_artifacts(split, monkeypatch):
    def unexpected_load(path):
        pytest.fail("invalid split must be rejected before loading artifacts")

    monkeypatch.setattr(playback_backend, "load_experiment_bundle", unexpected_load)
    with pytest.raises(ValueError, match="split must be"):
        ExperimentPlayback("missing", split=split)


@pytest.mark.parametrize("split", ["validation", "training"])
@pytest.mark.parametrize("artifact", ["train.npz", "validation.npz"])
def test_each_playback_split_verifies_both_datasets(
    playback, tmp_path, split, artifact
):
    destination = tmp_path / "changed"
    shutil.copytree(playback.path, destination)
    archive = destination / artifact
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ExperimentPlayback(destination, split=split)


def test_manual_joint_exploration_does_not_fabricate_sensor_observations(playback):
    q = playback.dataset.joint_positions[3].copy()
    before = playback.dataset.observed_poses.copy()
    state = playback.manual(q)
    replay = playback.sample(3)
    assert state.measured_pose is None
    assert state.sample_index is None
    assert state.error_reference == "synthetic truth"
    np.testing.assert_array_equal(state.poses["calibrated"], replay.poses["calibrated"])
    expected_error = np.linalg.norm(
        state.poses["calibrated"][:3, 3] - state.poses["truth"][:3, 3]
    )
    assert state.position_error_m == pytest.approx(expected_error)
    np.testing.assert_array_equal(playback.dataset.observed_poses, before)


def test_manual_joints_reject_invalid_values_and_limits(playback):
    for q in (
        np.zeros(5),
        np.full(6, np.nan),
        np.zeros(6, complex),
        np.zeros(6, bool),
        np.full(6, 100),
    ):
        with pytest.raises(ValueError):
            playback.manual(q)
    for index in (-1, True, 0.5, len(playback.dataset.joint_positions)):
        with pytest.raises(ValueError):
            playback.sample(index)


def test_backend_rejects_silent_complex_or_boolean_coercion():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    for q in (np.zeros(6, complex), np.zeros(6, bool), ["0"] * 6):
        with pytest.raises(ValueError):
            robot.forward(q)
    invalid = replace(robot.joints[0], origin=np.eye(4).astype(complex))
    with pytest.raises(ValueError):
        NumpyBackend((invalid, *robot.joints[1:]), robot.base_link, robot.end_link)


def test_unknown_bundle_type_is_explicitly_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"artifact_type": "unknown"}))
    with pytest.raises(ValueError, match="unsupported experiment"):
        load_experiment_bundle(tmp_path)
