import json
from dataclasses import replace

import numpy as np
import pytest

from kinematic_calibration.artifacts import load_experiment_bundle
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.backends.playback import ExperimentPlayback
from kinematic_calibration.experiment import example_urdf, run_experiment


@pytest.fixture(scope="module")
def playback(tmp_path_factory):
    path = tmp_path_factory.mktemp("playback")
    run_experiment(path, samples=25, validation_samples=12)
    return ExperimentPlayback(path)


def test_replay_matches_verified_predictions_and_observation_identity(playback):
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
