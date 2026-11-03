"""Replay the same unnormalized parameters accepted by saved-model readers."""

import hashlib
import json
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.cli import main
from kinematic_calib.dataset import KinematicDataset
from kinematic_calib.validation import predict_saved_poe
from kinematic_calibration.backends.measured import MeasuredPlayback
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.backends.playback import ExperimentPlayback
from kinematic_calibration.experiment import example_urdf, pose_metrics, run_experiment
from kinematic_calibration.simulation import MeasurementNoise

ROUND_OFF_CASES = [
    "baseline",
    "core-roundoff",
    "angular",
    "pitch",
    "rotation",
    "home-row",
]
INVALID_CASES = [
    ("angular", "unit angular"),
    ("pitch", "zero pitch"),
    ("rotation", "not orthonormal"),
    ("home-row", "homogeneous row"),
]


def _perturb(axes, home, case, *, accepted=True):
    if case in {"angular", "core-roundoff"}:
        scale = 5e-9 if case == "core-roundoff" else 5e-6 if accepted else 2e-5
        axes[0] *= 1 + scale
    if case in {"pitch", "core-roundoff"}:
        pitch = 5e-9 if case == "core-roundoff" else 5e-7 if accepted else 2e-6
        angular = axes[0, 3:]
        axes[0, :3] += pitch * angular / np.dot(angular, angular)
    if case == "rotation":
        home[:3, :3] *= 1 + (2e-7 if accepted else 1e-5)
    if case == "home-row":
        error = 5e-9 if accepted else 2e-8
        home[3] = [error, -error, error, 1 + error]


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hashes(path):
    return {file.name: _sha256(file) for file in path.iterdir() if file.is_file()}


@pytest.fixture(scope="module", params=["base", "camera-body", "self-calibrating"])
def measured_fit(request, tmp_path_factory):
    mode = request.param
    path = tmp_path_factory.mktemp(f"parameter-tolerance-{mode}")
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=83, translation_std_m=0.001, rotation_std_rad=0.002)
    configurations = np.random.default_rng(82).uniform(-1.5, 1.5, (40, nominal.dof))
    observed = np.asarray([truth.forward(row) for row in configurations])
    base_t_camera, ee_t_target = np.eye(4), np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix()
    base_t_camera[:3, 3] = [0.4, -0.2, 0.7]
    ee_t_target[:3, :3] = Rotation.from_rotvec([-0.1, 0.05, 0.12]).as_matrix()
    ee_t_target[:3, 3] = [0.03, -0.02, 0.1]
    if mode != "base":
        observed = np.linalg.inv(base_t_camera) @ observed @ ee_t_target
    frame = "base_T_ee" if mode == "base" else "camera_T_target"
    for name, rows, indices in (
        ("train", slice(0, 32), np.arange(100, 132)),
        ("validation", slice(32, 40), np.arange(1000, 1008)),
    ):
        KinematicDataset(
            configurations[rows], observed[rows], frame, indices, nominal.joint_names
        ).save(path / f"{name}.npz")
    np.save(path / "base_T_camera.npy", base_t_camera)
    np.save(path / "ee_T_target.npy", ee_t_target)
    arguments = [
        "solve",
        "--backend",
        "numpy",
        "--dataset",
        str(path / "train.npz"),
        "--urdf",
        str(example_urdf()),
        "--ee-name",
        "ee_link",
        "--model-mode",
        "self-calibrating" if mode == "self-calibrating" else "robot-body",
        "--output",
        str(path / "fit.npz"),
        "--quiet",
    ]
    if mode == "camera-body":
        arguments.extend(
            [
                "--base-t-camera",
                str(path / "base_T_camera.npy"),
                "--ee-t-target",
                str(path / "ee_T_target.npy"),
            ]
        )
    assert main(arguments) == 0
    assert json.loads((path / "fit.json").read_text())["accepted"] is True
    return path, mode


def _edited_measured_fit(source, destination, case, *, accepted=True):
    shutil.copytree(source, destination)
    arrays = _arrays(destination / "fit.npz")
    axes, home = arrays["screw_axes"], arrays["zero_configuration_pose"]
    _perturb(axes, home, case, accepted=accepted)
    np.savez_compressed(destination / "fit.npz", **arrays)
    report_path = destination / "fit.json"
    report = json.loads(report_path.read_text())
    report["parameters_npz_sha256"] = _sha256(destination / "fit.npz")
    report_path.write_text(json.dumps(report))
    return axes, home


def _measured_playback(path, mode):
    options = {}
    if mode == "camera-body":
        options = {
            "base_t_camera": path / "base_T_camera.npy",
            "ee_t_target": path / "ee_T_target.npy",
        }
    return MeasuredPlayback(
        path / "fit.npz", path / "validation.npz", example_urdf(), **options
    )


@pytest.mark.parametrize("case", ROUND_OFF_CASES)
def test_measured_playback_preserves_accepted_saved_parameters(
    measured_fit, tmp_path, case
):
    source, mode = measured_fit
    path = tmp_path / "fit"
    axes, home = _edited_measured_fit(source, path, case)
    before_files = _file_hashes(path)
    playback = _measured_playback(path, mode)
    joints = playback.dataset.joint_positions.copy()
    observations = playback.dataset.observed_poses.copy()
    expected = predict_saved_poe(axes, home, joints)
    np.testing.assert_allclose(
        playback.model["calibrated_predictions"], expected, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        playback.model["calibrated_predictions"][:, 3],
        np.broadcast_to(home[3], (len(joints), 4)),
        atol=2e-15,
        rtol=0,
    )
    assert playback.report["coordinate_frame"] == (
        "camera" if mode == "self-calibrating" else "base"
    )
    if mode == "self-calibrating":
        assert "initialization" in playback.report["display_note"]
        assert "not calibrated physical extrinsics" in playback.report["display_note"]
    state = playback.sample(3)
    np.testing.assert_allclose(
        state.poses["calibrated"], expected[3], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        state.poses["calibrated"][3], home[3], atol=2e-15, rtol=0
    )
    np.testing.assert_array_equal(state.measured_pose, observations[3])
    assert state.sample_index == 1003
    assert (
        state.error_reference == playback.report["data_role"] == "held-out measurements"
    )
    configuration = joints[0] + 0.05
    original_configuration = configuration.copy()
    configuration.setflags(write=False)
    manual = playback.manual(configuration)
    np.testing.assert_allclose(
        manual.poses["calibrated"],
        predict_saved_poe(axes, home, configuration[None])[0],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        manual.poses["calibrated"][3], home[3], atol=2e-15, rtol=0
    )
    assert manual.measured_pose is None and manual.sample_index is None
    assert manual.position_error_m is None and manual.orientation_error_rad is None
    assert manual.error_reference == "no measurement"
    np.testing.assert_array_equal(configuration, original_configuration)
    np.testing.assert_array_equal(playback.dataset.joint_positions, joints)
    np.testing.assert_array_equal(playback.dataset.observed_poses, observations)
    # At zero joints the motion product is identity, so the raw saved home
    # itself is the analytic result, including its accepted roundoff.
    np.testing.assert_array_equal(
        playback.manual(np.zeros_like(joints[0])).poses["calibrated"], home
    )
    assert _file_hashes(path) == before_files


@pytest.mark.parametrize("case,message", INVALID_CASES)
def test_measured_playback_still_rejects_parameters_outside_saved_tolerance(
    measured_fit, tmp_path, case, message
):
    source, mode = measured_fit
    path = tmp_path / "fit"
    _edited_measured_fit(source, path, case, accepted=False)
    with pytest.raises(ValueError, match=message):
        _measured_playback(path, mode)


@pytest.fixture(scope="module")
def experiment_fit(tmp_path_factory):
    path = tmp_path_factory.mktemp("experiment-parameter-tolerance")
    report = run_experiment(
        path, samples=25, validation_samples=8, seed=17, noise=MeasurementNoise(0, 0)
    )
    assert report["accepted"] is True
    return path


def _edited_experiment(source, destination, case, *, accepted=True):
    shutil.copytree(source, destination)
    model = _arrays(destination / "model.npz")
    axes, home = model["screw_axes"], model["home"]
    _perturb(axes, home, case, accepted=accepted)
    if accepted:
        validation = KinematicDataset.load(destination / "validation.npz")
        predicted = predict_saved_poe(axes, home, validation.joint_positions)
        model["calibrated_predictions"] = predicted
        report_path = destination / "metrics.json"
        report = json.loads(report_path.read_text())
        report["calibrated_vs_measurements"] = pose_metrics(
            predicted, validation.observed_poses
        )
        report["calibrated_vs_truth"] = pose_metrics(
            predicted, model["truth_predictions"]
        )
        report_path.write_text(json.dumps(report))
    np.savez_compressed(destination / "model.npz", **model)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("model.npz", "metrics.json"):
        manifest["files_sha256"][name] = _sha256(destination / name)
    manifest_path.write_text(json.dumps(manifest))
    return axes, home


@pytest.mark.parametrize("case", ROUND_OFF_CASES)
@pytest.mark.parametrize("split", ["validation", "training"])
def test_experiment_playback_preserves_accepted_saved_parameters(
    experiment_fit, tmp_path, case, split
):
    path = tmp_path / "experiment"
    axes, home = _edited_experiment(experiment_fit, path, case)
    before_files = _file_hashes(path)
    playback = ExperimentPlayback(path, split=split)
    joints = playback.dataset.joint_positions.copy()
    observations = playback.dataset.observed_poses.copy()
    expected = predict_saved_poe(axes, home, joints)
    np.testing.assert_allclose(
        playback.model["calibrated_predictions"], expected, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(playback.model["screw_axes"], axes)
    np.testing.assert_array_equal(playback.model["home"], home)
    np.testing.assert_allclose(
        playback.model["calibrated_predictions"][:, 3],
        np.broadcast_to(home[3], (len(joints), 4)),
        atol=2e-15,
        rtol=0,
    )
    state = playback.sample(3)
    np.testing.assert_allclose(
        state.poses["calibrated"], expected[3], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        state.poses["calibrated"][3], home[3], atol=2e-15, rtol=0
    )
    np.testing.assert_array_equal(state.measured_pose, observations[3])
    assert state.sample_index == playback.dataset.sample_indices[3]
    assert state.error_reference == (
        "training measurement" if split == "training" else "held-out measurement"
    )
    if split == "training":
        assert playback.report["data_role"] == "training"
    configuration = joints[0].copy()
    configuration.setflags(write=False)
    manual = playback.manual(configuration)
    np.testing.assert_allclose(
        manual.poses["calibrated"], expected[0], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        manual.poses["calibrated"][3], home[3], atol=2e-15, rtol=0
    )
    assert manual.measured_pose is None and manual.sample_index is None
    assert manual.error_reference == "synthetic truth"
    np.testing.assert_array_equal(configuration, joints[0])
    np.testing.assert_array_equal(playback.dataset.joint_positions, joints)
    np.testing.assert_array_equal(playback.dataset.observed_poses, observations)
    # At zero joints the motion product is identity, so the raw saved home
    # itself is the analytic result, including its accepted roundoff.
    np.testing.assert_array_equal(
        playback.manual(np.zeros_like(joints[0])).poses["calibrated"], home
    )
    assert _file_hashes(path) == before_files
    # Display artifacts remain mutable without changing the retained fitted model.
    playback.model["screw_axes"][:] = 0
    playback.model["home"][:] = 0
    np.testing.assert_allclose(
        playback.manual(configuration).poses["calibrated"],
        expected[0],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("case,message", INVALID_CASES)
@pytest.mark.parametrize("split", ["validation", "training"])
def test_experiment_playback_still_rejects_parameters_outside_saved_tolerance(
    experiment_fit, tmp_path, case, message, split
):
    path = tmp_path / "experiment"
    _edited_experiment(experiment_fit, path, case, accepted=False)
    with pytest.raises(ValueError, match=message):
        ExperimentPlayback(path, split=split)
