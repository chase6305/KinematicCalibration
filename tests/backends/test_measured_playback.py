"""Replay real solve artifacts without fabricating truth or manual observations."""

import hashlib
import json
import shutil
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset, ObservationFrame
from kinematic_calibration.backends.measured import MeasuredPlayback
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf


@pytest.fixture(scope="module")
def fitted_cases(tmp_path_factory):
    from kinematic_calib.cli import main

    root = tmp_path_factory.mktemp("measured-playback")
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=83, translation_std_m=0.001, rotation_std_rad=0.002)
    configurations = np.random.default_rng(82).uniform(-1.5, 1.5, (40, nominal.dof))
    base_poses = np.asarray([truth.forward(row) for row in configurations])
    base_t_camera = np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix()
    base_t_camera[:3, 3] = [0.4, -0.2, 0.7]
    ee_t_target = np.eye(4)
    ee_t_target[:3, :3] = Rotation.from_rotvec([-0.1, 0.05, 0.12]).as_matrix()
    ee_t_target[:3, 3] = [0.03, -0.02, 0.1]
    cases = {}
    for mode in ("base", "camera-body", "self-calibrating"):
        path = root / mode
        path.mkdir()
        camera_data = mode != "base"
        observed = (
            np.linalg.inv(base_t_camera) @ base_poses @ ee_t_target
            if camera_data
            else base_poses
        )
        frame = (
            ObservationFrame.CAMERA_T_TARGET
            if camera_data
            else ObservationFrame.BASE_T_EE
        )
        for name, rows, ids in (
            ("train", slice(0, 32), np.arange(100, 132)),
            ("validation", slice(32, 40), np.arange(1000, 1008)),
        ):
            KinematicDataset(
                configurations[rows], observed[rows], frame, ids, nominal.joint_names
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
        with patch.dict("sys.modules", {"roboticstoolbox": None}):
            assert main(arguments) == 0
        cases[mode] = (path, base_poses[32:], observed[32:])
    return cases


def replay(case, *, dataset="validation.npz", **options):
    path = case[0]
    if (
        path.name == "camera-body"
        and "base_t_camera" not in options
        and "ee_t_target" not in options
    ):
        options.update(
            base_t_camera=path / "base_T_camera.npy",
            ee_t_target=path / "ee_T_target.npy",
        )
    with patch.dict("sys.modules", {"roboticstoolbox": None}):
        return MeasuredPlayback(
            path / "fit.npz", path / dataset, example_urdf(), **options
        )


@pytest.mark.parametrize("mode", ["base", "camera-body", "self-calibrating"])
def test_replay_matches_independent_frame_relationships_and_has_no_truth(
    fitted_cases, mode
):
    case = fitted_cases[mode]
    playback = replay(case)
    state = playback.sample(2)
    expected_measured = case[2][2] if mode == "self-calibrating" else case[1][2]
    np.testing.assert_allclose(state.measured_pose, expected_measured, atol=2e-15)
    np.testing.assert_allclose(state.poses["calibrated"], expected_measured, atol=2e-7)
    assert set(state.links) == {"nominal"}
    assert set(state.poses) == {"nominal", "calibrated"}
    assert not any("truth" in key for key in playback.model)
    assert state.sample_index == 1002
    assert (
        state.error_reference == playback.report["data_role"] == "held-out measurements"
    )
    assert state.position_error_m < 2e-7
    assert playback.report["calibrated_vs_measurements"]["position_rmse_m"] < 2e-7
    assert playback.report["replay_end_link_identity_verified"] is True
    assert playback.report["coordinate_frame"] == (
        "camera" if mode == "self-calibrating" else "base"
    )


def test_self_calibrating_nominal_skeleton_is_explicitly_an_initialization_gauge(
    fitted_cases,
):
    from kinematic_calib.saved_model import load_fitted_model

    path = fitted_cases["self-calibrating"][0]
    fitted = load_fitted_model(path / "fit.npz", example_urdf())
    playback = replay(fitted_cases["self-calibrating"])
    state = playback.sample(0)
    np.testing.assert_array_equal(
        state.links["nominal"][playback.nominal.base_link],
        fitted.camera_t_base_initialization,
    )
    np.testing.assert_allclose(
        state.poses["nominal"],
        fitted.camera_t_base_initialization
        @ playback.nominal.forward(state.joint_positions)
        @ fitted.ee_t_target_initialization,
        atol=1e-14,
    )
    assert "initialization" in playback.report["display_note"]
    assert "not calibrated physical extrinsics" in playback.report["display_note"]


@pytest.mark.parametrize("mode", ["base", "camera-body", "self-calibrating"])
def test_manual_configuration_has_no_observation_or_residual_and_does_not_modify_inputs(
    fitted_cases, mode
):
    path = fitted_cases[mode][0]
    files = (
        list(path.glob("*.npz")) + list(path.glob("*.npy")) + list(path.glob("*.json"))
    )
    hashes = {name: hashlib.sha256(name.read_bytes()).hexdigest() for name in files}
    playback = replay(fitted_cases[mode])
    original_observed = playback.dataset.observed_poses.copy()
    configuration = playback.dataset.joint_positions[0].copy()
    state = playback.manual(configuration)
    assert state.measured_pose is None and state.sample_index is None
    assert state.position_error_m is None and state.orientation_error_rad is None
    assert state.error_reference == "no measurement"
    np.testing.assert_array_equal(configuration, playback.dataset.joint_positions[0])
    measured = playback.sample(0)
    measured.measured_pose[:] = 0
    state.joint_positions[:] = 0
    state.poses["calibrated"][:] = 0
    np.testing.assert_array_equal(playback.dataset.observed_poses, original_observed)
    np.testing.assert_array_equal(playback.dataset.joint_positions[0], configuration)
    assert hashes == {
        name: hashlib.sha256(name.read_bytes()).hexdigest() for name in files
    }


def test_training_identity_survives_renames_reencoding_and_changed_sample_ids(
    fitted_cases, tmp_path
):
    case = fitted_cases["base"]
    path = case[0]
    assert (
        replay(case, dataset="train.npz").report["data_role"] == "training measurements"
    )
    renamed = tmp_path / "renamed.npz"
    shutil.copyfile(path / "train.npz", renamed)
    assert replay(case, dataset=renamed).report["data_role"] == "training measurements"
    training = KinematicDataset.load(path / "train.npz")
    changed_ids = tmp_path / "new-ids.npz"
    replace(training, sample_indices=training.sample_indices + 10000).save(changed_ids)
    role = replay(case, dataset=changed_ids).report
    assert role["data_role"] == "measurements with training overlap"
    assert role["training_sample_overlap_count"] == len(training.joint_positions)
    assert role["training_id_overlap_count"] == 0


def test_changed_observations_with_reused_ids_are_not_labeled_held_out(
    fitted_cases, tmp_path
):
    case = fitted_cases["base"]
    training = KinematicDataset.load(case[0] / "train.npz")
    changed = training.observed_poses.copy()
    changed[:, 0, 3] += 0.01
    destination = tmp_path / "remeasured.npz"
    replace(training, observed_poses=changed).save(destination)
    report = replay(case, dataset=destination).report
    assert report["data_role"] == "measurements with training overlap"
    assert report["training_sample_overlap_count"] == 0
    assert report["training_id_overlap_count"] == len(training.joint_positions)


def test_overlap_count_includes_each_reused_measurement_row(fitted_cases, tmp_path):
    case = fitted_cases["base"]
    training = KinematicDataset.load(case[0] / "train.npz")
    repeated = KinematicDataset(
        np.repeat(training.joint_positions[:1], 3, axis=0),
        np.repeat(training.observed_poses[:1], 3, axis=0),
        training.observation_frame,
        np.arange(10000, 10003),
        training.joint_names,
    )
    destination = tmp_path / "repeated.npz"
    repeated.save(destination)
    report = replay(case, dataset=destination).report
    assert report["data_role"] == "measurements with training overlap"
    assert report["training_sample_overlap_count"] == 3
    assert report["training_id_overlap_count"] == 0


@pytest.mark.parametrize("field", ["source_sample_indices", "source_sample_sha256"])
def test_available_training_dataset_must_match_the_saved_sample_table(
    fitted_cases, tmp_path, field
):
    path = fitted_cases["base"][0]
    with np.load(path / "fit.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if field == "source_sample_indices":
        arrays[field] = arrays[field] + 10000
    else:
        arrays[field] = np.full(arrays[field].shape, "a" * 64)
    parameters = tmp_path / "changed-fit.npz"
    np.savez_compressed(parameters, **arrays)
    report = json.loads((path / "fit.json").read_text())
    report["parameters_npz_sha256"] = hashlib.sha256(
        parameters.read_bytes()
    ).hexdigest()
    parameters.with_suffix(".json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training dataset.*source sample table"):
        MeasuredPlayback(parameters, path / "train.npz", example_urdf())


def test_rejected_fit_remains_available_for_diagnostic_replay(fitted_cases, tmp_path):
    path = fitted_cases["base"][0]
    report = json.loads((path / "fit.json").read_text())
    report["accepted"] = False
    report["quality_failures"] = ["diagnostic example"]
    result = tmp_path / "rejected-fit.json"
    result.write_text(json.dumps(report))
    playback = MeasuredPlayback(
        path / "fit.npz", path / "validation.npz", example_urdf(), result=result
    )
    assert playback.report["accepted"] is False
    assert playback.sample(0).measured_pose is not None


def test_already_base_observations_are_not_double_converted_or_required_to_load_camera_files(
    fitted_cases, tmp_path
):
    case = fitted_cases["camera-body"]
    path = case[0]
    source = KinematicDataset.load(path / "train.npz")
    converted = source.as_robot_body(
        np.load(path / "base_T_camera.npy"), np.load(path / "ee_T_target.npy")
    )
    destination = tmp_path / "base-data.npz"
    replace(converted, sample_indices=converted.sample_indices + 10000).save(
        destination
    )
    playback = MeasuredPlayback(path / "fit.npz", destination, example_urdf())
    np.testing.assert_array_equal(
        playback.dataset.observed_poses, converted.observed_poses
    )
    assert playback.report["coordinate_frame"] == "base"
    assert playback.report["data_role"] == "measurements (independence unverified)"


def test_camera_body_replay_requires_both_bound_fixed_transform_files(
    fitted_cases, tmp_path
):
    path = fitted_cases["camera-body"][0]
    arguments = (path / "fit.npz", path / "validation.npz", example_urdf())
    with pytest.raises(ValueError, match="both fixed"):
        MeasuredPlayback(*arguments)
    with pytest.raises(ValueError, match="both fixed"):
        MeasuredPlayback(*arguments, base_t_camera=path / "base_T_camera.npy")
    modified = np.load(path / "base_T_camera.npy")
    modified[0, 3] += 0.01
    wrong = tmp_path / "wrong-camera.npy"
    np.save(wrong, modified)
    with pytest.raises(ValueError, match="does not match the fitted pair"):
        MeasuredPlayback(
            *arguments, base_t_camera=wrong, ee_t_target=path / "ee_T_target.npy"
        )


def test_dataset_identity_and_nominal_limit_violations_are_explicit(
    fitted_cases, tmp_path
):
    case = fitted_cases["base"]
    source = KinematicDataset.load(case[0] / "validation.npz")
    swapped = tmp_path / "swapped.npz"
    replace(source, joint_names=tuple(reversed(source.joint_names))).save(swapped)
    with pytest.raises(ValueError, match="joint_names"):
        replay(case, dataset=swapped)
    positions = source.joint_positions.copy()
    positions[0, 0] = 100
    outside = tmp_path / "outside.npz"
    replace(source, joint_positions=positions).save(outside)
    with pytest.raises(ValueError, match="sample 1000.*outside nominal"):
        replay(case, dataset=outside)


def test_manual_and_sample_indices_reject_invalid_values(fitted_cases):
    playback = replay(fitted_cases["base"])
    for positions in (
        np.zeros(5),
        np.zeros(6, bool),
        np.zeros(6, complex),
        np.full(6, np.nan),
        np.full(6, 100),
    ):
        with pytest.raises(ValueError):
            playback.manual(positions)
    for index in (-1, True, 0.5, len(playback.dataset.joint_positions)):
        with pytest.raises(ValueError):
            playback.sample(index)
