"""Real dataset-file workflow through the NumPy model loader, without RTB."""

import json
import sys

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from kinematic_calib import KinematicDataset, ObservationFrame
from kinematic_calib.cli import build_parser, main
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf


def _data(tmp_path, *, camera=False, base_link=None):
    nominal = NumpyBackend.from_urdf(
        example_urdf(), end_link="ee_link", base_link=base_link
    )
    truth = nominal.perturbed(seed=13, translation_std_m=0.001, rotation_std_rad=0.002)
    joints = np.random.default_rng(72).uniform(-1.5, 1.5, (44, nominal.dof))
    poses = np.asarray([truth.forward(q) for q in joints])
    base_t_camera = np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix()
    base_t_camera[:3, 3] = [0.4, -0.2, 0.7]
    ee_t_target = np.eye(4)
    ee_t_target[:3, :3] = Rotation.from_rotvec([-0.1, 0.05, 0.12]).as_matrix()
    ee_t_target[:3, 3] = [0.03, -0.02, 0.1]
    frame = ObservationFrame.BASE_T_EE
    if camera:
        poses = np.asarray(
            [np.linalg.inv(base_t_camera) @ pose @ ee_t_target for pose in poses]
        )
        frame = ObservationFrame.CAMERA_T_TARGET
    KinematicDataset(joints[:32], poses[:32], frame, np.arange(100, 132)).save(
        tmp_path / "train.npz"
    )
    KinematicDataset(joints[32:], poses[32:], frame, np.arange(1000, 1012)).save(
        tmp_path / "validation.npz"
    )
    np.save(tmp_path / "base_T_camera.npy", base_t_camera)
    np.save(tmp_path / "ee_T_target.npy", ee_t_target)
    return nominal


def _solve_arguments(tmp_path, mode, *, base_link=None):
    arguments = [
        "solve",
        "--backend",
        "numpy",
        "--dataset",
        str(tmp_path / "train.npz"),
        "--urdf",
        str(example_urdf()),
        "--ee-name",
        "ee_link",
        "--model-mode",
        mode,
        "--output",
        str(tmp_path / "fit.npz"),
        "--quiet",
    ]
    if base_link is not None:
        arguments.extend(["--base-link", base_link])
    return arguments


def _independent_saved_errors(tmp_path, *, fixed_camera=False, parameters="fit.npz"):
    dataset = KinematicDataset.load(tmp_path / "train.npz")
    observed = dataset.observed_poses
    if fixed_camera:
        base_t_camera = np.load(tmp_path / "base_T_camera.npy")
        ee_t_target = np.load(tmp_path / "ee_T_target.npy")
        observed = base_t_camera @ observed @ np.linalg.inv(ee_t_target)
    with np.load(tmp_path / parameters) as saved:
        axes = saved["screw_axes"]
        home = saved["zero_configuration_pose"]
    predicted = []
    for joints in dataset.joint_positions:
        pose = np.eye(4)
        for axis, angle in zip(axes, joints, strict=True):
            # Saved space screws use [v,w]; construct the Lie algebra directly.
            vx, vy, vz, wx, wy, wz = axis
            generator = np.array(
                [
                    [0.0, -wz, wy, vx],
                    [wz, 0.0, -wx, vy],
                    [-wy, wx, 0.0, vz],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            pose = pose @ expm(angle * generator)
        predicted.append(pose @ home)
    predicted = np.asarray(predicted)
    position = np.linalg.norm(observed[:, :3, 3] - predicted[:, :3, 3], axis=1)
    relative_rotation = observed[:, :3, :3] @ predicted[:, :3, :3].transpose(0, 2, 1)
    # Rotation vectors retain accuracy near zero, unlike arccos(trace).
    orientation = Rotation.from_matrix(relative_rotation).magnitude()
    return {
        "position_error_mean_m": float(position.mean()),
        "position_error_max_m": float(position.max()),
        "orientation_error_mean_rad": float(orientation.mean()),
        "orientation_error_max_rad": float(orientation.max()),
    }


def _noisy_data(tmp_path):
    _data(tmp_path)
    dataset = KinematicDataset.load(tmp_path / "train.npz")
    rng = np.random.default_rng(981)
    encoders = dataset.joint_positions + rng.normal(
        0.0, 0.0002, dataset.joint_positions.shape
    )
    observed = dataset.observed_poses.copy()
    observed[:, :3, 3] += rng.normal(0.0, 0.0005, (len(observed), 3))
    observed[:, :3, :3] = (
        Rotation.from_rotvec(rng.normal(0.0, 0.001, (len(observed), 3))).as_matrix()
        @ observed[:, :3, :3]
    )
    KinematicDataset(
        encoders, observed, dataset.observation_frame, dataset.sample_indices
    ).save(tmp_path / "train.npz")


@pytest.mark.parametrize(
    "mode,camera",
    [
        ("robot-body", False),
        ("robot-body", True),
        ("self-calibrating", True),
    ],
)
def test_dataset_solve_verify_and_independent_validation_without_rtb(
    tmp_path, monkeypatch, mode, camera
):
    monkeypatch.setitem(sys.modules, "roboticstoolbox", None)
    _data(tmp_path, camera=camera)
    solve = _solve_arguments(tmp_path, mode)
    external = []
    if camera and mode == "robot-body":
        external = [
            "--base-t-camera",
            str(tmp_path / "base_T_camera.npy"),
            "--ee-t-target",
            str(tmp_path / "ee_T_target.npy"),
        ]
    assert main(solve + external) == 0
    report = json.loads((tmp_path / "fit.json").read_text())
    assert report["backend"] == "numpy"
    assert report["accepted"]
    expected = _independent_saved_errors(
        tmp_path, fixed_camera=camera and mode == "robot-body"
    )
    for metric, value in expected.items():
        assert report[metric] == pytest.approx(value, rel=1e-10, abs=1e-12)
    assert not report["full_urdf_export_supported"]
    assert not report["partial_urdf_export_supported"]
    assert report["urdf_joint_definitions"] is None
    assert report["joint_names"] == [f"Joint{index}" for index in range(1, 7)]
    assert (
        main(
            [
                "verify-result",
                "--dataset",
                str(tmp_path / "train.npz"),
                "--urdf",
                str(example_urdf()),
                "--result",
                str(tmp_path / "fit.json"),
                "--parameters",
                str(tmp_path / "fit.npz"),
                *external,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(tmp_path / "fit.npz"),
                "--dataset",
                str(tmp_path / "validation.npz"),
                "--output",
                str(tmp_path / "validation.json"),
                *external,
            ]
        )
        == 0
    )
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["independent_validation"]
    assert validation["position_error_max_m"] < 1e-6
    assert validation["orientation_error_max_rad"] < 1e-6
    assert (
        main(
            [
                "export-urdf",
                "--result",
                str(tmp_path / "fit.json"),
                "--urdf",
                str(example_urdf()),
                "--output",
                str(tmp_path / "invalid.urdf"),
                "--allow-partial",
            ]
        )
        == 2
    )
    assert not (tmp_path / "invalid.urdf").exists()


@pytest.mark.parametrize(
    "mode,camera",
    [("robot-body", False), ("robot-body", True), ("self-calibrating", True)],
)
def test_one_iteration_report_describes_saved_parameters(tmp_path, mode, camera):
    _data(tmp_path, camera=camera)
    arguments = _solve_arguments(tmp_path, mode) + ["--max-iterations", "1"]
    if camera and mode == "robot-body":
        arguments += [
            "--base-t-camera",
            str(tmp_path / "base_T_camera.npy"),
            "--ee-t-target",
            str(tmp_path / "ee_T_target.npy"),
        ]
    assert main(arguments) == 1
    report = json.loads((tmp_path / "fit.json").read_text())
    assert not report["converged"]
    assert not report["accepted"]
    assert report["termination_reason"] == "max_iterations"
    assert report["quality_failures"] == ["solver did not converge"]
    expected = _independent_saved_errors(
        tmp_path, fixed_camera=camera and mode == "robot-body"
    )
    for metric, value in expected.items():
        assert report[metric] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_converged_noisy_report_describes_saved_parameters(tmp_path):
    _noisy_data(tmp_path)
    assert main(_solve_arguments(tmp_path, "robot-body")) == 0
    report = json.loads((tmp_path / "fit.json").read_text())
    assert report["converged"] and report["full_rank"] and report["accepted"]
    for metric, value in _independent_saved_errors(tmp_path).items():
        assert report[metric] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_position_threshold_uses_converged_saved_model(tmp_path):
    _noisy_data(tmp_path)
    arguments = _solve_arguments(tmp_path, "robot-body")
    assert main(arguments) == 0
    final_error = _independent_saved_errors(tmp_path)["position_error_max_m"]
    # A relative 1e-6 margin is well above matrix-exponential roundoff.
    threshold = final_error * (1.0 - 1e-6)
    arguments[arguments.index("--output") + 1] = str(tmp_path / "constrained.npz")
    status = main(arguments + ["--max-position-error", str(threshold)])
    report = json.loads((tmp_path / "constrained.json").read_text())
    assert report["converged"] and report["full_rank"]
    assert not report["dataset_quality_warnings"]
    assert report["condition_number"] < report["quality_thresholds"][
        "max_condition_number"
    ]
    expected = _independent_saved_errors(tmp_path, parameters="constrained.npz")
    assert expected["position_error_max_m"] > threshold
    assert expected["orientation_error_max_rad"] < report["quality_thresholds"][
        "max_orientation_error_rad"
    ]
    assert not report["accepted"]
    assert status == 1
    assert len(report["quality_failures"]) == 1
    assert report["quality_failures"][0].startswith("maximum position error ")


def test_numpy_can_fit_a_selected_subchain_in_its_explicit_base_frame(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "roboticstoolbox", None)
    nominal = _data(tmp_path, base_link="Link1")
    assert nominal.dof == 5
    assert main(_solve_arguments(tmp_path, "robot-body", base_link="Link1")) == 0
    report = json.loads((tmp_path / "fit.json").read_text())
    assert report["base_link"] == "Link1"
    assert report["dof"] == 5
    assert report["robot_total_joints"] == 6
    assert report["joint_names"] == [f"Joint{index}" for index in range(2, 7)]


@pytest.mark.parametrize(
    "camera,base_link", [(False, None), (True, None), (False, "Link1")]
)
def test_unified_numpy_geometry_workflow_verifies_poe_without_urdf_export(
    tmp_path, monkeypatch, camera, base_link
):
    monkeypatch.setitem(sys.modules, "roboticstoolbox", None)
    _data(tmp_path, camera=camera, base_link=base_link)
    output = tmp_path / "calibration"
    arguments = [
        "calibrate",
        "--backend",
        "numpy",
        "--calibration-type",
        "geometry",
        "--dataset",
        str(tmp_path / "train.npz"),
        "--urdf",
        str(example_urdf()),
        "--output-dir",
        str(output),
        "--skip-handeye",
    ]
    if base_link is not None:
        arguments.extend(["--base-link", base_link])
    assert main(arguments) == 0
    manifest = json.loads((output / "calibration_manifest.json").read_text())
    assert manifest["accepted"] and manifest["backend"] == "numpy"
    assert manifest["base_link"] == base_link
    geometry = manifest["stages"]["geometry"]
    assert geometry["verification_exit_code"] == 0
    assert geometry["exported_urdf"] is None
    assert "link frames" in geometry["export_skipped_reason"]
    assert not list(output.glob("*.urdf"))
    result = json.loads((output / "final_result.json").read_text())
    assert result["kinematic_calibration"]["parameters"] == geometry["parameters"]
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                geometry["parameters"],
                "--dataset",
                str(tmp_path / "validation.npz"),
                "--output",
                str(output / "validation.json"),
            ]
        )
        == 0
    )
    validation = json.loads((output / "validation.json").read_text())
    assert validation["independent_validation"]
    assert validation["position_error_max_m"] < 1e-6


@pytest.mark.parametrize("calibration_type", ["both", "zero-offset"])
def test_numpy_unified_workflow_rejects_unsupported_stages_before_writing(
    tmp_path, caplog, calibration_type
):
    output = tmp_path / "calibration"
    assert (
        main(
            [
                "calibrate",
                "--backend",
                "numpy",
                "--calibration-type",
                calibration_type,
                "--dataset",
                str(tmp_path / "missing.npz"),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert "requires --calibration-type geometry" in caplog.text
    assert not output.exists()


def test_old_parser_keeps_rtb_as_its_explicit_compatibility_default():
    args = build_parser().parse_args(
        [
            "solve",
            "--dataset",
            "data.npz",
            "--urdf",
            "robot.urdf",
            "--ee-name",
            "ee_link",
            "--model-mode",
            "robot-body",
            "--output",
            "fit.npz",
        ]
    )
    assert args.backend == "rtb"
    assert args.base_link is None


@pytest.mark.parametrize(
    "option",
    ["--experimental-nominal-allocation", "--experimental-svd-gauge-allocation"],
)
def test_numpy_rejects_urdf_frame_allocation_options(tmp_path, option):
    _data(tmp_path)
    assert main(_solve_arguments(tmp_path, "robot-body") + [option]) == 2
    assert not (tmp_path / "fit.npz").exists()


def test_result_verification_binds_numpy_base_frame_metadata(tmp_path):
    _data(tmp_path)
    assert main(_solve_arguments(tmp_path, "robot-body")) == 0
    report_path = tmp_path / "fit.json"
    report = json.loads(report_path.read_text())
    report["base_link"] = "unrelated"
    report_path.write_text(json.dumps(report))
    assert (
        main(
            [
                "verify-result",
                "--dataset",
                str(tmp_path / "train.npz"),
                "--urdf",
                str(example_urdf()),
                "--result",
                str(report_path),
                "--parameters",
                str(tmp_path / "fit.npz"),
            ]
        )
        == 2
    )
