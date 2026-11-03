"""Real dataset-file workflow through the NumPy model loader, without RTB."""

import json
import sys

import numpy as np
import pytest
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
