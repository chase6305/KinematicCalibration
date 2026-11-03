"""Verify the public in-memory workflow independently of CLI file handling."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from kinematic_calib import FitOptions, KinematicDataset, ObservationFrame, fit_dataset
from kinematic_calib.validation import evaluate_saved_poe


def _model():
    axes = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-0.2, 0.0, 0.1, 0.0, 1.0, 0.0],
            [0.0, 0.3, -0.1, 1.0, 0.0, 0.0],
        ]
    )
    home = np.eye(4)
    home[:3, 3] = [0.5, 0.1, 0.4]
    return axes, home


def _forward(axes, home, configurations):
    poses = []
    for q in configurations:
        pose = np.eye(4)
        for axis, angle in zip(axes, q, strict=True):
            vx, vy, vz, wx, wy, wz = axis
            generator = np.array(
                [
                    [0.0, -wz, wy, vx],
                    [wz, 0.0, -wx, vy],
                    [-wy, wx, 0.0, vz],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            pose = pose @ expm(generator * angle)
        poses.append(pose @ home)
    return np.asarray(poses)


@pytest.mark.parametrize(
    "options",
    [
        {"model_mode": "unknown"},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"damping": -1.0},
        {"damping": float("nan")},
    ],
)
def test_fit_options_reject_invalid_controls(options):
    with pytest.raises(ValueError):
        FitOptions(**options)


def test_in_memory_fit_recovers_terminal_perturbation_on_held_out_configurations():
    pytest.importorskip("spatialmath")
    from robotkinecal import SerialRobotKineCal

    axes, home = _model()
    truth_home = home.copy()
    truth_home[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.015]).as_matrix()
    truth_home[:3, 3] += [0.003, -0.002, 0.004]
    joints = np.random.default_rng(81).uniform(-1.5, 1.5, (32, len(axes)))
    observed = _forward(axes, truth_home, joints)
    dataset = KinematicDataset(joints, observed, ObservationFrame.BASE_T_EE)
    original_joints = joints.copy()
    original_observed = observed.copy()

    fit = fit_dataset(dataset, SerialRobotKineCal.from_screw_axes(axes, home))

    validation_joints = np.random.default_rng(92).uniform(-1.5, 1.5, (16, len(axes)))
    errors = evaluate_saved_poe(
        fit.screw_axes,
        fit.zero_pose.A,
        validation_joints,
        _forward(axes, truth_home, validation_joints),
    )
    assert fit.result.has_converged
    assert errors["position_error_max_m"] < 1e-7
    assert errors["orientation_error_max_rad"] < 1e-7
    assert np.array_equal(joints, original_joints)
    assert np.array_equal(observed, original_observed)


def test_camera_observation_conversion_preserves_body_fit():
    pytest.importorskip("spatialmath")
    from robotkinecal import SerialRobotKineCal

    axes, home = _model()
    joints = np.random.default_rng(83).uniform(-1.5, 1.5, (20, len(axes)))
    base_t_camera = np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.2, 0.3, -0.1]).as_matrix()
    base_t_camera[:3, 3] = [0.4, 0.2, 0.8]
    ee_t_target = np.eye(4)
    ee_t_target[:3, 3] = [0.03, 0.02, 0.1]
    observed = np.asarray(
        [
            np.linalg.inv(base_t_camera) @ pose @ ee_t_target
            for pose in _forward(axes, home, joints)
        ]
    )
    dataset = KinematicDataset(joints, observed, ObservationFrame.CAMERA_T_TARGET)
    fit = fit_dataset(
        dataset,
        SerialRobotKineCal.from_screw_axes(axes, home),
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
    )

    assert fit.result.has_converged
    assert fit.dataset.observation_frame == ObservationFrame.BASE_T_EE
    assert dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET
    assert np.allclose(fit.dataset.observed_poses, _forward(axes, home, joints))


def test_unified_workflow_selects_body_mode_without_handeye(tmp_path, monkeypatch):
    """Direct pose datasets from headless engines need no extrinsic solve."""
    from kinematic_calib.cli import main

    dataset_path = tmp_path / "poses.npz"
    axes, home = _model()
    joints = np.random.default_rng(84).uniform(-1.5, 1.5, (20, len(axes)))
    KinematicDataset(
        joints,
        _forward(axes, home, joints),
        ObservationFrame.BASE_T_EE,
    ).save(dataset_path)
    urdf = tmp_path / "nominal.urdf"
    urdf.write_text("<robot name='test'/>")
    selected = []
    monkeypatch.setattr(
        "kinematic_calib.cli._solve", lambda args: selected.append(args.model_mode) or 0
    )
    monkeypatch.setattr("kinematic_calib.cli._verify_result", lambda args: 0)
    monkeypatch.setattr(
        "kinematic_calib.cli._export_urdf",
        lambda args: selected.append(args.output) or 0,
    )

    output = tmp_path / "artifacts"
    code = main(
        [
            "calibrate",
            "--dataset",
            str(dataset_path),
            "--urdf",
            str(urdf),
            "--calibration-type",
            "geometry",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert selected == ["robot-body", output / "nominal_calibrated.urdf"]


def test_unified_workflow_rejects_invalid_archive_before_writing_outputs(tmp_path):
    from kinematic_calib.cli import main

    dataset = tmp_path / "invalid.npz"
    dataset.write_bytes(b"not an archive")
    urdf = tmp_path / "nominal.urdf"
    urdf.write_text("<robot name='test'/>")
    output = tmp_path / "artifacts"
    assert (
        main(
            [
                "calibrate",
                "--dataset",
                str(dataset),
                "--urdf",
                str(urdf),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
