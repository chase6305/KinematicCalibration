"""Optional numerical integration tests for the vendored RobotKineCal solver."""

import numpy as np
import pytest

from kinematic_calib.benchmark import run_perturbation_benchmark
from kinematic_calib.robot_model import (
    estimate_composite_extrinsics,
    estimate_joint_zero_offsets,
)

rtb = pytest.importorskip("roboticstoolbox")
robotkinecal = pytest.importorskip("robotkinecal")


@pytest.mark.parametrize("dof", [1, 2, 4, 7, 8])
def test_revolute_serial_chains_of_different_dof_converge(dof: int) -> None:
    links = [
        rtb.RevoluteDH(
            a=0.08 + 0.01 * index,
            d=0.04 + 0.005 * (index % 3),
            alpha=(-1) ** index * (0.25 + 0.03 * index),
            offset=0.02 * index,
        )
        for index in range(dof)
    ]
    robot = rtb.DHRobot(links, name=f"test_{dof}dof")
    sample_count = max(12, int(np.ceil((4 * dof + 6) / 6)) + 6)
    configurations = np.random.default_rng(1000 + dof).uniform(
        -1.5, 1.5, (sample_count, dof)
    )
    observations = [robot.fkine(configuration) for configuration in configurations]
    calibrator = robotkinecal.SerialRobotKineCal(robot, verbose=False)
    calibrator.set_observations(configurations, observations)

    result = calibrator.solve(max_iterations=10)
    final_iteration = result.iteration_results[-1]
    statistics = final_iteration.get_statistics()

    assert result.has_converged
    assert calibrator.N_JOINTS == dof
    assert statistics["matrix_rank"] == final_iteration.N_PARAMS


def test_known_kinematic_perturbation_is_recovered_on_held_out_poses() -> None:
    links = [
        rtb.RevoluteDH(a=0.1 + 0.02 * index, d=0.04, alpha=(-1) ** index * 0.4)
        for index in range(3)
    ]
    robot = rtb.DHRobot(links, name="perturbation_recovery")

    report = run_perturbation_benchmark(
        robot,
        robot.links[-1].name,
        train_samples=30,
        validation_samples=20,
        perturbation=0.01,
        seed=42,
    )

    assert report["converged"]
    assert report["regressor_rank"] == report["parameter_count"]
    assert report["screw_axis_mean_improvement"] > 0.99
    assert report["terminal_error_improvement"] > 0.99
    assert report["validation_position_mean_improvement"] > 0.99
    assert report["validation_orientation_mean_improvement"] > 0.99


def test_perturbation_benchmark_injects_reproducible_measurement_noise() -> None:
    links = [
        rtb.RevoluteDH(a=0.12 + 0.02 * index, d=0.05, alpha=(-1) ** index * 0.4)
        for index in range(3)
    ]
    robot = rtb.DHRobot(links, name="noisy_perturbation_recovery")
    arguments = dict(
        robot=robot,
        ee_name=robot.links[-1].name,
        train_samples=40,
        validation_samples=20,
        perturbation=0.01,
        seed=77,
        position_noise_std_m=0.001,
        orientation_noise_std_rad=0.002,
        joint_noise_std_rad=0.0005,
        outlier_ratio=0.1,
        outlier_position_std_m=0.01,
        outlier_orientation_std_rad=0.02,
    )

    first = run_perturbation_benchmark(**arguments)
    second = run_perturbation_benchmark(**arguments)

    assert first["noise_configuration"] == second["noise_configuration"]
    assert first["schema_version"] == 1
    assert first["report_type"] == "kinematic-perturbation-recovery-benchmark"
    assert first["injected_noise_statistics"] == second["injected_noise_statistics"]
    assert first["validation_after"] == second["validation_after"]
    assert first["noise_configuration"]["outlier_count"] == 4
    assert len(first["noise_configuration"]["outlier_indices"]) == 4
    assert first["injected_noise_statistics"]["joint_error_rms_rad"] > 0.0
    assert first["injected_noise_statistics"]["position_mean_m"] > 0.0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("position_noise_std_m", -0.001, "standard deviations"),
        ("orientation_noise_std_rad", float("nan"), "standard deviations"),
        ("joint_noise_std_rad", -0.001, "standard deviations"),
        ("outlier_ratio", 1.1, "outlier_ratio"),
    ],
)
def test_perturbation_benchmark_rejects_invalid_noise_configuration(
    keyword: str, value: float, message: str
) -> None:
    robot = rtb.DHRobot([rtb.RevoluteDH(a=0.1)], name="invalid_noise")

    with pytest.raises(ValueError, match=message):
        run_perturbation_benchmark(
            robot,
            robot.links[-1].name,
            **{keyword: value},
        )


def test_ur10_cli_export_matches_independent_truth_fk(tmp_path, capsys):
    """Exercise file artifacts and actual non-Z-axis URDF model export."""
    from importlib.resources import files
    from pathlib import Path

    from kinematic_calib import KinematicDataset, ObservationFrame
    from kinematic_calib.cli import main
    from kinematic_calibration.backends.numpy import NumpyBackend

    nominal_path = Path(
        str(files("kinematic_calibration") / "assets" / "robots" / "ur10.urdf")
    )
    nominal = NumpyBackend.from_urdf(nominal_path, end_link="ee_link")
    truth = nominal.perturbed(
        seed=47,
        translation_std_m=0.001,
        rotation_std_rad=0.002,
    )
    joints = np.random.default_rng(41).uniform(-2.0, 2.0, (40, 6))
    poses = np.asarray([truth.forward(q) for q in joints])
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    KinematicDataset(
        joints[:30],
        poses[:30],
        ObservationFrame.BASE_T_EE,
        joint_names=nominal.joint_names,
    ).save(train)
    KinematicDataset(
        joints[30:],
        poses[30:],
        ObservationFrame.BASE_T_EE,
        np.arange(30, 40),
        nominal.joint_names,
    ).save(validation)
    parameters = tmp_path / "fit.npz"
    report = parameters.with_suffix(".json")
    exported = tmp_path / "calibrated.urdf"

    assert (
        main(
            [
                "solve",
                "--dataset",
                str(train),
                "--urdf",
                str(nominal_path),
                "--ee-name",
                "ee_link",
                "--output",
                str(parameters),
                "--model-mode",
                "robot-body",
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "verify-result",
                "--result",
                str(report),
                "--dataset",
                str(train),
                "--urdf",
                str(nominal_path),
                "--parameters",
                str(parameters),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(validation),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "export-urdf",
                "--result",
                str(report),
                "--urdf",
                str(nominal_path),
                "--output",
                str(exported),
            ]
        )
        == 0
    )

    calibrated = NumpyBackend.from_urdf(exported, end_link="ee_link")
    for q in joints[30:]:
        assert np.allclose(calibrated.forward(q), truth.forward(q), atol=1e-7)

    # An accepted fit must not make its own optimization samples independent.
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(train),
            ]
        )
        == 2
    )


def test_composite_extrinsics_are_initialized_without_external_priors() -> None:
    from spatialmath import SE3

    from robotkinecal import SerialRobotKineCal

    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=0.3, d=0.1, alpha=0.4),
            rtb.RevoluteDH(a=0.2, d=0.05, alpha=-0.3),
            rtb.RevoluteDH(a=0.15, d=0.08, alpha=0.2),
        ]
    )
    calibrator = SerialRobotKineCal(robot, verbose=False)
    joints = np.random.default_rng(7).uniform(-1.5, 1.5, (24, 3))
    camera_t_base = SE3.RPY([0.2, -0.1, 0.3]) * SE3.Trans(0.4, -0.2, 1.1)
    ee_t_target = SE3.RPY([-0.1, 0.15, 0.05]) * SE3.Trans(0.03, -0.08, 0.12)
    observations = np.asarray(
        [(camera_t_base * robot.fkine(q) * ee_t_target).A for q in joints]
    )

    estimated_camera_t_base, estimated_ee_t_target, report = (
        estimate_composite_extrinsics(calibrator, joints, observations)
    )

    assert report["success"]
    assert np.allclose(estimated_camera_t_base, camera_t_base.A, atol=1e-7)
    assert np.allclose(estimated_ee_t_target, ee_t_target.A, atol=1e-7)


def test_internal_zero_offsets_are_recovered_without_external_transforms() -> None:
    from spatialmath import SE3

    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=0.25, d=0.1, alpha=0.4),
            rtb.RevoluteDH(a=0.2, d=0.08, alpha=-0.3),
            rtb.RevoluteDH(a=0.18, d=0.06, alpha=0.25),
            rtb.RevoluteDH(a=0.12, d=0.04, alpha=-0.2),
        ]
    )
    joints = np.random.default_rng(17).uniform(-1.5, 1.5, (36, 4))
    truth_offsets = np.array([0.0, 0.025, -0.018, 0.0])
    camera_t_base = SE3.RPY([0.2, -0.1, 0.3]) * SE3.Trans(0.4, -0.2, 1.1)
    ee_t_target = SE3.RPY([-0.1, 0.15, 0.05]) * SE3.Trans(0.03, -0.08, 0.12)
    observations = np.asarray(
        [
            (camera_t_base * robot.fkine(q + truth_offsets) * ee_t_target).A
            for q in joints
        ]
    )

    result = estimate_joint_zero_offsets(
        robot, robot.links[-1].name, joints, observations
    )

    assert result["full_rank"]
    assert result["estimated_indices"].tolist() == [1, 2]
    assert np.allclose(result["offsets"], truth_offsets, atol=1e-7)


def test_all_zero_offsets_are_recovered_with_fixed_external_transforms() -> None:
    from spatialmath import SE3

    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=0.25, d=0.1, alpha=0.4),
            rtb.RevoluteDH(a=0.2, d=0.08, alpha=-0.3),
            rtb.RevoluteDH(a=0.18, d=0.06, alpha=0.25),
        ]
    )
    joints = np.random.default_rng(23).uniform(-1.5, 1.5, (30, 3))
    truth_offsets = np.array([0.012, -0.021, 0.016])
    camera_t_base = SE3.RPY([0.2, -0.1, 0.3]) * SE3.Trans(0.4, -0.2, 1.1)
    ee_t_target = SE3.RPY([-0.1, 0.15, 0.05]) * SE3.Trans(0.03, -0.08, 0.12)
    observations = np.asarray(
        [
            (camera_t_base * robot.fkine(q + truth_offsets) * ee_t_target).A
            for q in joints
        ]
    )

    result = estimate_joint_zero_offsets(
        robot,
        robot.links[-1].name,
        joints,
        observations,
        camera_t_base=camera_t_base.A,
        ee_t_target=ee_t_target.A,
    )

    assert result["full_rank"]
    assert result["estimated_indices"].tolist() == [0, 1, 2]
    assert np.allclose(result["offsets"], truth_offsets, atol=1e-8)


def test_injected_encoder_bias_is_recovered_with_opposite_sign() -> None:
    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=0.25, d=0.1, alpha=0.4),
            rtb.RevoluteDH(a=0.2, d=0.08, alpha=-0.3),
            rtb.RevoluteDH(a=0.18, d=0.06, alpha=0.25),
        ]
    )
    actual_joints = np.random.default_rng(29).uniform(-1.5, 1.5, (30, 3))
    injected_bias = np.array([0.012, -0.021, 0.016])
    recorded_joints = actual_joints + injected_bias
    observations = np.asarray([robot.fkine(q).A for q in actual_joints])

    result = estimate_joint_zero_offsets(
        robot,
        robot.links[-1].name,
        recorded_joints,
        observations,
        camera_t_base=np.eye(4),
        ee_t_target=np.eye(4),
    )

    assert result["full_rank"]
    assert np.allclose(result["offsets"], -injected_bias, atol=1e-8)
