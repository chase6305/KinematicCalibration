"""Optional numerical integration tests for the vendored RobotKineCal solver."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kinematic_calib.benchmark import run_perturbation_benchmark
from kinematic_calib.robot_model import (
    estimate_composite_extrinsics,
    estimate_joint_zero_offsets,
)

rtb = pytest.importorskip("roboticstoolbox")
robotkinecal = pytest.importorskip("robotkinecal")


def test_pure_poe_model_uses_in_memory_service_without_urdf_link_frames():
    from spatialmath import SE3, Twist3

    from kinematic_calib import KinematicDataset, ObservationFrame, fit_dataset
    from kinematic_calib.validation import evaluate_saved_poe
    from kinematic_calibration.backends.numpy import NumpyBackend
    from kinematic_calibration.experiment import example_urdf

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    axes, home = nominal.poe_parameters()
    robot = rtb.PoERobot(
        [rtb.PoELink(Twist3(axis)) for axis in axes], SE3(home, check=False)
    )
    calibrator = robotkinecal.SerialRobotKineCal(robot)
    assert calibrator.joint_zero_conf_poses is None
    truth = nominal.perturbed(seed=59)
    joints = np.random.default_rng(58).uniform(-1.5, 1.5, (40, nominal.dof))
    dataset = KinematicDataset(
        joints[:30],
        np.asarray([truth.forward(row) for row in joints[:30]]),
        ObservationFrame.BASE_T_EE,
        joint_names=nominal.joint_names,
    )

    fit = fit_dataset(dataset, calibrator, model_joint_names=nominal.joint_names)
    errors = evaluate_saved_poe(
        fit.screw_axes,
        fit.zero_pose.A,
        joints[30:],
        np.asarray([truth.forward(row) for row in joints[30:]]),
    )

    assert fit.result.has_converged
    assert fit.joint_identity["verified"]
    assert errors["position_error_max_m"] < 1e-7
    assert errors["orientation_error_max_rad"] < 1e-7


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


def _selected_chain_urdf(tmp_path, case, *, fixed_path=None):
    intermediate = case != "full"
    branched = case == "branched"
    root = ET.Element("robot", name="selected_chain")
    # RTB assigns global indices in link traversal order. Put the unrelated
    # motor first so the selected columns map to [1, 2], not [0, 1].
    links = ["base"] + (["side"] if branched else [])
    links += ["shoulder", "elbow", "wrist", "tip"]
    if intermediate:
        links.append("tool")
    if fixed_path is not None:
        links.append("spacer")
    for name in links:
        ET.SubElement(root, "link", name=name)

    def joint(name, parent, child, xyz, axis=None):
        element = ET.Element(
            "joint", name=name, type="revolute" if axis is not None else "fixed"
        )
        ET.SubElement(element, "parent", link=parent)
        ET.SubElement(element, "child", link=child)
        ET.SubElement(element, "origin", xyz=xyz, rpy="0 0 0")
        if axis is not None:
            ET.SubElement(element, "axis", xyz=axis)
            ET.SubElement(
                element, "limit", lower="-3", upper="3", effort="5", velocity="2"
            )
        return element

    # XML declaration order differs from selected chain order; wrist is the
    # elbow's first child even when the requested end is the fixed tool.
    joints = [
        joint("j_wrist", "elbow", "wrist", "0.1 0.05 0.2", "1 0 0"),
        joint("j_tip", "wrist", "tip", "0.1 0.2 0.1"),
    ]
    if intermediate:
        joints.append(joint("j_tool", "elbow", "tool", "0.2 0.1 0.3"))
    if branched:
        joints.append(joint("j_side", "base", "side", "-0.1 0.2 0.05", "0 1 0"))
    joints.extend(
        [
            joint(
                "j_elbow", "spacer" if fixed_path == "between" else "shoulder",
                "elbow", "0.3 0 0.2", "0 1 0",
            ),
            joint(
                "j_shoulder", "spacer" if fixed_path == "before" else "base",
                "shoulder", "0.1 0.2 0.1", "0 0 1",
            ),
        ]
    )
    end = "tool" if intermediate else "tip"
    if fixed_path is not None:
        parent = {"before": "base", "between": "shoulder", "tail": end}[fixed_path]
        joints.append(joint("j_spacer", parent, "spacer", "0.02 0.03 0.04"))
        if fixed_path == "tail":
            end = "spacer"
    root.extend(joints)
    path = tmp_path / "selected.urdf"
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, end


@pytest.mark.parametrize("case", ["full", "intermediate", "branched"])
def test_selected_urdf_chain_fits_and_exports_in_local_joint_order(tmp_path, case):
    from kinematic_calib import KinematicDataset, ObservationFrame
    from kinematic_calib.cli import main
    from kinematic_calib.robot_model import load_calibration_model, validate_nominal_poe
    from kinematic_calibration.backends.numpy import NumpyBackend

    source, end = _selected_chain_urdf(tmp_path, case)
    source_bytes = source.read_bytes()
    nominal = NumpyBackend.from_urdf(source, end_link=end)
    model = load_calibration_model(source, end, backend="rtb")
    calibrator = model.calibrator
    expected_indices = [1, 2] if case == "branched" else list(range(nominal.dof))
    assert [joint.jindex for joint in calibrator.joints] == expected_indices
    assert tuple(model.joint_names) == nominal.joint_names
    if case != "full":
        assert calibrator.joints[-1].children[0].name == "wrist"
    joints = np.random.default_rng(791).uniform(-1.0, 1.0, (40, nominal.dof))
    validate_nominal_poe(calibrator, joints[0])
    np.testing.assert_allclose(
        calibrator.forward_kinematics_batch(joints),
        [nominal.forward(q) for q in joints], atol=1e-12, rtol=0,
    )
    # Retain the four original positional result arguments.
    legacy = robotkinecal.CalibrationResult(
        calibrator.robot_model, 1e-5, calibrator.joints, 7
    )
    assert legacy.convergence_tolerance == 1e-5
    assert legacy.divergence_patience == 7

    truth = nominal.perturbed(seed=47, translation_std_m=0.001, rotation_std_rad=0.002)
    training = tmp_path / "training.npz"
    KinematicDataset(
        joints[:30], np.asarray([truth.forward(q) for q in joints[:30]]),
        ObservationFrame.BASE_T_EE, joint_names=nominal.joint_names,
    ).save(training)
    parameters = tmp_path / "fit.npz"
    assert main(
        [
            "solve", "--backend", "rtb", "--dataset", str(training),
            "--urdf", str(source), "--ee-name", end,
            "--model-mode", "robot-body", "--output", str(parameters), "--quiet",
        ]
    ) == 0
    report_path = parameters.with_suffix(".json")
    report = json.loads(report_path.read_text())
    expected_terminal = "wrist-tip" if case == "full" else "elbow-tool"
    assert report["raw_solver_urdf_joint_definitions"][-1]["name"] == expected_terminal
    assert report["joint_names"] == list(nominal.joint_names)
    exported = tmp_path / "calibrated.urdf"
    assert main(
        [
            "export-urdf", "--result", str(report_path),
            "--urdf", str(source), "--output", str(exported),
        ]
    ) == 0
    calibrated = NumpyBackend.from_urdf(exported, end_link=end)
    np.testing.assert_allclose(
        [calibrated.forward(q) for q in joints[30:]],
        [truth.forward(q) for q in joints[30:]], atol=1e-7, rtol=0,
    )
    before = {joint.get("name"): joint for joint in ET.parse(source).getroot().findall("joint")}
    after = {joint.get("name"): joint for joint in ET.parse(exported).getroot().findall("joint")}
    changed_names = set(nominal.joint_names) | {"j_tip" if case == "full" else "j_tool"}
    for name in set(before) - changed_names:
        assert after[name].find("origin").attrib == before[name].find("origin").attrib
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("case", ["full", "intermediate", "branched"])
def test_selected_chain_zero_offsets_use_global_fk_indices(tmp_path, monkeypatch, case):
    from kinematic_calib.robot_model import load_calibration_model
    from kinematic_calibration.backends.numpy import NumpyBackend

    source, end = _selected_chain_urdf(tmp_path, case)
    truth = NumpyBackend.from_urdf(source, end_link=end)
    model = load_calibration_model(source, end, backend="rtb")
    robot = model.calibrator.robot_model
    indices = [joint.jindex for joint in model.calibrator.joints]
    actual = np.random.default_rng(792).uniform(-1.0, 1.0, (24, truth.dof))
    bias = np.array([0.012, -0.021, 0.016])[:truth.dof]
    recorded = actual + bias
    observed = np.asarray([truth.forward(q) for q in actual])
    received_positions = []
    original_fkine = robot.fkine

    def checked_fkine(positions, **kwargs):
        received_positions.append(np.asarray(positions).copy())
        return original_fkine(positions, **kwargs)

    monkeypatch.setattr(robot, "fkine", checked_fkine)
    result = estimate_joint_zero_offsets(
        robot, end, recorded, observed,
        camera_t_base=np.eye(4), ee_t_target=np.eye(4),
    )

    assert result["full_rank"]
    assert result["estimated_indices"].tolist() == list(range(truth.dof))
    np.testing.assert_allclose(result["offsets"], -bias, atol=1e-8, rtol=0)
    assert result["position_error_max_m"] < 1e-8
    assert result["orientation_error_max_rad"] < 1e-8
    positions = np.asarray(received_positions)
    assert positions.shape[1] == robot.n
    unselected = sorted(set(range(robot.n)) - set(indices))
    np.testing.assert_array_equal(positions[:, unselected], 0.0)


@pytest.mark.parametrize("fixed_path", ["before", "between", "tail"])
def test_urdf_export_rejects_unrepresentable_fixed_paths(tmp_path, fixed_path):
    from kinematic_calib import (
        FitOptions,
        KinematicDataset,
        ObservationFrame,
        fit_dataset,
    )
    from kinematic_calib.robot_model import load_calibration_model
    from kinematic_calibration.backends.numpy import NumpyBackend

    source, end = _selected_chain_urdf(tmp_path, "branched", fixed_path=fixed_path)
    truth = NumpyBackend.from_urdf(source, end_link=end)
    model = load_calibration_model(source, end, backend="rtb")
    joints = np.random.default_rng(793).uniform(-1.0, 1.0, (20, truth.dof))
    dataset = KinematicDataset(
        joints, np.asarray([truth.forward(q) for q in joints]),
        ObservationFrame.BASE_T_EE, joint_names=truth.joint_names,
    )
    fit = fit_dataset(dataset, model.calibrator, options=FitOptions(max_iterations=3))
    assert fit.result.ee_name == end
    with pytest.raises(ValueError, match="URDF export requires"):
        model.calibrator.get_urdf_xyzrpy(fit.result)


def test_partial_definitions_reject_fixed_paths_between_active_joints(tmp_path):
    from kinematic_calib.robot_model import (
        get_identifiable_urdf_joint_definitions,
        load_calibration_model,
    )

    source, end = _selected_chain_urdf(tmp_path, "branched", fixed_path="between")
    model = load_calibration_model(source, end, backend="rtb")
    assert model.calibrator.joints[1].parent.name == "spacer"
    with pytest.raises(ValueError, match="fixed paths between active joints"):
        get_identifiable_urdf_joint_definitions(model.calibrator, np.eye(4))


@pytest.mark.parametrize(
    "fixed_path", [None, "before", "tail", "between"],
    ids=["adjacent", "before-first", "after-last", "between-active"],
)
def test_self_calibrating_partial_definitions_respect_fixed_path_scope(
    tmp_path, caplog, capsys, fixed_path
):
    from spatialmath import SE3

    from kinematic_calib import KinematicDataset, ObservationFrame
    from kinematic_calib.cli import main
    from kinematic_calibration.backends.numpy import NumpyBackend

    source, end = _selected_chain_urdf(tmp_path, "branched", fixed_path=fixed_path)
    truth = NumpyBackend.from_urdf(source, end_link=end)
    joints = np.random.default_rng(794).uniform(-1.0, 1.0, (24, truth.dof))
    camera_t_base = SE3.RPY([0.2, -0.1, 0.3]) @ SE3.Trans(0.4, -0.2, 1.1)
    ee_t_target = SE3.RPY([-0.1, 0.15, 0.05]) @ SE3.Trans(0.03, -0.08, 0.12)
    observations = np.asarray(
        [camera_t_base.A @ truth.forward(q) @ ee_t_target.A for q in joints]
    )
    dataset = tmp_path / "camera-observations.npz"
    KinematicDataset(
        joints, observations, ObservationFrame.CAMERA_T_TARGET,
        joint_names=truth.joint_names,
    ).save(dataset)
    parameters = tmp_path / "fit.npz"
    report_path = parameters.with_suffix(".json")
    code = main(
        [
            "solve", "--backend", "rtb", "--dataset", str(dataset),
            "--urdf", str(source), "--ee-name", end,
            "--model-mode", "self-calibrating", "--output", str(parameters),
            "--quiet",
        ]
    )
    if fixed_path == "between":
        assert code == 2
        assert "fixed paths between active joints" in caplog.text
        captured = capsys.readouterr()
        assert "Traceback" not in caplog.text + captured.out + captured.err
        assert not parameters.exists()
        assert not report_path.exists()
        return

    assert code == 0
    report = json.loads(report_path.read_text())
    definitions = report["identifiable_urdf_joint_definitions"]
    assert report["partial_urdf_export_supported"] is True
    assert report["full_urdf_export_supported"] is False
    assert len(definitions) == 1
    assert definitions[0]["joint_name"] == "j_elbow"
    assert definitions[0]["parent_link"] == "shoulder"
    assert definitions[0]["child_link"] == "elbow"
    expected = np.eye(4)
    expected[:3, 3] = [0.3, 0.0, 0.2]
    # Match the existing extrinsic initialization/fit accuracy, rather than
    # requiring its least-squares termination to reach matrix-roundoff error.
    np.testing.assert_allclose(definitions[0]["transform"], expected, atol=1e-7, rtol=0)
    exported = tmp_path / "partial.urdf"
    assert main(
        [
            "export-urdf", "--result", str(report_path), "--urdf", str(source),
            "--output", str(exported), "--allow-partial",
        ]
    ) == 0
    partial = NumpyBackend.from_urdf(exported, end_link=end)
    np.testing.assert_allclose(
        [partial.forward(q) for q in joints],
        [truth.forward(q) for q in joints], atol=1e-7, rtol=0,
    )
