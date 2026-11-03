from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calibration.backends import KinematicsBackend, NumpyBackend
from kinematic_calibration.experiment import example_urdf, predict_poe


def test_arbitrary_axes_and_fixed_tool_match_poe():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    assert isinstance(robot, KinematicsBackend)
    axes, home = robot.poe_parameters()
    joints = np.random.default_rng(4).uniform(-2, 2, (12, 6))
    expected = np.asarray([robot.forward(q) for q in joints])
    np.testing.assert_allclose(predict_poe(axes, home, joints), expected, atol=1e-12)
    assert robot.joint_names == tuple(f"Joint{i}" for i in range(1, 7))


def write_chain(tmp_path: Path, *, kind="revolute", axis="1 1 0", extra=""):
    path = tmp_path / "test.urdf"
    path.write_text(f'''<robot name="test">
    <link name="base"/><link name="arm"/><link name="tool"/>
    <joint name="a" type="{kind}"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 1" rpy="0.1 0.2 0.3"/><axis xyz="{axis}"/>
    <limit lower="-2" upper="2"/>{extra}</joint>
    <joint name="tip" type="fixed"><parent link="arm"/><child link="tool"/>
    <origin xyz="1 0 0"/></joint></robot>''')
    return path


def test_axis_normalization_and_transform_order(tmp_path):
    robot = NumpyBackend.from_urdf(write_chain(tmp_path), end_link="tool")
    origin_r = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    moved_r = (
        origin_r
        @ Rotation.from_rotvec(np.array([1, 1, 0]) / np.sqrt(2) * 0.5).as_matrix()
    )
    expected = np.eye(4)
    expected[:3, :3] = moved_r
    expected[:3, 3] = [0, 0, 1] + moved_r[:, 0]
    np.testing.assert_allclose(robot.forward([0.5]), expected, atol=1e-14)


@pytest.mark.parametrize(
    "kind,axis,extra",
    [
        ("prismatic", "1 0 0", ""),
        ("revolute", "0 0 0", ""),
        ("revolute", "1 0 0", '<mimic joint="other"/>'),
        ("revolute", "nan 0 1", ""),
    ],
)
def test_unsupported_or_invalid_models_fail(tmp_path, kind, axis, extra):
    with pytest.raises(ValueError):
        NumpyBackend.from_urdf(
            write_chain(tmp_path, kind=kind, axis=axis, extra=extra), end_link="tool"
        )


def test_subchain_and_joint_inputs():
    robot = NumpyBackend.from_urdf(
        example_urdf(), base_link="Link1", end_link="ee_link"
    )
    assert robot.dof == 5
    assert next(iter(robot.link_transforms(np.zeros(5)))) == "Link1"
    for invalid in ([1], [np.nan] * 5, np.zeros((1, 5))):
        with pytest.raises(ValueError):
            robot.forward(invalid)


def test_truth_is_independent_repeatable_and_moves_geometry():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    home = robot.forward(np.zeros(6))
    truth = robot.perturbed(seed=8)
    np.testing.assert_array_equal(robot.forward(np.zeros(6)), home)
    np.testing.assert_array_equal(
        truth.forward(np.zeros(6)), robot.perturbed(seed=8).forward(np.zeros(6))
    )
    assert np.linalg.norm(home - truth.forward(np.zeros(6))) > 1e-3


def test_link_transform_arrays_are_independent_snapshots():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    configuration = np.linspace(-0.3, 0.4, robot.dof)
    transforms = robot.link_transforms(configuration)
    expected_end = robot.forward(configuration)
    np.testing.assert_array_equal(transforms[robot.end_link], expected_end)
    values = list(transforms.values())
    assert all(
        not np.shares_memory(first, second)
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    )
    values[1][:] = 0
    np.testing.assert_array_equal(transforms[robot.end_link], expected_end)
    np.testing.assert_array_equal(robot.forward(configuration), expected_end)


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(float).max,
    reason="platform longdouble has the same range as float64",
)
def test_finite_extended_precision_inputs_must_remain_finite_in_float64():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    positions = np.zeros(robot.dof, dtype=np.longdouble)
    positions[0] = np.longdouble("1e400")
    assert np.isfinite(positions).all()
    with pytest.raises(ValueError, match="finite"):
        robot.forward(positions)
    with pytest.raises(ValueError, match="finite"):
        robot.link_transforms(positions)
