"""Synthetic observations follow an explicit configuration sequence."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import ObservationFrame, camera_target_to_base_ee
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.simulation import MeasurementNoise, measure_configurations


@pytest.fixture
def truth():
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    return nominal.perturbed(seed=41)


def test_explicit_configuration_order_and_identity_survive_measurement(truth):
    joints = np.array([[0.2] * 6, [-0.4] * 6, [0.1] * 6, [0.2] * 6])
    original = joints.copy()
    ids = np.array([91, 12, 4, 25])
    dataset = measure_configurations(
        truth, joints, noise=MeasurementNoise(0, 0), sample_indices=ids
    )

    assert dataset.observation_frame is ObservationFrame.BASE_T_EE
    assert dataset.joint_names == truth.joint_names
    np.testing.assert_array_equal(dataset.joint_positions, original)
    np.testing.assert_array_equal(dataset.sample_indices, ids)
    np.testing.assert_allclose(
        dataset.observed_poses, [truth.forward(row) for row in original], atol=1e-15
    )
    np.testing.assert_array_equal(joints, original)
    joints[:] = 0
    np.testing.assert_array_equal(dataset.joint_positions, original)


def test_explicit_camera_measurements_use_both_fixed_transforms(truth):
    base_t_camera, ee_t_target = np.eye(4), np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.4, -0.1, 0.2]).as_matrix()
    base_t_camera[:3, 3] = [0.7, -0.3, 1.2]
    ee_t_target[:3, :3] = Rotation.from_rotvec([-0.1, 0.2, 0.3]).as_matrix()
    ee_t_target[:3, 3] = [0.03, -0.02, 0.1]
    joints = np.array([[0.2] * 6, [-0.3] * 6])
    dataset = measure_configurations(
        truth,
        joints,
        noise=MeasurementNoise(0, 0),
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
    )

    assert dataset.observation_frame is ObservationFrame.CAMERA_T_TARGET
    np.testing.assert_allclose(
        camera_target_to_base_ee(dataset.observed_poses, base_t_camera, ee_t_target),
        [truth.forward(row) for row in joints],
        atol=2e-15,
    )


def test_noise_is_reproducible_without_reordering_or_changing_configurations(truth):
    joints = np.full((4, 6), 0.2)
    first = measure_configurations(truth, joints, seed=14)
    repeated = measure_configurations(truth, joints, seed=14)
    different = measure_configurations(truth, joints, seed=15)
    np.testing.assert_array_equal(first.joint_positions, joints)
    np.testing.assert_array_equal(first.observed_poses, repeated.observed_poses)
    assert not np.array_equal(first.observed_poses, different.observed_poses)
    assert not np.array_equal(first.observed_poses[0], first.observed_poses[1])


@pytest.mark.parametrize(
    "joints",
    [
        np.zeros(6),
        np.empty((0, 6)),
        np.zeros((2, 5)),
        np.zeros((2, 6), bool),
        np.zeros((2, 6), complex),
        np.full((2, 6), "0"),
        np.full((2, 6), np.nan),
        np.full((2, 6), np.inf),
        np.full((2, 6), 100.0),
    ],
)
def test_invalid_configurations_fail_before_sensor_queries(truth, joints, monkeypatch):
    def unexpected_measurement(*_):
        raise AssertionError("invalid plan queried the sensor")

    monkeypatch.setattr(type(truth), "forward", unexpected_measurement)
    with pytest.raises(ValueError, match="joint_positions"):
        measure_configurations(truth, joints)
