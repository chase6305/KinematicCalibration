"""Observation setup owns its copies and retains scalar rejection behavior."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from robotkinecal import SerialRobotKineCal, calibration


def make_model():
    axes = np.array(
        [[0, 0, 0, 0, 0, 1], [0, 0, 0.2, 0, 1, 0], [0, 0.2, 0, 1, 0, 0]],
        dtype=float,
    )
    return SerialRobotKineCal.from_screw_axes(axes, np.eye(4))


def values(count):
    rng = np.random.default_rng(20261119)
    joints = rng.uniform(-1, 1, (count, 3))
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.4, (count, 3))).as_matrix()
    poses[:, :3, 3] = rng.normal(0, 0.2, (count, 3))
    return joints, poses


def view(array, layout):
    if layout == "F":
        return np.asfortranarray(array)
    if layout == "strided":
        storage = np.empty((2 * len(array), *array.shape[1:]))
        storage[::2] = array
        return storage[::2]
    if layout == "negative":
        return array[::-1].copy()[::-1]
    return array.copy()


@pytest.mark.parametrize("count", [16, 17, 1025])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
@pytest.mark.parametrize("container", [list, tuple])
def test_ordinary_poses_are_checked_in_batches_and_keep_independent_snapshots(
    monkeypatch, count, layout, container
):
    model = make_model()
    joints, poses = [view(array, layout) for array in values(count)]
    expected_joints, expected_poses = joints.copy(), poses.copy()
    joints.setflags(write=False)
    poses.setflags(write=False)
    observations = container(SE3(pose, check=False) for pose in poses)
    scalar_check = calibration._rigid_pose
    checked = []

    def tracked_check(matrix):
        checked.append(matrix)
        return scalar_check(matrix)

    monkeypatch.setattr(calibration, "_rigid_pose", tracked_check)
    with np.errstate(divide="warn", over="warn", under="ignore", invalid="warn"):
        model.set_observations(joints, observations)

    assert not checked
    np.testing.assert_array_equal(model._joint_positions_array, expected_joints)
    np.testing.assert_array_equal(model._observation_array, expected_poses)
    assert model._observation_array.flags.c_contiguous
    assert not joints.flags.writeable and not poses.flags.writeable
    for index, observation in enumerate(observations):
        np.testing.assert_array_equal(
            model._observations[index].A, expected_poses[index]
        )
        assert not np.shares_memory(model._observations[index].A, observation.A)
        assert not np.shares_memory(model._observation_array, observation.A)
        assert not np.shares_memory(
            model._observation_array, model._observations[index].A
        )
    model._observations[0].A[0, 3] += 1
    np.testing.assert_array_equal(model._observation_array, expected_poses)
    np.testing.assert_array_equal(poses, expected_poses)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
def test_numeric_identity_inputs_keep_float64_storage_and_owned_copies(dtype):
    model = make_model()
    joints = np.zeros((17, 3), dtype=dtype)
    joints[:, 0] = np.arange(17)
    poses = np.repeat(np.eye(4, dtype=dtype)[None], 17, axis=0)
    observations = [SE3(pose, check=False) for pose in poses]
    model.set_observations(joints, observations)

    assert model._joint_positions_array.dtype == np.float64
    assert model._observation_array.dtype == np.float64
    np.testing.assert_array_equal(model._observation_array, poses)
    joints[:] = 0
    poses[:, 0, 3] = 9
    np.testing.assert_array_equal(model._joint_positions_array[:, 0], np.arange(17))
    assert np.all(model._observation_array[:, 0, 3] == 0)


def corrupt(observations, index, fault):
    if fault == "rotation":
        matrix = np.eye(4)
        matrix[0, 0] = 2
    elif fault == "nonfinite":
        matrix = np.eye(4)
        matrix[0, 3] = np.nan
    elif fault == "shape":
        matrix = np.eye(3)
    else:
        matrix = np.eye(4, dtype=complex)
        matrix[0, 3] = 1j
    # Represent corrupt external SE3 instances without relying on a particular
    # SpatialMath constructor's dtype conversion or shape checks.
    observations[index].data[0] = matrix


@pytest.mark.parametrize(
    "first,second,message",
    [
        ("rotation", "nonfinite", "Pose observation 2 is not a valid SE3 pose"),
        ("rotation", "shape", "Pose observation 2 is not a valid SE3 pose"),
        (
            "nonfinite",
            "rotation",
            "Pose observation 2 must be a finite single SE3 pose",
        ),
        ("shape", "dtype", "Pose observation 2 must be a finite single SE3 pose"),
        ("dtype", "shape", "Pose observation 2 must contain real numeric values"),
    ],
)
def test_invalid_batches_report_the_first_scalar_error(first, second, message):
    model = make_model()
    joints, poses = values(17)
    observations = [SE3(pose, check=False) for pose in poses]
    corrupt(observations, 2, first)
    corrupt(observations, 16, second)
    with pytest.raises(ValueError, match=message) as error:
        model.set_observations(joints, observations)
    assert (error.value.__cause__ is not None) == (first == "rotation")


def test_joint_and_mixed_type_errors_keep_their_original_precedence():
    model = make_model()
    joints, poses = values(17)
    observations = [SE3(pose, check=False) for pose in poses]
    corrupt(observations, 0, "rotation")
    observations[-1] = np.zeros(3)
    with pytest.raises(ValueError, match="All observations must have the same type"):
        model.set_observations(joints, observations)
    joints[2, 1] = np.nan
    with pytest.raises(ValueError, match="Joint configuration 2 contains non-finite"):
        model.set_observations(joints, observations)


@pytest.mark.parametrize("error_mode", ["over", "under", "invalid", "divide"])
def test_special_floating_error_modes_keep_scalar_validation(monkeypatch, error_mode):
    model = make_model()
    joints, poses = values(17)
    observations = [SE3(pose, check=False) for pose in poses]
    scalar_check = calibration._rigid_pose
    checked = []

    def tracked_check(matrix):
        checked.append(matrix)
        return scalar_check(matrix)

    monkeypatch.setattr(calibration, "_rigid_pose", tracked_check)
    with np.errstate(**{error_mode: "raise"}):
        model.set_observations(joints, observations)

    assert len(checked) == len(poses)
    np.testing.assert_array_equal(model._observation_array, poses)


def test_reloading_pose_and_position_observations_rebuilds_the_right_arrays():
    model = make_model()
    joints, poses = values(17)
    points = poses[:, :3, 3]
    for measured in (
        points,
        [SE3(pose, check=False) for pose in poses],
        points,
    ):
        model.set_observations(joints, measured)
        point_mode = isinstance(measured, np.ndarray)
        assert model.position_only == point_mode
        assert model.OBS_SIZE == (3 if point_mode else 6)
        assert model.N_PARAMS == 4 * 3 + model.OBS_SIZE
        assert model._B_Matrix is None
        np.testing.assert_array_equal(
            model._observation_array, points if point_mode else poses
        )
        model.B_Matrix()


@pytest.mark.parametrize(
    "offset,accepted", [(2**-44, True), (2**-42, True), (2**-41, True), (2**-40, False)]
)
def test_near_rigid_rotations_keep_the_existing_scalar_tolerance(offset, accepted):
    model = make_model()
    poses = np.repeat(np.eye(4)[None], 17, axis=0)
    poses[3, 0, 0] += offset
    observations = [SE3(pose, check=False) for pose in poses]
    joints = np.zeros((17, 3))
    if accepted:
        model.set_observations(joints, observations)
        np.testing.assert_array_equal(model._observation_array, poses)
    else:
        with pytest.raises(
            ValueError, match="Pose observation 3 is not a valid SE3 pose"
        ):
            model.set_observations(joints, observations)


@pytest.mark.parametrize("offset,accepted", [(2**-40, True), (2**-39, False)])
def test_near_homogeneous_rows_are_retained_without_normalizing(offset, accepted):
    model = make_model()
    poses = np.repeat(np.eye(4)[None], 17, axis=0)
    poses[3, 3, 0] = offset
    observations = [SE3(pose, check=False) for pose in poses]
    if accepted:
        model.set_observations(np.zeros((17, 3)), observations)
        np.testing.assert_array_equal(model._observation_array, poses)
    else:
        with pytest.raises(
            ValueError, match="Pose observation 3 is not a valid SE3 pose"
        ):
            model.set_observations(np.zeros((17, 3)), observations)


def test_finite_rotations_with_overflowing_products_are_rejected():
    model = make_model()
    poses = np.repeat(np.eye(4)[None], 17, axis=0)
    poses[3, :3, :3] = 1e308 * np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1]])
    observations = [SE3(pose, check=False) for pose in poses]
    with pytest.raises(ValueError, match="Pose observation 3 is not a valid SE3 pose"):
        model.set_observations(np.zeros((17, 3)), observations)


def test_the_last_partial_batch_cannot_hide_an_invalid_pose():
    model = make_model()
    joints, poses = values(1025)
    poses[-1, 0, 3] = np.nan
    observations = [SE3(pose, check=False) for pose in poses]
    with pytest.raises(
        ValueError, match="Pose observation 1024 must be a finite single SE3 pose"
    ):
        model.set_observations(joints, observations)
