import numpy as np
import pytest

from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.planning import select_informative_poses


def test_pose_selection_escapes_repeated_unobservable_poses():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.vstack(
        [
            np.zeros((20, 6)),
            np.random.default_rng(7).uniform(-2, 2, (40, 6)),
        ]
    )
    baseline = select_informative_poses(robot, candidates[:20], count=10)
    selected = select_informative_poses(robot, candidates, count=10)
    assert baseline.rank == 6
    assert baseline.condition_number is None
    assert selected.rank == selected.parameter_count == 30
    assert len(np.unique(selected.candidate_indices)) == 10
    assert selected.condition_number < 100
    np.testing.assert_array_equal(
        selected.joint_positions, candidates[selected.candidate_indices]
    )


def test_selection_rejects_out_of_limits_and_invalid_candidates():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    for candidates in [
        np.ones((5, 6)) * 10,
        np.ones((5, 6)) * np.nan,
        np.ones((5, 6), dtype=complex),
        np.ones((5, 6), dtype=bool),
    ]:
        with pytest.raises(ValueError):
            select_informative_poses(robot, candidates, count=3)


def test_selection_matches_scalar_regressors_and_direct_information_scores():
    from spatialmath import SE3

    from robotkinecal import SerialRobotKineCal

    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.random.default_rng(103).uniform(-2, 2, (30, robot.dof))
    original_candidates = candidates.copy()
    characteristic_length = 0.4
    regularization = 1e-6
    selected = select_informative_poses(
        robot,
        candidates,
        count=8,
        characteristic_length_m=characteristic_length,
        regularization=regularization,
    )

    calibrator = SerialRobotKineCal.from_screw_axes(*robot.poe_parameters())
    calibrator.set_observations(candidates[:1], [SE3(robot.forward(candidates[0]))])
    blocks = np.asarray([calibrator.A_Matrix(row) for row in candidates])
    blocks[:, :3] /= characteristic_length
    parameter_count = blocks.shape[-1]
    information = regularization * np.eye(parameter_count)
    available = list(range(len(candidates)))
    reference_indices = []
    for _ in range(8):
        scores = [
            np.linalg.slogdet(information + blocks[index].T @ blocks[index])[1]
            for index in available
        ]
        chosen = available.pop(int(np.argmax(scores)))
        reference_indices.append(chosen)
        information += blocks[chosen].T @ blocks[chosen]

    np.testing.assert_array_equal(selected.candidate_indices, reference_indices)
    selected_matrix = blocks[reference_indices].reshape(-1, parameter_count)
    np.testing.assert_allclose(
        selected.singular_values,
        np.linalg.svd(selected_matrix, compute_uv=False),
        atol=1e-12,
    )
    assert selected.rank == np.linalg.matrix_rank(selected_matrix)
    np.testing.assert_allclose(
        selected.condition_number, np.linalg.cond(selected_matrix), rtol=1e-12
    )
    np.testing.assert_array_equal(candidates, original_candidates)


@pytest.mark.parametrize("count", [0, 6, 1.5, True, np.bool_(True), [2]])
def test_selection_rejects_invalid_count(count):
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    with pytest.raises(ValueError, match="count"):
        select_informative_poses(robot, np.zeros((5, robot.dof)), count=count)


@pytest.mark.parametrize("name", ["characteristic_length_m", "regularization"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, 1j, "1", [1]])
def test_selection_rejects_invalid_scalar_controls(name, value):
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    with pytest.raises(ValueError, match=name):
        select_informative_poses(
            robot, np.zeros((5, robot.dof)), count=2, **{name: value}
        )


def test_selection_keeps_repeated_candidate_rows_deterministically():
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.zeros((5, robot.dof))
    selected = select_informative_poses(robot, candidates, count=np.int64(5))
    np.testing.assert_array_equal(selected.candidate_indices, np.arange(5))
    np.testing.assert_array_equal(selected.joint_positions, candidates)
    assert not np.shares_memory(selected.joint_positions, candidates)


@pytest.mark.parametrize("dtype", [bool, complex, str])
def test_selection_rejects_nonreal_backend_limits(dtype):
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    robot.joint_limits = robot.joint_limits.astype(dtype)
    with pytest.raises(ValueError, match="joint limits"):
        select_informative_poses(robot, np.zeros((5, robot.dof)), count=2)
