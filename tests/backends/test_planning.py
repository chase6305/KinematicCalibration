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
