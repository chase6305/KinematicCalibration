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


@pytest.mark.parametrize("candidate_count", [1023, 1024, 1025, 2053])
def test_large_selection_matches_direct_information_scores(candidate_count):
    """Chunk boundaries must not change the D-optimal set or its diagnostics."""
    from spatialmath import SE3

    from robotkinecal import SerialRobotKineCal

    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.random.default_rng(61119).uniform(-2, 2, (candidate_count, 6))
    original_candidates = candidates.copy()
    selected = select_informative_poses(
        robot, candidates, count=8, characteristic_length_m=0.4
    )

    # Independent scalar regressors and full parameter-space determinants,
    # without the 6x6 determinant identity used by the production scorer.
    calibrator = SerialRobotKineCal.from_screw_axes(*robot.poe_parameters())
    calibrator.set_observations(candidates[:1], [SE3(robot.forward(candidates[0]))])
    blocks = np.asarray([calibrator.A_Matrix(row) for row in candidates])
    blocks[:, :3] /= 0.4
    information = 1e-6 * np.eye(blocks.shape[-1])
    available = np.arange(candidate_count)
    expected = []
    for _ in range(8):
        proposed = blocks[available]
        signs, scores = np.linalg.slogdet(
            information + proposed.transpose(0, 2, 1) @ proposed
        )
        assert np.all(signs > 0)
        chosen = int(available[np.argmax(scores)])
        expected.append(chosen)
        available = available[available != chosen]
        information += blocks[chosen].T @ blocks[chosen]
    np.testing.assert_array_equal(selected.candidate_indices, expected)
    selected_matrix = blocks[expected].reshape(-1, blocks.shape[-1])
    np.testing.assert_allclose(
        selected.singular_values,
        np.linalg.svd(selected_matrix, compute_uv=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert selected.rank == np.linalg.matrix_rank(selected_matrix)
    np.testing.assert_allclose(
        selected.condition_number, np.linalg.cond(selected_matrix), rtol=1e-12
    )
    np.testing.assert_array_equal(selected.joint_positions, candidates[expected])
    np.testing.assert_array_equal(candidates, original_candidates)


def test_scoring_bounds_batch_size_and_preserves_ties_across_boundaries(monkeypatch):
    from kinematic_calibration import planning

    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.zeros((2053, 6))
    blocks = np.zeros((len(candidates), 6, 30))
    blocks[:, :, :6] = np.eye(6)
    # The winners span multiple chunks; equal scores retain original row order.
    blocks[[1023, 1024, 2052]] *= 2
    original_blocks = blocks.copy()
    blocks.setflags(write=False)
    monkeypatch.setattr(planning, "_regressor_blocks", lambda *args: blocks)
    slogdet = np.linalg.slogdet
    batch_sizes = []

    def bounded_slogdet(matrices):
        batch_sizes.append(len(matrices))
        return slogdet(matrices)

    monkeypatch.setattr(np.linalg, "slogdet", bounded_slogdet)
    selected = select_informative_poses(robot, candidates, count=4)
    np.testing.assert_array_equal(selected.candidate_indices, [1023, 1024, 2052, 0])
    assert max(batch_sizes) <= 1024
    assert selected.rank == 6
    assert selected.condition_number is None
    np.testing.assert_array_equal(blocks, original_blocks)


@pytest.mark.parametrize("invalid_row", [0, 1024, 2052])
def test_singular_scores_in_any_chunk_reject_the_whole_selection(
    monkeypatch, invalid_row
):
    from kinematic_calibration import planning

    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.zeros((2053, 6))
    blocks = np.zeros((len(candidates), 6, 30))
    blocks[invalid_row, 0, 0] = np.inf
    monkeypatch.setattr(planning, "_regressor_blocks", lambda *args: blocks)
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="information score.*singular"):
            select_informative_poses(robot, candidates, count=1)


def test_planning_bounds_regressor_construction_and_score_batches(monkeypatch):
    from robotkinecal import SerialRobotKineCal

    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    candidates = np.random.default_rng(2053).uniform(-2, 2, (2053, robot.dof))
    original_regressor_blocks = SerialRobotKineCal.regressor_blocks
    original_slogdet = np.linalg.slogdet
    regressor_sizes, score_sizes = [], []

    def tracked_regressors(self, configurations):
        regressor_sizes.append(len(configurations))
        return original_regressor_blocks(self, configurations)

    def tracked_scores(matrices):
        score_sizes.append(len(matrices))
        return original_slogdet(matrices)

    monkeypatch.setattr(SerialRobotKineCal, "regressor_blocks", tracked_regressors)
    monkeypatch.setattr(np.linalg, "slogdet", tracked_scores)
    result = select_informative_poses(robot, candidates, count=8)
    assert max(regressor_sizes) <= 1024
    assert sum(regressor_sizes) == len(candidates)
    assert max(score_sizes) <= 1024
    assert result.rank == result.parameter_count == 30
