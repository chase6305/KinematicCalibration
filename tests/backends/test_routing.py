"""Visit order tests use independent path-length and permutation checks."""

from itertools import permutations
from math import dist

import numpy as np
import pytest

from kinematic_calibration.routing import plan_visit_order


def path_length(positions, order, start=None, closed=False):
    path = positions[order]
    if start is not None:
        path = np.vstack([start, path])
        if closed:
            path = np.vstack([path, start])
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def test_orders_line_visits_and_preserves_first_pose():
    positions = np.array([[0.0], [4.0], [1.0], [3.0], [2.0]])
    original = positions.copy()
    route = plan_visit_order(positions)
    np.testing.assert_array_equal(route.indices, [0, 2, 4, 3, 1])
    np.testing.assert_array_equal(route.segment_distances_rad, [1, 1, 1, 1])
    assert route.total_distance_rad == 4.0
    assert route.initial_distance_rad == path_length(positions, np.arange(5))
    np.testing.assert_array_equal(positions, original)


def test_refinement_reaches_independent_short_fixture_optimum():
    positions = np.array([[2, 5], [5, 0], [0, 2], [2, 3], [1, 4], [3, 1]])
    route = plan_visit_order(positions)
    optimum = min(
        path_length(positions, [0, *tail])
        for tail in permutations(range(1, len(positions)))
    )
    # This fixture's nearest-neighbor route leaves a removable detour.
    assert route.total_distance_rad < path_length(positions, [0, 4, 3, 2, 5, 1])
    assert route.total_distance_rad == pytest.approx(optimum)


@pytest.mark.parametrize("closed", [False, True])
def test_explicit_start_and_return_match_short_fixture_optimum(closed):
    positions = np.array([[2.0, 0], [0.0, 1], [2.0, 1], [0.0, 0]])
    start = np.array([-1.0, 0])
    route = plan_visit_order(
        positions, start_configuration=start, return_to_start=closed
    )
    optimum = min(
        path_length(positions, list(order), start, closed)
        for order in permutations(range(len(positions)))
    )
    assert route.total_distance_rad == pytest.approx(optimum)
    assert route.total_distance_rad == pytest.approx(
        path_length(positions, route.indices, start, closed)
    )
    assert len(route.segment_distances_rad) == len(positions) + int(closed)
    assert route.initial_distance_rad == pytest.approx(
        path_length(positions, np.arange(len(positions)), start, closed)
    )


@pytest.mark.parametrize("closed", [False, True])
def test_visits_every_row_once_and_never_lengthens_input(closed):
    positions = np.random.default_rng(123).uniform(-3, 3, (40, 6))
    start = np.zeros(6)
    original = positions.copy()
    route = plan_visit_order(
        positions, start_configuration=start, return_to_start=closed
    )
    repeated = plan_visit_order(
        positions, start_configuration=start, return_to_start=closed
    )
    np.testing.assert_array_equal(np.sort(route.indices), np.arange(len(positions)))
    np.testing.assert_array_equal(route.indices, repeated.indices)
    np.testing.assert_array_equal(positions, original)
    assert route.total_distance_rad <= route.initial_distance_rad
    assert route.total_distance_rad == pytest.approx(
        path_length(positions, route.indices, start, closed)
    )


def test_duplicate_configurations_keep_distinct_row_identities():
    positions = np.array([[0.0], [2.0], [0.0], [1.0], [1.0]])
    route = plan_visit_order(positions)
    np.testing.assert_array_equal(np.sort(route.indices), np.arange(5))
    assert route.total_distance_rad == 2.0
    assert np.count_nonzero(route.segment_distances_rad == 0) == 2


@pytest.mark.parametrize("closed", [False, True])
def test_single_pose_includes_start_and_optional_return(closed):
    route = plan_visit_order(
        [[2.0, 0]], start_configuration=[0.0, 0], return_to_start=closed
    )
    np.testing.assert_array_equal(route.indices, [0])
    np.testing.assert_array_equal(
        route.segment_distances_rad, [2.0, 2.0] if closed else [2.0]
    )


def test_single_pose_without_start_has_no_travel():
    route = plan_visit_order([[2.0, 0]])
    assert route.total_distance_rad == route.initial_distance_rad == 0.0
    assert route.segment_distances_rad.shape == (0,)


def test_distances_do_not_wrap_across_joint_limits():
    route = plan_visit_order([[-3.0], [3.0]])
    assert route.total_distance_rad == 6.0


@pytest.mark.parametrize(
    "positions",
    [
        [],
        np.empty((0, 3)),
        np.empty((2, 0)),
        [1, 2],
        [[np.nan]],
        [[np.inf]],
        [[True]],
        [[1j]],
        [["1"]],
    ],
)
def test_invalid_configurations_are_rejected(positions):
    with pytest.raises(ValueError, match="joint_positions"):
        plan_visit_order(positions)


@pytest.mark.parametrize("start", [[0], [[0, 0]], [True, False], [np.inf, 0], [0j, 0]])
def test_invalid_start_is_rejected(start):
    with pytest.raises(ValueError, match="start_configuration"):
        plan_visit_order([[0, 0], [1, 1]], start_configuration=start)


def test_return_requires_a_start_and_boolean_flag():
    with pytest.raises(ValueError, match="requires start_configuration"):
        plan_visit_order([[0, 0]], return_to_start=True)
    with pytest.raises(ValueError, match="boolean"):
        plan_visit_order([[0, 0]], return_to_start="yes")


def test_unrepresentable_distances_are_rejected_without_runtime_warning():
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="distances exceed"):
            plan_visit_order([[-1e308], [1e308]])


def test_distance_norm_avoids_overflowing_intermediate_squares():
    route = plan_visit_order([[0, 0], [1e200, 1e200]])
    assert route.total_distance_rad == pytest.approx(np.sqrt(2) * 1e200)


def test_unrepresentable_total_distance_is_rejected_without_runtime_warning():
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="total joint-space distance exceeds"):
            plan_visit_order([[-4e307], [4e307], [-4e307], [4e307]])


@pytest.mark.parametrize("dof", [1, 6, 7])
@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_pairwise_distances_match_independent_stable_norm(dof, scale):
    from kinematic_calibration.routing import _distance_matrix

    positions = np.random.default_rng(42).uniform(-1, 1, (12, dof)) * scale
    positions[-1] = positions[0]
    original = positions.copy()
    distances = _distance_matrix(positions)
    # math.dist scales intermediate values, independently of NumPy's hypot reduction.
    expected = np.array([[dist(a, b) for b in positions] for a in positions])
    np.testing.assert_allclose(distances, expected, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), np.zeros(len(positions)))
    np.testing.assert_array_equal(positions, original)
