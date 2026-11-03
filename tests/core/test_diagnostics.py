"""Regressor diagnostics and uncertainty against known linear problems."""

import numpy as np
import pytest
from spatialmath import SE3

from robotkinecal import CalibrationResult


def make_iteration(matrix, corrections=None, errors=None):
    """Build a one-joint pose result without running the nonlinear solver."""
    sample_count = matrix.shape[0] // 6
    if corrections is None:
        corrections = np.zeros((10, 1))
    if errors is None:
        errors = np.ones((matrix.shape[0], 1))
    return CalibrationResult.IterationResult(
        [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
        SE3(),
        matrix,
        corrections,
        errors,
        [0.0] * sample_count,
        [0.0] * sample_count,
    )


def test_ill_conditioned_uncertainty_matches_known_covariance():
    # The covariance follows directly from the chosen orthogonal factors and
    # singular values. Forming A.T @ A would lose the weakest directions here.
    rng = np.random.default_rng(20261103)
    left, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    right, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    singular_values = np.geomspace(1.0, 1e-10, 10)
    matrix = (left[:, :10] * singular_values) @ right.T
    corrections = rng.normal(scale=0.01, size=(10, 1))
    errors = matrix @ corrections + 2.0 * left[:, 10:11]
    iteration = make_iteration(matrix, corrections, errors)

    # The residual is orthogonal to the design, with squared norm four.
    residual_variance = 4.0 / (24 - 10)
    parameter_variances = residual_variance * (right**2 @ singular_values**-2)
    expected = np.linalg.norm(parameter_variances[:4])
    actual = iteration.get_statistics()["joints_uncertainty"]

    assert actual.shape == (1,)
    np.testing.assert_allclose(actual, [expected], rtol=2e-6)
    assert iteration.matrix_rank == 10
    np.testing.assert_allclose(iteration.condition_number, 1e10, rtol=2e-6)


@pytest.mark.parametrize("missing_direction", [False, True])
def test_numerically_unidentifiable_uncertainty_is_unavailable(missing_direction):
    matrix = np.vstack((np.eye(10), np.zeros((8, 10))))
    # A nonzero but sub-tolerance direction is as unidentifiable numerically
    # as an exact null space, even though its algebraic rank is ten.
    matrix[9, 9] = 0.0 if missing_direction else np.finfo(float).eps
    iteration = make_iteration(matrix)
    assert iteration.matrix_rank == 9
    assert np.isnan(iteration.get_statistics()["joints_uncertainty"]).all()


def test_uncertainty_requires_residual_degrees_of_freedom():
    iteration = make_iteration(np.eye(6, 10))
    assert np.isnan(iteration.get_statistics()["joints_uncertainty"]).all()


def test_zero_regressor_has_zero_rank_and_infinite_condition():
    iteration = make_iteration(np.zeros((18, 10)))
    assert iteration.matrix_rank == 0
    assert np.isinf(iteration.condition_number)
    np.testing.assert_array_equal(iteration.singular_values, np.zeros(10))
    assert np.isnan(iteration.get_statistics()["joints_uncertainty"]).all()
