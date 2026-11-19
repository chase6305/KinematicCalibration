"""Reuse the rank-check spectrum without losing independent result snapshots."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from robotkinecal import CalibrationResult, SerialRobotKineCal


def make_calibrator(position_only):
    angles = np.linspace(-2, 2, 12)
    home = np.eye(4)
    home[:3, 3] = [0.4, 0.2, 0.3]
    axis = np.array([0, 0, 0, 0, 0, 1], dtype=float)
    calibrator = SerialRobotKineCal.from_screw_axes(axis[None], home)
    # Independently specified true revolute axis and home pose; no nominal FK
    # is used as a measurement.
    true_direction = Rotation.from_rotvec([0.004, -0.002, 0]).apply([0, 0, 1])
    point = np.array([0.003, -0.001, 0.0])
    rotations = Rotation.from_rotvec(angles[:, None] * true_direction).as_matrix()
    motion = np.repeat(np.eye(4)[None], len(angles), axis=0)
    motion[:, :3, :3] = rotations
    motion[:, :3, 3] = point - rotations @ point
    true_home = home.copy()
    true_home[:3, :3] = Rotation.from_rotvec([0.05, -0.02, 0.03]).as_matrix()
    true_home[:3, 3] += [0.001, -0.002, 0.003]
    poses = motion @ true_home
    observations = (
        list(poses[:, :3, 3])
        if position_only
        else [SE3(pose, check=False) for pose in poses]
    )
    calibrator.set_observations(angles[:, None], observations)
    return calibrator


@pytest.mark.parametrize("position_only", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("damping", [0, 0.01])
def test_solver_reuses_rank_spectrum_and_preserves_owned_diagnostics(
    monkeypatch, position_only, weighted, damping
):
    calibrator = make_calibrator(position_only)
    factorization = np.linalg.svd
    matrix_rank = np.linalg.matrix_rank
    factored = []
    rank_checks = []

    def tracked_svd(matrix, *args, **kwargs):
        values = factorization(matrix, *args, **kwargs)
        factored.append((matrix.shape, values))
        return values

    def tracked_rank(matrix, *args, **kwargs):
        # NumPy's matrix_rank can call a private SVD alias rather than the
        # public np.linalg.svd entry point, so account for it separately.
        rank_checks.append(matrix.shape)
        return matrix_rank(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    monkeypatch.setattr(np.linalg, "matrix_rank", tracked_rank)
    result = calibrator.solve(
        max_iterations=1,
        damping=damping,
        sample_weights=np.linspace(0.3, 1.0, calibrator.N_OBSERVATIONS)
        if weighted
        else None,
    )
    last = result.iteration_results[-1]

    assert last.matrix_rank == last.N_PARAMS
    assert len(factored) + len(rank_checks) == (2 if weighted else 1)
    assert not rank_checks
    assert all(shape == last.A_all.shape for shape, _ in factored)
    np.testing.assert_array_equal(
        last.singular_values, factorization(last.A_all, compute_uv=False)
    )
    np.testing.assert_array_equal(
        last.weighted_singular_values,
        factorization(last.weighted_A_all, compute_uv=False),
    )
    expected_raw = last.singular_values.copy()
    expected_weighted = last.weighted_singular_values.copy()
    # The solver's transient factorization buffer must not alias either stored
    # spectrum, including when unit weights let the raw spectrum be reused.
    factored[0][1][:] = np.nan
    np.testing.assert_array_equal(last.singular_values, expected_raw)
    np.testing.assert_array_equal(last.weighted_singular_values, expected_weighted)
    assert not np.shares_memory(last.singular_values, last.weighted_singular_values)


@pytest.mark.parametrize("weighted", [False, True])
def test_public_iteration_constructor_still_computes_its_own_diagnostics(
    monkeypatch, weighted
):
    matrix = np.random.default_rng(24).normal(size=(24, 10))
    factorization = np.linalg.svd
    shapes = []

    def tracked_svd(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        return factorization(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    result = CalibrationResult.IterationResult(
        [np.array([0, 0, 0, 0, 0, 1], dtype=float)],
        SE3(),
        matrix,
        np.zeros((10, 1)),
        np.ones((24, 1)),
        [0] * 4,
        [0] * 4,
        row_weights=np.linspace(0.3, 1.0, 24) if weighted else None,
    )

    assert len(shapes) == (2 if weighted else 1)
    np.testing.assert_array_equal(
        result.singular_values, factorization(matrix, compute_uv=False)
    )
    np.testing.assert_array_equal(
        result.weighted_singular_values,
        factorization(result.weighted_A_all, compute_uv=False),
    )
