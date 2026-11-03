"""Per-observation residual multipliers and weighted numerical diagnostics."""

import numpy as np
import pytest
from spatialmath import SE3

from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.simulation import sample_measurements
from robotkinecal import SerialRobotKineCal


def problem(position_only=False):
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=43)
    data = sample_measurements(truth, count=30, seed=42)
    observations = (
        list(data.observed_poses[:, :3, 3])
        if position_only
        else [SE3(pose) for pose in data.observed_poses]
    )
    calibrator = SerialRobotKineCal.from_screw_axes(*nominal.poe_parameters())
    calibrator.set_observations(data.joint_positions, observations)
    return calibrator, data.joint_positions, observations


@pytest.mark.parametrize("position_only", [False, True])
def test_ones_preserve_original_solver_and_raw_result_fields(position_only):
    reference, joints, _ = problem(position_only)
    weighted, _, _ = problem(position_only)
    baseline = reference.solve(max_iterations=30)
    result = weighted.solve(max_iterations=30, sample_weights=np.ones(len(joints)))
    assert result.termination_reason == baseline.termination_reason
    assert result.nb_iterations_executed == baseline.nb_iterations_executed
    for original, current in zip(
        baseline.iteration_results, result.iteration_results, strict=True
    ):
        np.testing.assert_array_equal(current.A_all, original.A_all)
        np.testing.assert_array_equal(current.twist_errors, original.twist_errors)
        np.testing.assert_array_equal(
            current.twist_corrections, original.twist_corrections
        )
        np.testing.assert_array_equal(
            current.weighted_singular_values, current.singular_values
        )
        assert current.weighted_matrix_rank == current.matrix_rank
        assert current.weighted_condition_number == current.condition_number


@pytest.mark.parametrize("position_only", [False, True])
def test_sqrt_multiplicities_recover_the_solution_from_repeated_measurements(
    position_only,
):
    weighted, joints, observations = problem(position_only)
    duplicate, _, _ = problem(position_only)
    counts = np.resize([1, 2, 3], len(joints))
    repeated_joints = np.repeat(joints, counts, axis=0)
    repeated_observations = [
        obs
        for obs, count in zip(observations, counts, strict=True)
        for _ in range(count)
    ]
    duplicate.set_observations(repeated_joints, repeated_observations)
    result = weighted.solve(max_iterations=30, sample_weights=np.sqrt(counts))
    other_result = duplicate.solve(max_iterations=30)
    assert result.has_converged and other_result.has_converged
    held_out = np.random.default_rng(612).uniform(-2, 2, size=(20, weighted.N_JOINTS))
    np.testing.assert_allclose(
        weighted.forward_kinematics_batch(held_out),
        duplicate.forward_kinematics_batch(held_out),
        atol=2e-8,
    )


@pytest.mark.parametrize("position_only", [False, True])
def test_sample_and_global_weights_multiply_all_rows_and_match_diagnostics(
    position_only,
):
    calibrator, joints, _ = problem(position_only)
    weights = np.linspace(0.2, 1.3, len(joints))
    result = calibrator.solve(
        max_iterations=30,
        sample_weights=weights,
        position_weight=2.0,
        orientation_weight=0.3,
    )
    expected = np.repeat(weights, calibrator.OBS_SIZE) * np.tile(
        [2.0] * 3 if position_only else [2.0] * 3 + [0.3] * 3, len(joints)
    )
    for iteration in result.iteration_results:
        np.testing.assert_allclose(iteration.row_weights, expected)
        np.testing.assert_allclose(
            iteration.weighted_A_all, expected[:, None] * iteration.A_all
        )
        np.testing.assert_allclose(
            iteration.weighted_twist_errors, expected[:, None] * iteration.twist_errors
        )
        np.testing.assert_allclose(
            iteration.weighted_singular_values,
            np.linalg.svd(iteration.weighted_A_all, compute_uv=False),
        )
        assert iteration.weighted_matrix_rank == np.linalg.matrix_rank(
            iteration.weighted_A_all
        )
        np.testing.assert_allclose(
            iteration.weighted_condition_number,
            np.linalg.cond(iteration.weighted_A_all),
        )
        assert (
            iteration.post_update_objective_norm <= iteration.pre_update_objective_norm
        )
        statistics = iteration.get_statistics()
        assert statistics["weighted_matrix_rank"] == iteration.weighted_matrix_rank
        assert (
            statistics["uncertainty_method"]
            == "local_linear_weighted_residual_heuristic"
        )


@pytest.mark.parametrize("position_only", [False, True])
def test_common_sample_weight_scale_preserves_undamped_fit(position_only):
    reference, joints, _ = problem(position_only)
    weights = np.linspace(0.3, 1.0, len(joints))
    baseline = reference.solve(max_iterations=30, sample_weights=weights)
    expected = reference.forward_kinematics_batch(joints)
    assert baseline.has_converged
    for scale in [1e-8, 1e8]:
        calibrator, _, _ = problem(position_only)
        result = calibrator.solve(max_iterations=30, sample_weights=scale * weights)
        assert result.has_converged
        np.testing.assert_allclose(
            calibrator.forward_kinematics_batch(joints), expected, atol=2e-8
        )


def test_weighted_uncertainty_matches_independent_information_matrix():
    calibrator, joints, _ = problem()
    result = calibrator.solve(
        max_iterations=1, sample_weights=np.linspace(0.3, 1.0, len(joints))
    )
    iteration = result.iteration_results[-1]
    matrix, observed = iteration.weighted_A_all, iteration.weighted_twist_errors
    residual = observed - matrix @ iteration.twist_corrections
    variance_scale = float(np.sum(residual**2)) / (matrix.shape[0] - matrix.shape[1])
    variances = variance_scale * np.diag(np.linalg.inv(matrix.T @ matrix))
    expected = np.linalg.norm(variances[:-6].reshape(-1, 4), axis=1)
    np.testing.assert_allclose(
        iteration.get_statistics()["joints_uncertainty"],
        expected,
        rtol=1e-10,
        atol=1e-20,
    )


def test_tiny_positive_weights_expose_effective_rank_loss_without_changing_raw_rank():
    calibrator, joints, _ = problem()
    weights = np.full(len(joints), 1e-20)
    weights[0] = 1.0
    result = calibrator.solve(max_iterations=1, sample_weights=weights)
    final = result.iteration_results[-1]
    assert final.matrix_rank == final.N_PARAMS
    assert final.weighted_matrix_rank < final.N_PARAMS
    assert final.weighted_condition_number > final.condition_number
    assert np.all(np.isnan(final.get_statistics()["joints_uncertainty"]))


def test_damping_uses_the_combined_weighted_problem():
    calibrator, joints, _ = problem()
    result = calibrator.solve(
        max_iterations=1,
        sample_weights=np.linspace(0.2, 2.0, len(joints)),
        position_weight=3.0,
        orientation_weight=0.5,
        damping=0.1,
    )
    final = result.iteration_results[-1]
    expected = np.linalg.lstsq(
        np.vstack([final.weighted_A_all, np.sqrt(0.1) * np.eye(final.N_PARAMS)]),
        np.vstack([final.weighted_twist_errors, np.zeros((final.N_PARAMS, 1))]),
        rcond=1e-12,
    )[0]
    np.testing.assert_allclose(final.twist_corrections, expected, atol=1e-14)


@pytest.mark.parametrize(
    "weights",
    [
        1.0,
        np.ones(29),
        np.ones((30, 1)),
        np.ones(30, dtype=bool),
        np.full(30, "1"),
        np.ones(30, dtype=complex),
        np.r_[0.0, np.ones(29)],
        np.r_[-1.0, np.ones(29)],
        np.r_[np.nan, np.ones(29)],
        np.r_[np.inf, np.ones(29)],
    ],
)
def test_invalid_sample_weights_are_rejected(weights):
    calibrator, _, _ = problem()
    with pytest.raises(ValueError, match="sample_weights"):
        calibrator.solve(sample_weights=weights)


def test_overflowed_combined_weights_are_rejected():
    calibrator, joints, _ = problem()
    with pytest.raises(ValueError, match="Combined residual weights"):
        calibrator.solve(
            sample_weights=np.full(len(joints), 1e300), position_weight=1e100
        )
