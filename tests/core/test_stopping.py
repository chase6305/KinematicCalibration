"""Noisy numerical stalls must remain distinct from substantive search failures."""

import numpy as np
import pytest
from spatialmath import SE3

from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.simulation import sample_measurements
from robotkinecal import SerialRobotKineCal


def noisy_calibrator(position_only=False):
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=43)
    data = sample_measurements(truth, count=30, seed=42)
    calibrator = SerialRobotKineCal.from_screw_axes(*nominal.poe_parameters())
    observations = (
        data.observed_poses[:, :3, 3]
        if position_only
        else [SE3(pose) for pose in data.observed_poses]
    )
    calibrator.set_observations(data.joint_positions, observations)
    return calibrator, data.joint_positions


def test_demo30_noisy_fixture_does_not_report_substantive_search_failure():
    # This exact seed/configuration failed in the stage-1 Python 3.12 wheel.
    calibrator, _ = noisy_calibrator()
    result = calibrator.solve(max_iterations=30)
    assert result.has_converged
    assert result.termination_reason in {"converged", "small_update"}
    assert result.iteration_results[-1].post_update_objective_norm > 1e-4


def test_roundoff_rejection_near_noisy_fixed_point_has_explicit_auditable_reason(
    monkeypatch,
):
    calibrator, _ = noisy_calibrator()
    state = {}
    linearize = calibrator._linearization
    update = calibrator.update_twist_definitions
    residual = calibrator._twist_error_vector

    def tracked_linearization():
        products = linearize()
        state["pre_norm"] = np.linalg.norm(products[0])
        return products

    def tracked_update(correction, step_size=1.0):
        state["proposed_update"] = np.linalg.norm(correction)
        return update(correction, step_size)

    def rounded_trial_residual():
        vector = residual()
        if state.get("proposed_update", np.inf) < 1e-10:
            # Model a repeatable evaluation-roundoff sign at an otherwise
            # negligible update, independent of platform/BLAS rounding.
            target_norm = state["pre_norm"] * (1 + 128 * np.finfo(float).eps)
            vector = vector * max(1.0, target_norm / np.linalg.norm(vector))
        return vector

    monkeypatch.setattr(calibrator, "_linearization", tracked_linearization)
    monkeypatch.setattr(calibrator, "update_twist_definitions", tracked_update)
    monkeypatch.setattr(calibrator, "_twist_error_vector", rounded_trial_residual)
    result = calibrator.solve(max_iterations=30)
    final = result.iteration_results[-1]
    assert result.has_converged
    assert result.termination_reason == "small_update"
    assert result.convergence_criterion == "small_update_and_relative_objective_stall"
    assert final.step_size == 0.0
    assert final.proposed_parameter_update_norm < 1e-10
    assert final.relative_objective_change < 1e-8
    stats = final.get_statistics()
    assert (
        stats["proposed_parameter_update_norm"] == final.proposed_parameter_update_norm
    )
    assert stats["relative_objective_change"] == final.relative_objective_change


@pytest.mark.parametrize("position_only", [False, True])
def test_common_weight_units_do_not_change_undamped_stopping_or_solution(position_only):
    reference, joints = noisy_calibrator(position_only)
    baseline = reference.solve(max_iterations=30)
    expected = reference.forward_kinematics_batch(joints)
    assert baseline.has_converged
    for scale in [1e-8, 1e8]:
        calibrator, _ = noisy_calibrator(position_only)
        result = calibrator.solve(
            max_iterations=30, position_weight=scale, orientation_weight=scale
        )
        assert result.has_converged
        assert result.nb_iterations_executed == baseline.nb_iterations_executed
        np.testing.assert_allclose(
            calibrator.forward_kinematics_batch(joints), expected, atol=1e-8
        )
        assert result.iteration_results[-1].objective_scale == scale


@pytest.mark.parametrize("position_only", [False, True])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_insufficient_search_range_with_substantial_update_remains_failure(
    position_only, scale
):
    calibrator, _ = noisy_calibrator(position_only)
    nominal = np.asarray(calibrator.joint_screw_axis).copy()
    result = calibrator.solve(
        max_iterations=1,
        step_size=100.0,
        min_step_size=100.0,
        position_weight=scale,
        orientation_weight=scale,
    )
    assert not result.has_converged
    assert result.termination_reason == "line_search_failed"
    assert result.convergence_criterion is None
    assert result.iteration_results[-1].proposed_parameter_update_norm > 1e-3
    np.testing.assert_array_equal(calibrator.joint_screw_axis, nominal)


@pytest.mark.parametrize("name", ["update_tolerance", "relative_objective_tolerance"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True])
def test_invalid_stopping_tolerances_are_rejected(name, value):
    calibrator, _ = noisy_calibrator()
    with pytest.raises(ValueError, match=name):
        calibrator.solve(**{name: value})
