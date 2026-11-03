"""Independent POE construction and calibration without a simulator or RTB."""

import subprocess
import sys

import numpy as np
import pytest
from spatialmath import SE3, Twist3

from robotkinecal import SerialRobotKineCal


def make_problem(position_only=False):
    rng = np.random.default_rng(412)
    angular = rng.normal(size=(6, 3))
    angular /= np.linalg.norm(angular, axis=1)[:, None]
    points = rng.uniform(-0.4, 0.4, size=(6, 3))
    axes = np.column_stack((np.cross(points, angular), angular))
    home = SE3.Rt(SE3.RPY([0.3, -0.2, 0.1]).R, [0.4, 0.2, 0.6])
    truth_axes = np.stack(
        [Twist3(rng.normal(scale=0.002, size=6)).SE3().Ad() @ axis for axis in axes]
    )
    truth_home = Twist3(rng.normal(scale=0.002, size=6)).SE3() @ home
    truth = SerialRobotKineCal.from_screw_axes(truth_axes, truth_home)
    configurations = rng.uniform(-2, 2, size=(30, 6))
    poses = [truth.forward_kinematics(7, q) for q in configurations]
    observations = np.stack([pose.t for pose in poses]) if position_only else poses
    nominal = SerialRobotKineCal.from_screw_axes(axes, home)
    nominal.set_observations(configurations, observations)
    return nominal, truth, configurations


@pytest.mark.parametrize("position_only", [False, True])
def test_axis_and_home_errors_are_recovered_and_generalize(position_only):
    calibrator, truth, _ = make_problem(position_only)
    result = calibrator.solve(max_iterations=15)
    assert result.has_converged
    assert result.iteration_results[-1].post_update_twist_errors_norm < 1e-6
    held_out = np.random.default_rng(71).uniform(-2, 2, size=(8, 6))
    for q in held_out:
        actual = calibrator.forward_kinematics(7, q)
        expected = truth.forward_kinematics(7, q)
        np.testing.assert_allclose(actual.t, expected.t, atol=1e-6)
        if not position_only:
            np.testing.assert_allclose(actual.R, expected.R, atol=1e-6)


def test_import_and_screw_axis_constructor_do_not_import_robotics_toolbox():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from robotkinecal import SerialRobotKineCal; "
            "SerialRobotKineCal.from_screw_axes([[0, 0, 0, 0, 0, 1]], "
            "[[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]); "
            "assert 'roboticstoolbox' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "axes",
    [
        [],
        [[0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 2]],
        [[0, 0, 1, 0, 0, 1]],
        [[np.nan, 0, 0, 0, 0, 1]],
    ],
)
def test_invalid_screw_axes_are_rejected(axes):
    with pytest.raises(ValueError):
        SerialRobotKineCal.from_screw_axes(axes, np.eye(4))


@pytest.mark.parametrize("home", [np.eye(3), np.zeros((4, 4)), np.full((4, 4), np.nan)])
def test_invalid_home_transform_is_rejected(home):
    with pytest.raises(ValueError):
        SerialRobotKineCal.from_screw_axes([[0, 0, 0, 0, 0, 1]], home)


def test_caller_arrays_and_result_snapshots_are_independent():
    calibrator, _, _ = make_problem()
    nominal_axes = [axis.copy() for axis in calibrator.joint_screw_axis]
    result = calibrator.solve(max_iterations=2)
    final_axes = result.get_screw_axes()
    calibrator.reset()
    np.testing.assert_allclose(calibrator.joint_screw_axis, nominal_axes)
    np.testing.assert_allclose(result.get_screw_axes(), final_axes)


def test_changing_observation_type_rebuilds_basis():
    calibrator, truth, configurations = make_problem()
    pose_basis = calibrator.B_Matrix()
    points = np.stack([truth.forward_kinematics(7, q).t for q in configurations])
    calibrator.set_observations(configurations, points)
    point_basis = calibrator.B_Matrix()
    assert pose_basis.shape == (42, 30)
    assert point_basis.shape == (39, 27)
    assert calibrator.solve(max_iterations=15).has_converged


def test_invalid_pose_observation_is_rejected():
    calibrator, _, configurations = make_problem()
    invalid = np.eye(4)
    invalid[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrator.set_observations(configurations[:1], [SE3(invalid, check=False)])


def test_screw_axis_model_urdf_export_has_clear_error():
    calibrator, _, _ = make_problem()
    result = calibrator.solve(max_iterations=1)
    with pytest.raises(TypeError, match="URDF backend"):
        result.get_urdf_xyzrpy([])


@pytest.mark.parametrize("invalid", [True, "1.0", 1.0 + 0.01j])
def test_non_real_input_is_rejected_without_lossy_conversion(invalid):
    with pytest.raises(ValueError, match="real numeric"):
        SerialRobotKineCal.from_screw_axes(np.full((1, 6), invalid), np.eye(4))
    calibrator, _, _ = make_problem()
    with pytest.raises(ValueError, match="real numeric"):
        calibrator.forward_kinematics(7, np.full(6, invalid))


def test_near_pi_observation_errors_remain_finite():
    calibrator, _, configurations = make_problem()
    observations = []
    for index, q in enumerate(configurations):
        predicted = calibrator.forward_kinematics(7, q)
        rotation = SE3.Rz(np.pi - (index + 1) * 1e-7).R @ predicted.R
        observations.append(SE3.Rt(rotation, predicted.t))
    calibrator.set_observations(configurations, observations)
    result = calibrator.solve(max_iterations=1, damping=1e-3)
    iteration = result.iteration_results[0]
    assert np.all(np.isfinite(iteration.twist_errors))
    assert np.isfinite(iteration.post_update_twist_errors_norm)
    # Same endpoint with a large orientation error must not report a position error.
    np.testing.assert_allclose(iteration.position_errors, 0.0, atol=1e-12)
    np.testing.assert_allclose(iteration.orientation_errors, np.pi, atol=4e-6)


def test_repeated_configuration_exposes_unobservable_model():
    calibrator, truth, configurations = make_problem()
    repeated = np.repeat(configurations[:1], 30, axis=0)
    observations = [truth.forward_kinematics(7, q) for q in repeated]
    calibrator.set_observations(repeated, observations)
    result = calibrator.solve(max_iterations=6)
    final = result.iteration_results[-1]
    assert final.matrix_rank < final.N_PARAMS
    assert np.all(np.isfinite(final.twist_corrections))
    # A pseudoinverse must not masquerade as certainty in the null space.
    assert np.all(np.isnan(final.get_statistics()["joints_uncertainty"]))


def test_noisy_weighted_calibration_tracks_the_optimized_objective():
    calibrator, truth, configurations = make_problem()
    rng = np.random.default_rng(166)
    observations = []
    for q in configurations:
        pose = truth.forward_kinematics(7, q)
        orientation_noise = SE3.RPY(rng.normal(scale=0.1, size=3)).R
        observations.append(SE3.Rt(orientation_noise @ pose.R, pose.t))
    calibrator.set_observations(configurations, observations)
    result = calibrator.solve(
        max_iterations=30, position_weight=100.0, orientation_weight=0.01
    )
    assert result.termination_reason != "diverged"
    for iteration in result.iteration_results:
        assert iteration.post_update_objective_norm <= (
            iteration.pre_update_objective_norm + 1e-12
        )
    # Conflicting orientation measurements have a nonzero optimum. The
    # translational part of a spatial twist couples rotation and translation,
    # so this is deliberately not an endpoint-position accuracy assertion.
    assert result.iteration_results[-1].post_update_objective_norm < (
        0.9 * result.iteration_results[0].pre_update_objective_norm
    )
