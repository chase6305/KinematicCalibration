"""Batch POE equivalence against the retained scalar implementation."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from robotkinecal import SerialRobotKineCal
from robotkinecal._batch import pose_log_batch
from robotkinecal.calibration import _pose_log


def make_calibrator(dof, position_only):
    rng = np.random.default_rng(600 + dof)
    angular = rng.normal(size=(dof, 3))
    angular /= np.linalg.norm(angular, axis=1)[:, None]
    points = rng.uniform(-0.5, 0.5, size=(dof, 3))
    axes = np.column_stack((np.cross(points, angular), angular))
    home = SE3.Rt(Rotation.from_rotvec([0.2, -0.3, 0.4]).as_matrix(), [0.3, 0.6, 0.4])
    calibrator = SerialRobotKineCal.from_screw_axes(axes, home)
    configurations = rng.uniform(-3, 3, size=(40, dof))
    configurations[0] = 0.0
    configurations[1] = 1e-12
    configurations[2] = np.pi
    configurations[3] = -np.pi + 1e-8
    observations = [calibrator.forward_kinematics(dof + 1, q) for q in configurations]
    if position_only:
        observations = [pose.t for pose in observations]
    calibrator.set_observations(configurations, observations)
    return calibrator, configurations


@pytest.mark.parametrize("dof", [1, 6, 7])
@pytest.mark.parametrize("position_only", [False, True])
def test_batch_fk_and_regressors_match_scalar_before_after_update_reset(
    dof, position_only
):
    calibrator, configurations = make_calibrator(dof, position_only)
    for state in ["nominal", "updated", "reset"]:
        if state == "updated":
            perturbation = np.random.default_rng(12).normal(
                scale=0.03, size=calibrator.N_PARAMS
            )
            calibrator.update_twist_definitions(perturbation)
        elif state == "reset":
            calibrator.reset()
        expected_poses = np.asarray(
            [calibrator.forward_kinematics(dof + 1, q).A for q in configurations]
        )
        expected_blocks = np.asarray([calibrator.A_Matrix(q) for q in configurations])
        np.testing.assert_allclose(
            calibrator.forward_kinematics_batch(configurations),
            expected_poses,
            atol=2e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            calibrator.regressor_blocks(configurations),
            expected_blocks,
            atol=2e-12,
            rtol=1e-12,
        )


def test_batch_basis_dimensions_follow_observation_type_switch():
    calibrator, configurations = make_calibrator(6, False)
    assert calibrator.regressor_blocks(configurations).shape == (40, 6, 30)
    poses = calibrator.forward_kinematics_batch(configurations)
    calibrator.set_observations(configurations, poses[:, :3, 3])
    assert calibrator.regressor_blocks(configurations).shape == (40, 6, 27)
    np.testing.assert_allclose(
        calibrator.regressor_blocks(configurations),
        np.asarray([calibrator.A_Matrix(q) for q in configurations]),
        atol=2e-12,
    )


def test_single_configuration_batch_matches_scalar():
    calibrator, configurations = make_calibrator(7, False)
    assert calibrator.forward_kinematics_batch(configurations[:1]).shape == (1, 4, 4)
    assert calibrator.regressor_blocks(configurations[:1]).shape == (1, 6, 34)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.zeros((0, 6)),
        np.zeros((3, 5)),
        np.full((2, 6), np.nan),
        np.ones((2, 6), dtype=complex),
        np.ones(6),
    ],
)
def test_batch_input_contract_rejects_invalid_arrays(bad):
    calibrator, _ = make_calibrator(6, False)
    with pytest.raises(ValueError):
        calibrator.forward_kinematics_batch(bad)
    with pytest.raises(ValueError):
        calibrator.regressor_blocks(bad)


def test_batch_log_matches_scalar_and_independent_exponential_at_branch_boundaries():
    angles = np.array([0.0, 1e-12, 1e-7, 0.5, 2.0, np.pi - 1e-8, np.pi])
    rng = np.random.default_rng(809)
    angular_axes = rng.normal(size=(len(angles), 3))
    angular_axes /= np.linalg.norm(angular_axes, axis=1)[:, None]
    rotations = Rotation.from_rotvec(angles[:, None] * angular_axes).as_matrix()
    translations = rng.normal(size=(len(angles), 3)) * 100
    logarithms = pose_log_batch(rotations, translations)
    expected = np.asarray(
        [
            _pose_log(SE3.Rt(rotation, translation))
            for rotation, translation in zip(rotations, translations, strict=True)
        ]
    )
    np.testing.assert_allclose(logarithms, expected, atol=1e-12)
    for rotation, translation, logarithm in zip(
        rotations, translations, logarithms, strict=True
    ):
        v, w = logarithm[:3], logarithm[3:]
        wedge = np.zeros((4, 4))
        wedge[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
        wedge[:3, 3] = v
        actual = expm(wedge)
        np.testing.assert_allclose(actual[:3, :3], rotation, atol=2e-12)
        np.testing.assert_allclose(actual[:3, 3], translation, atol=2e-10)


class ScalarReference(SerialRobotKineCal):
    """Keep optimization identical while replacing all batch FK/regressors."""

    def _model_batch(self, joint_positions, *, with_regressor=False):
        poses = np.asarray(
            [self.forward_kinematics(self.N_JOINTS + 1, q).A for q in joint_positions]
        )
        blocks = (
            np.asarray([self.A_Matrix(q) for q in joint_positions])
            if with_regressor
            else None
        )
        return poses, blocks


@pytest.mark.parametrize("dof", [6, 7])
@pytest.mark.parametrize("position_only", [False, True])
def test_complete_batch_solve_matches_scalar_reference_on_independent_held_out_poses(
    dof, position_only
):
    batched, configurations = make_calibrator(dof, position_only)
    reference = ScalarReference.from_screw_axes(
        batched.joint_screw_axis, batched.zero_conf_EE_pose
    )
    perturbation = SE3.Rt(
        Rotation.from_rotvec([0.01, -0.02, 0.015]).as_matrix(), [0.003, -0.005, 0.002]
    )
    measurements = [
        batched.forward_kinematics(dof + 1, q) @ perturbation for q in configurations
    ]
    if position_only:
        measurements = [pose.t for pose in measurements]
    for calibrator in [batched, reference]:
        calibrator.set_observations(configurations, measurements)
    batched_result = batched.solve(max_iterations=15)
    reference_result = reference.solve(max_iterations=15)
    assert batched_result.has_converged and reference_result.has_converged
    held_out = np.random.default_rng(967).uniform(-3.0, 3.0, size=(20, dof))
    predictions = batched.forward_kinematics_batch(held_out)
    expected = np.asarray(
        [reference.forward_kinematics(dof + 1, q).A for q in held_out]
    )
    np.testing.assert_allclose(predictions, expected, atol=1e-9)
    np.testing.assert_allclose(
        batched_result.iteration_results[-1].post_update_twist_errors_norm,
        reference_result.iteration_results[-1].post_update_twist_errors_norm,
        atol=1e-9,
    )
    assert (
        batched_result.iteration_results[-1].matrix_rank
        == reference_result.iteration_results[-1].matrix_rank
    )
