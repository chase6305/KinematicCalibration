"""Encoder-offset fitting and diagnostics without the optional RTB adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from kinematic_calib.robot_model import estimate_joint_zero_offsets


def analytic_pose(angle):
    """Independent one-axis rotation about the fixed endpoint [0, 1, 0]."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0, 0],
            [sine, cosine, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


class FixedEndpointRobot:
    def fkine(self, joint_positions, *, end):
        assert end == "tool"
        return SE3(analytic_pose(joint_positions[0]), check=False)


def fit_fixed_extrinsics(robot, joints, observations, **options):
    return estimate_joint_zero_offsets(
        robot,
        "tool",
        joints,
        observations,
        camera_t_base=np.eye(4),
        ee_t_target=np.eye(4),
        **options,
    )


@pytest.mark.parametrize("orientation_error", [0.4, np.pi - 1e-8, np.pi])
def test_orientation_conflict_at_fixed_endpoint_has_zero_position_error(
    orientation_error,
):
    joints = np.linspace(-1, 1, 8)[:, None]
    observations = np.asarray([analytic_pose(row[0]) for row in joints])
    observations[:, :3, :3] = (
        Rotation.from_euler("x", orientation_error).as_matrix()
        @ observations[:, :3, :3]
    )

    result = fit_fixed_extrinsics(FixedEndpointRobot(), joints, observations)

    # Coupled spatial-twist residuals need not return zero offsets for this
    # conflicting fit. The endpoint stays fixed; reconstruct rotations at the
    # returned offset independently to check the reported angular errors.
    predicted = np.asarray(
        [analytic_pose(row[0] + result["offsets"][0]) for row in joints]
    )
    expected_orientation_errors = Rotation.from_matrix(
        observations[:, :3, :3] @ predicted[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    np.testing.assert_array_equal(result["position_errors_m"], np.zeros(len(joints)))
    assert result["position_error_rms_m"] == 0
    np.testing.assert_allclose(
        result["orientation_errors_rad"], expected_orientation_errors
    )
    assert result["orientation_error_rms_rad"] == pytest.approx(
        np.sqrt(np.mean(expected_orientation_errors**2))
    )
    assert np.isfinite(result["cost"])
    assert np.isfinite(result["singular_values"]).all()


def test_encoder_bias_is_recovered_with_opposite_sign_and_fixed_frame_transforms():
    actual_angles = np.linspace(-1.1, 1.3, 12)
    encoder_bias = 0.017
    recorded_joints = (actual_angles + encoder_bias)[:, None]
    camera_t_base = SE3.RPY([0.2, -0.3, 0.1]) * SE3.Trans(0.4, -0.1, 0.8)
    ee_t_target = SE3.RPY([-0.1, 0.05, 0.2]) * SE3.Trans(0.03, 0.02, 0.1)
    observations = np.asarray(
        [
            camera_t_base.A @ analytic_pose(angle) @ ee_t_target.A
            for angle in actual_angles
        ]
    )

    result = estimate_joint_zero_offsets(
        FixedEndpointRobot(),
        "tool",
        recorded_joints,
        observations,
        camera_t_base=camera_t_base.A,
        ee_t_target=ee_t_target.A,
    )

    np.testing.assert_allclose(result["offsets"], [-encoder_bias], atol=1e-9)
    assert result["full_rank"] and result["estimated_indices"].tolist() == [0]
    assert result["position_error_max_m"] < 1e-10
    assert result["orientation_error_max_rad"] < 1e-9


def test_uncertainty_retains_weak_directions_of_a_known_local_jacobian(monkeypatch):
    # Control the optimizer's local linear problem. Its known orthogonal
    # factors give covariance directly, without forming a normal matrix.
    rng = np.random.default_rng(628)
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    singular_values = np.array([1.0, 1e-5, 1e-10])
    captured = {}

    def known_linear_problem(residual, initial, **_):
        errors = residual(initial)
        left, _ = np.linalg.qr(np.column_stack([errors, rng.normal(size=(24, 3))]))
        jacobian = (left[:, 1:4] * singular_values) @ right.T
        captured["squared_error"] = float(errors @ errors)
        return SimpleNamespace(
            success=True,
            x=initial,
            jac=jacobian,
            fun=errors,
            message="known local linear problem",
            nfev=1,
            cost=float(errors @ errors) / 2,
        )

    monkeypatch.setattr("scipy.optimize.least_squares", known_linear_problem)
    joints = np.zeros((4, 3))
    observations = np.repeat(analytic_pose(0)[None], 4, axis=0)
    observations[:, :3, :3] = Rotation.from_euler("x", 0.4).as_matrix()

    result = fit_fixed_extrinsics(FixedEndpointRobot(), joints, observations)

    variance = captured["squared_error"] / (24 - 3)
    expected_std = np.sqrt(variance * (right**2 @ singular_values**-2))
    np.testing.assert_allclose(result["offset_std_rad"], expected_std, rtol=2e-6)
    assert result["jacobian_rank"] == result["parameter_count"] == 3
    assert result["condition_number"] == pytest.approx(1e10, rel=2e-6)


@pytest.mark.parametrize(
    "joints",
    [
        [],
        [0.0],
        np.zeros((2, 0)),
        np.zeros((2, 1), dtype=bool),
        np.zeros((2, 1), dtype=complex),
        [[np.nan], [0]],
        [[np.inf], [0]],
    ],
)
def test_invalid_joint_inputs_fail_before_forward_kinematics(joints):
    with pytest.raises(ValueError, match="joint_positions"):
        fit_fixed_extrinsics(
            FixedEndpointRobot(), joints, np.repeat(np.eye(4)[None], 2, axis=0)
        )


def test_mismatched_sample_counts_fail_before_forward_kinematics():
    with pytest.raises(ValueError, match="sample counts"):
        fit_fixed_extrinsics(FixedEndpointRobot(), np.zeros((2, 1)), np.eye(4)[None])


@pytest.mark.parametrize("bound", [True, 0, -1, np.nan, np.inf, 1j, [0.5]])
def test_invalid_offset_bounds_are_rejected(bound):
    with pytest.raises(ValueError, match="max_offset_rad"):
        fit_fixed_extrinsics(
            FixedEndpointRobot(),
            np.zeros((2, 1)),
            np.repeat(np.eye(4)[None], 2, axis=0),
            max_offset_rad=bound,
        )
