"""Independent analytic motions and compatibility of batched POE validation."""

from itertools import permutations

import numpy as np
import pytest

from kinematic_calib.validation import evaluate_saved_poe


def rotation_x(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])


def rotation_y(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


@pytest.mark.parametrize("two_axes", [False, True])
@pytest.mark.parametrize("translation_scale", [1.0, 10000.0])
def test_validation_matches_analytic_revolute_motions_and_known_pose_errors(
    two_axes, translation_scale
):
    # The first axis is world Z; the second, when present, is Y through [a,0,0].
    # These direct rigid-motion equations do not use matrix exponentials or FK.
    offset = np.array([0.4, 0.0, 0.0])
    axes = np.array([[0, 0, 0, 0, 0, 1.0], [0, 0, 0.4, 0, 1.0, 0]])
    home = np.eye(4)
    home[:3, :3] = rotation_x(0.2)
    home[:3, 3] = np.array([0.7, -0.2, 0.3]) * translation_scale
    configurations = np.array(
        [
            [0, 0],
            [1e-12, -1e-12],
            [np.pi - 1e-9, np.pi / 2],
            [-np.pi, -0.3],
            [0.3, -0.2],
        ]
    )
    if not two_axes:
        axes, configurations = axes[:1], configurations[:, :1]
    observed = np.repeat(np.eye(4)[None], len(configurations), axis=0)
    position_errors = np.array([0.001, 0.003, 0.002, 0.004, 0.005])
    orientation_errors = np.array([0, 1e-12, 0.2, np.pi - 1e-9, np.pi])
    for index, angles in enumerate(configurations):
        first = rotation_z(angles[0])
        second = rotation_y(angles[1]) if two_axes else np.eye(3)
        observed[index, :3, :3] = (
            rotation_x(orientation_errors[index]) @ first @ second @ home[:3, :3]
        )
        observed[index, :3, 3] = first @ (offset + second @ (home[:3, 3] - offset))
        observed[index, 0, 3] += position_errors[index]
    indices = np.array([91, 17, 205, 8, 63])
    original_observations = observed.copy()
    original_configurations = configurations.copy()

    result = evaluate_saved_poe(axes, home, configurations, observed, indices)

    actual_position = [row["position_error_m"] for row in result["sample_errors"]]
    actual_orientation = [
        row["orientation_error_rad"] for row in result["sample_errors"]
    ]
    np.testing.assert_allclose(actual_position, position_errors, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        actual_orientation, orientation_errors, atol=3e-15, rtol=0
    )
    assert result["position_error_mean_m"] == pytest.approx(
        position_errors.mean(), abs=1e-10
    )
    assert result["position_error_rms_m"] == pytest.approx(
        np.sqrt(np.mean(position_errors**2)), abs=1e-10
    )
    assert result["position_error_p95_m"] == pytest.approx(
        np.percentile(position_errors, 95), abs=1e-10
    )
    assert result["orientation_error_rms_rad"] == pytest.approx(
        np.sqrt(np.mean(orientation_errors**2))
    )
    assert result["position_error_worst_sample_index"] == 4
    assert result["position_error_worst_source_sample_index"] == 63
    assert result["orientation_error_worst_source_sample_index"] == 63
    assert [
        row["source_sample_index"] for row in result["sample_errors"]
    ] == indices.tolist()
    assert [
        row["joint_positions_rad"] for row in result["sample_errors"]
    ] == configurations.tolist()
    np.testing.assert_array_equal(observed, original_observations)
    np.testing.assert_array_equal(configurations, original_configurations)


def test_accepted_home_roundoff_keeps_general_inverse_semantics():
    # This accepted 4x4 home has a small bottom-row perturbation. Its inverse
    # contains a rotation-block shear that a rigid transpose would miss.
    home = np.eye(4)
    home[1, 3] = 20000
    home[3, 0] = 5e-9
    result = evaluate_saved_poe(
        np.array([[0, 0, 0, 0, 0, 1.0]]),
        home,
        np.zeros((2, 1)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        np.array([17, 43]),
    )
    # The inverse upper-left block is [[1,0,0],[1e-4,1,0],[0,0,1]].
    # Its nearest rotation has angle atan(1e-4/2). Older SciPy's quaternion
    # projection differs only at cubic order, below this absolute tolerance.
    expected_angle = np.arctan(1e-4 / 2)
    assert result["orientation_error_mean_rad"] == pytest.approx(
        expected_angle, abs=2e-12
    )
    assert result["position_error_mean_m"] == 20000


def test_scalar_distance_accumulation_preserves_near_tied_worst_source_id():
    translations = np.array(
        list(
            permutations(
                [-4.180422279378624, -3.3593128893899204e-5, -4.104663570938061e-5]
            )
        )
    )
    measured = np.repeat(np.eye(4)[None], len(translations), axis=0)
    measured[:, :3, 3] = translations
    indices = np.arange(len(translations)) * 3 + 123
    # Equal mathematical distances can round differently by accumulation order.
    # Retain the scalar validator's ordering on each supported NumPy/BLAS build.
    scalar_errors = np.asarray([np.linalg.norm(row) for row in translations])
    expected_worst = int(np.argmax(scalar_errors))

    result = evaluate_saved_poe(
        np.array([[0, 0, 0, 0, 0, 1.0]]),
        np.eye(4),
        np.zeros((len(translations), 1)),
        measured,
        indices,
    )

    np.testing.assert_array_equal(
        [row["position_error_m"] for row in result["sample_errors"]], scalar_errors
    )
    assert result["position_error_worst_sample_index"] == expected_worst
    assert result["position_error_worst_source_sample_index"] == indices[expected_worst]
