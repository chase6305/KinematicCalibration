import numpy as np
import pytest

from kinematic_calib.validation import (
    evaluate_saved_poe,
    validate_saved_poe_parameters,
)


def test_saved_poe_validation_recovers_exact_revolute_motion() -> None:
    screw_axes = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    joints = np.array([[0.0], [0.2], [-0.4]])
    poses = []
    for (angle,) in joints:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        poses.append(
            [
                [cosine, -sine, 0.0, 0.0],
                [sine, cosine, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    result = evaluate_saved_poe(
        screw_axes,
        np.eye(4),
        joints,
        np.asarray(poses),
        sample_indices=np.array([10, 20, 30]),
    )

    assert result["position_error_max_m"] < 1e-12
    assert result["orientation_error_max_rad"] < 1e-12
    assert result["position_error_p95_m"] < 1e-12
    assert result["position_error_worst_sample_index"] in range(3)
    assert len(result["sample_errors"]) == 3
    assert [item["source_sample_index"] for item in result["sample_errors"]] == [
        10,
        20,
        30,
    ]
    assert result["sample_errors"][1]["joint_positions_rad"] == [0.2]


def test_saved_poe_validation_rejects_non_rigid_zero_pose() -> None:
    zero_pose = np.eye(4)
    zero_pose[0, 0] = 2.0

    with pytest.raises(ValueError, match="not orthonormal"):
        evaluate_saved_poe(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            zero_pose,
            np.zeros((1, 1)),
            np.eye(4)[None],
        )


def test_saved_poe_validation_rejects_non_unit_revolute_axis() -> None:
    with pytest.raises(ValueError, match="unit angular"):
        evaluate_saved_poe(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
            np.eye(4),
            np.zeros((1, 1)),
            np.eye(4)[None],
        )


def test_saved_poe_parameter_validation_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_saved_poe_parameters(np.empty((0, 6)), np.eye(4))


def test_position_metric_does_not_mix_orientation_error_with_translation() -> None:
    home = np.eye(4)
    home[:3, 3] = [1.0, 0.5, 0.2]
    measured = home.copy()
    angle = 0.1
    measured[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    report = evaluate_saved_poe(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        home,
        np.zeros((1, 1)),
        measured[None],
    )
    assert report["position_error_max_m"] == 0.0
    assert report["orientation_error_max_rad"] == pytest.approx(angle)


def test_validation_rejects_nonrigid_observations_and_empty_data() -> None:
    axes = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    measured = np.eye(4)
    measured[0, 0] = 2.0
    with pytest.raises(ValueError, match="non-orthonormal"):
        evaluate_saved_poe(axes, np.eye(4), np.zeros((1, 1)), measured[None])
    with pytest.raises(ValueError, match="joint_positions"):
        evaluate_saved_poe(axes, np.eye(4), np.empty((0, 1)), np.empty((0, 4, 4)))


def test_validation_rejects_helical_axes():
    with pytest.raises(ValueError, match="zero pitch"):
        validate_saved_poe_parameters(
            np.array([[0.0, 0.0, 0.1, 0.0, 0.0, 1.0]]), np.eye(4)
        )
