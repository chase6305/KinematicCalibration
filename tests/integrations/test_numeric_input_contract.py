"""Reject accidental complex arrays before imaginary values can be discarded."""

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset, ObservationFrame, invert_transform
from kinematic_calib.validation import evaluate_saved_poe, validate_saved_poe_parameters


def test_dataset_rejects_complex_encoder_and_pose_arrays():
    joints = np.zeros((1, 1), dtype=complex)
    joints[0, 0] = 0.1 + 0.2j
    poses = np.eye(4)[None]
    with pytest.raises(ValueError, match="real numeric"):
        KinematicDataset(joints, poses, ObservationFrame.BASE_T_EE)
    complex_poses = poses.astype(complex)
    complex_poses[0, 0, 3] += 0.2j
    with pytest.raises(ValueError, match="real numeric"):
        KinematicDataset(joints.real, complex_poses, ObservationFrame.BASE_T_EE)
    with pytest.raises(ValueError, match="real numeric"):
        invert_transform(complex_poses[0])


def test_saved_model_and_validation_reject_complex_parameters():
    axes = np.array([[0, 0, 0, 0, 0, 1.0]])
    home = np.eye(4)
    complex_axes = axes.astype(complex)
    complex_axes[0, 0] = 0.3j
    with pytest.raises(ValueError, match="real numeric"):
        validate_saved_poe_parameters(complex_axes, home)
    with pytest.raises(ValueError, match="real numeric"):
        evaluate_saved_poe(axes, home, np.array([[0.1 + 0.3j]]), home[None])


def test_boolean_or_out_of_int64_source_ids_do_not_become_sample_integers():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    joints = np.zeros((2, 1))
    for ids in ([False, True], [0.0, 2.0**63], np.array([0, 2**63], dtype=np.uint64)):
        with pytest.raises(ValueError, match="integers"):
            KinematicDataset(joints, poses, ObservationFrame.BASE_T_EE, ids)
