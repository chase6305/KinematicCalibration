"""Independent residual evaluation for saved POE calibration parameters."""

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from .dataset import KinematicDataset, ObservationFrame


def _twist_matrix(axis: np.ndarray) -> np.ndarray:
    v = axis[:3]
    w = axis[3:]
    wx, wy, wz = w
    matrix = np.array(
        [
            [0.0, -wz, wy, v[0]],
            [wz, 0.0, -wx, v[1]],
            [-wy, wx, 0.0, v[2]],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    return matrix


def validate_saved_poe_parameters(
    screw_axes: np.ndarray, zero_configuration_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a saved revolute space-frame POE model."""
    raw_axes, raw_pose = np.asarray(screw_axes), np.asarray(zero_configuration_pose)
    if raw_axes.dtype.kind not in "iuf" or raw_pose.dtype.kind not in "iuf":
        raise ValueError("saved POE parameters must contain real numeric values")
    axes = np.asarray(raw_axes, dtype=float)
    zero_pose = np.asarray(raw_pose, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 6 or len(axes) == 0:
        raise ValueError("screw_axes must have non-empty shape [DOF, 6]")
    if not np.all(np.isfinite(axes)):
        raise ValueError("screw_axes contain non-finite values")
    angular_norms = np.linalg.norm(axes[:, 3:], axis=1)
    if not np.allclose(angular_norms, 1.0, rtol=0, atol=1e-5):
        raise ValueError("saved revolute screw axes must have unit angular vectors")
    if not np.allclose(
        np.sum(axes[:, :3] * axes[:, 3:], axis=1), 0.0, rtol=0, atol=1e-6
    ):
        raise ValueError("saved revolute screw axes must have zero pitch (v dot w = 0)")
    if zero_pose.shape != (4, 4) or not np.all(np.isfinite(zero_pose)):
        raise ValueError("zero_configuration_pose must be a finite 4x4 matrix")
    if not np.allclose(zero_pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-8):
        raise ValueError("zero_configuration_pose has an invalid homogeneous row")
    zero_rotation = zero_pose[:3, :3]
    if not np.allclose(zero_rotation.T @ zero_rotation, np.eye(3), rtol=0, atol=1e-6):
        raise ValueError("zero_configuration_pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(zero_rotation), 1.0, rtol=0, atol=1e-6):
        raise ValueError("zero_configuration_pose rotation determinant must be +1")
    return axes, zero_pose


def evaluate_saved_poe(
    screw_axes: np.ndarray,
    zero_configuration_pose: np.ndarray,
    joint_positions: np.ndarray,
    observed_poses: np.ndarray,
    sample_indices: np.ndarray | None = None,
) -> dict:
    """Evaluate saved space-frame POE parameters against pose observations."""
    axes, zero_pose = validate_saved_poe_parameters(screw_axes, zero_configuration_pose)
    raw_joints, raw_observed = np.asarray(joint_positions), np.asarray(observed_poses)
    if raw_joints.dtype.kind not in "iuf" or raw_observed.dtype.kind not in "iuf":
        raise ValueError("validation observations must contain real numeric values")
    joints = np.asarray(raw_joints, dtype=float)
    observed = np.asarray(raw_observed, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != len(axes):
        raise ValueError("joint positions do not match saved screw-axis count")
    if observed.shape != (len(joints), 4, 4):
        raise ValueError("observed poses must have shape [N, 4, 4]")
    dataset = KinematicDataset(
        joints, observed, ObservationFrame.BASE_T_EE, sample_indices
    )
    source_indices = dataset.sample_indices

    generators = [_twist_matrix(axis) for axis in axes]
    predicted_poses = np.repeat(np.eye(4)[None], len(joints), axis=0)
    # General matrix exponentials keep this FK independent of the optimizer.
    for generator, positions in zip(generators, joints.T, strict=True):
        predicted_poses = predicted_poses @ expm(positions[:, None, None] * generator)
    predicted_poses = predicted_poses @ zero_pose
    # Retain the general inverse for accepted near-rigid home matrices, whose
    # small roundoff need not have an inverse equal to a rigid transpose.
    relative_poses = observed @ np.linalg.inv(predicted_poses)
    # Preserve scalar norm accumulation so near-tied worst-sample IDs agree
    # with the original validator as well as the reported residual values.
    position_errors = np.asarray(
        [
            np.linalg.norm(measured[:3, 3] - predicted[:3, 3])
            for measured, predicted in zip(observed, predicted_poses, strict=True)
        ]
    )
    orientation_errors = Rotation.from_matrix(relative_poses[:, :3, :3]).magnitude()
    return {
        "samples": int(len(joints)),
        "position_error_mean_m": float(position_errors.mean()),
        "position_error_max_m": float(position_errors.max()),
        "position_error_rms_m": float(np.sqrt(np.mean(position_errors**2))),
        "position_error_p95_m": float(np.percentile(position_errors, 95)),
        "position_error_worst_sample_index": int(np.argmax(position_errors)),
        "position_error_worst_source_sample_index": int(
            source_indices[np.argmax(position_errors)]
        ),
        "orientation_error_mean_rad": float(orientation_errors.mean()),
        "orientation_error_max_rad": float(orientation_errors.max()),
        "orientation_error_rms_rad": float(np.sqrt(np.mean(orientation_errors**2))),
        "orientation_error_p95_rad": float(np.percentile(orientation_errors, 95)),
        "orientation_error_worst_sample_index": int(np.argmax(orientation_errors)),
        "orientation_error_worst_source_sample_index": int(
            source_indices[np.argmax(orientation_errors)]
        ),
        "sample_errors": [
            {
                "sample_index": int(index),
                "source_sample_index": int(source_indices[index]),
                "joint_positions_rad": joints[index].tolist(),
                "position_error_m": float(position_error),
                "orientation_error_rad": float(orientation_error),
            }
            for index, (position_error, orientation_error) in enumerate(
                zip(position_errors, orientation_errors, strict=True)
            )
        ],
    }
