"""Array implementations of the same space-frame POE/adjoint equations.

The public scalar SpatialMath implementation remains in ``calibration.py`` and
serves as the independent reference for these batched calculations.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def skew_batch(vectors):
    result = np.zeros((len(vectors), 3, 3))
    result[:, 0, 1] = -vectors[:, 2]
    result[:, 0, 2] = vectors[:, 1]
    result[:, 1, 0] = vectors[:, 2]
    result[:, 1, 2] = -vectors[:, 0]
    result[:, 2, 0] = -vectors[:, 1]
    result[:, 2, 1] = vectors[:, 0]
    return result


def adjoint_batch(rotations, translations):
    result = np.zeros((len(rotations), 6, 6))
    result[:, :3, :3] = rotations
    result[:, :3, 3:] = skew_batch(translations) @ rotations
    result[:, 3:, 3:] = rotations
    return result


def pose_log_batch(rotations, translations):
    """Vectorized stable SE(3) logarithm, returning [N,6] in [v,w] order."""
    angular = Rotation.from_matrix(rotations).as_rotvec()
    theta = np.linalg.norm(angular, axis=1)
    coefficients = np.empty(len(theta))
    small = theta < 1e-4
    coefficients[small] = (
        1.0 / 12.0 + theta[small] ** 2 / 720.0 + theta[small] ** 4 / 30240.0
    )
    angles = theta[~small]
    coefficients[~small] = (1.0 - 0.5 * angles / np.tan(0.5 * angles)) / angles**2
    cross = np.cross(angular, translations)
    linear = (
        translations - 0.5 * cross + coefficients[:, None] * np.cross(angular, cross)
    )
    return np.column_stack((linear, angular))


def evaluate_poe_batch(axes, home, configurations, basis=None, observation_size=6):
    """Evaluate poses and optional adjoint regressor with shared chain prefixes.

    Rodrigues rotations plus their fixed-axis translation evaluate ``exp(S*q)``.
    The pitch term preserves the scalar exponential even when floating-point
    updates leave a tiny nonzero ``v dot w``. No cache survives model updates.
    """
    count = len(configurations)
    rotations = np.repeat(np.eye(3)[None], count, axis=0)
    translations = np.zeros((count, 3))
    blocks = None
    previous_adjoint = None
    if basis is not None:
        blocks = np.empty((count, 6, basis.shape[1]))
        previous_adjoint = np.repeat(np.eye(6)[None], count, axis=0)
    for index, axis in enumerate(axes):
        linear, angular = axis[:3], axis[3:]
        angular_norm_squared = np.dot(angular, angular)
        angles = configurations[:, index]
        if angular_norm_squared == 0:
            local_rotations = np.repeat(np.eye(3)[None], count, axis=0)
            local_translations = angles[:, None] * linear
        else:
            local_rotations = Rotation.from_rotvec(
                angles[:, None] * angular
            ).as_matrix()
            point = np.cross(angular, linear) / angular_norm_squared
            pitch_translation = angular * (
                np.dot(angular, linear) / angular_norm_squared
            )
            local_translations = (
                point
                - np.einsum("nij,j->ni", local_rotations, point)
                + angles[:, None] * pitch_translation
            )
        translations = translations + np.einsum(
            "nij,nj->ni", rotations, local_translations
        )
        rotations = rotations @ local_rotations
        if basis is not None:
            current_adjoint = adjoint_batch(rotations, translations)
            joint_basis = basis[
                6 * index : 6 * (index + 1), 4 * index : 4 * (index + 1)
            ]
            blocks[:, :, 4 * index : 4 * (index + 1)] = (
                previous_adjoint - current_adjoint
            ) @ joint_basis
            previous_adjoint = current_adjoint
    if basis is not None:
        blocks[:, :, -observation_size:] = previous_adjoint[:, :, :observation_size]
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = rotations @ home[:3, :3]
    poses[:, :3, 3] = translations + np.einsum("nij,j->ni", rotations, home[:3, 3])
    return poses, blocks
