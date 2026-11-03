"""Independent exponential-map checks for the solver's SE(3) logarithm."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from robotkinecal.calibration import _pose_log


def exponential_via_matrix_expm(vector):
    vx, vy, vz, wx, wy, wz = vector
    matrix = np.array(
        [
            [0, -wz, wy, vx],
            [wz, 0, -wx, vy],
            [-wy, wx, 0, vz],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return expm(matrix)


@pytest.mark.parametrize(
    "angle", [0.0, 1e-12, 1e-7, 1e-3, 0.5, 2.1, np.pi - 1e-8, np.pi]
)
@pytest.mark.parametrize("translation_scale", [0.0, 1.0, 1e6])
def test_logarithm_roundtrips_through_independent_matrix_exponential(
    angle, translation_scale
):
    axis = np.array([0.2, -0.3, 0.7])
    axis /= np.linalg.norm(axis)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    matrix[:3, 3] = translation_scale * np.array([1.2, -0.9, 2.0])
    logarithm = _pose_log(SE3(matrix))
    assert np.all(np.isfinite(logarithm))
    reconstructed = exponential_via_matrix_expm(logarithm)
    np.testing.assert_allclose(reconstructed[:3, :3], matrix[:3, :3], atol=1e-8)
    np.testing.assert_allclose(
        reconstructed[:3, 3], matrix[:3, 3], atol=2e-8, rtol=1e-12
    )
    if angle < np.pi - 1e-8:
        np.testing.assert_allclose(logarithm[3:], angle * axis, atol=1e-12)


def test_pure_rotation_about_offset_point_uses_linear_angular_convention():
    axis = np.array([0.0, 0.0, 1.0])
    point = np.array([1.5, -2.0, 0.7])
    angle = 1.7
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    matrix[:3, 3] = point - matrix[:3, :3] @ point
    expected = angle * np.r_[np.cross(point, axis), axis]
    actual = _pose_log(SE3(matrix))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(exponential_via_matrix_expm(actual), matrix, atol=1e-12)


@pytest.mark.parametrize("angle", [0.001, 0.2, 1.5, 2.9])
def test_matches_spatialmath_logarithm_away_from_branch_boundary(angle):
    axis = np.array([0.1, 0.8, -0.3])
    axis /= np.linalg.norm(axis)
    pose = SE3.Rt(Rotation.from_rotvec(angle * axis).as_matrix(), [0.7, -0.2, 0.9])
    np.testing.assert_allclose(_pose_log(pose), pose.log(twist=True), atol=1e-12)
