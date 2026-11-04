import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")

from kinematic_calibration.handeye.opencv import (
    load_opencv_observations,
    solve_opencv_eye_to_hand,
)


def make_pose(rotation_vector, translation) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    pose[:3, 3] = translation
    return pose


@pytest.mark.parametrize("method", ["shah", "li"])
def test_opencv_eye_to_hand_recovers_ax_zb_solution(method: str) -> None:
    base_T_camera = make_pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    ee_T_board = make_pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    base_T_ee = np.stack(
        [
            make_pose(
                [0.08 * index, -0.05 * index**2, 0.04 * (index + 1)],
                [0.1 * np.sin(index), 0.08 * np.cos(index), 0.35 + 0.03 * index],
            )
            for index in range(1, 10)
        ]
    )
    camera_T_board = np.stack(
        [np.linalg.inv(base_T_camera) @ pose @ ee_T_board for pose in base_T_ee]
    )

    result = solve_opencv_eye_to_hand(base_T_ee, camera_T_board, method=method)

    np.testing.assert_allclose(result.base_T_camera, base_T_camera, atol=1e-7)
    np.testing.assert_allclose(result.ee_T_board, ee_T_board, atol=1e-7)
    assert result.position_error_max_m < 1e-8
    assert result.orientation_error_max_rad < 1e-7
    assert result.motion_rotation_rank >= 2
    assert result.max_rotation_from_first_rad > 0.1
    assert result.rotation_singular_value_2_rad > 0.0
    assert result.rotation_direction_ratio > 0.01


def test_opencv_eye_to_hand_rejects_unknown_method() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    with pytest.raises(ValueError, match="unsupported OpenCV"):
        solve_opencv_eye_to_hand(poses, poses, method="unknown")


def test_opencv_eye_to_hand_rejects_degenerate_rotation_excitation() -> None:
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4) * 0.1

    with pytest.raises(ValueError, match="independent rotation directions"):
        solve_opencv_eye_to_hand(poses, poses, method="shah")


def test_opencv_eye_to_hand_rejects_nearly_single_axis_excitation() -> None:
    base_T_ee = np.stack(
        [
            make_pose(
                [0.0, 1e-5 * index**2, 0.1 * index],
                [0.1 * index, 0.0, 0.0],
            )
            for index in range(5)
        ]
    )
    base_T_camera = make_pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    ee_T_board = make_pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    camera_T_board = np.stack(
        [np.linalg.inv(base_T_camera) @ pose @ ee_T_board for pose in base_T_ee]
    )

    with pytest.raises(ValueError, match="nearly single-axis"):
        solve_opencv_eye_to_hand(base_T_ee, camera_T_board, method="shah")


def write_observation_archive(path, *, poses, indices, translation_count=2) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=poses,
        camera_T_board=poses,
        candidate_indices=np.asarray(indices),
        translation_candidate_count=np.asarray(translation_count),
        rotation_candidate_count=np.asarray(2),
    )


def test_observation_archive_rejects_duplicate_candidate_indices(tmp_path) -> None:
    path = tmp_path / "samples.npz"
    write_observation_archive(
        path,
        poses=np.repeat(np.eye(4)[None], 3, axis=0),
        indices=[0, 1, 1],
    )

    with pytest.raises(ValueError, match="candidate_indices are invalid"):
        load_opencv_observations(path)


def test_observation_archive_rejects_out_of_range_candidate_index(tmp_path) -> None:
    path = tmp_path / "samples.npz"
    write_observation_archive(
        path,
        poses=np.repeat(np.eye(4)[None], 3, axis=0),
        indices=[0, 1, 4],
    )

    with pytest.raises(ValueError, match="candidate_indices are invalid"):
        load_opencv_observations(path)


def test_observation_archive_rejects_non_rigid_pose(tmp_path) -> None:
    path = tmp_path / "samples.npz"
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, 0, 0] = 2.0
    write_observation_archive(path, poses=poses, indices=[0, 1, 2])

    with pytest.raises(ValueError, match="invalid rotations"):
        load_opencv_observations(path)


@pytest.mark.parametrize("dtype", [complex, bool, str])
def test_opencv_backend_rejects_non_real_or_nonnumeric_pose_types(dtype):
    data = np.repeat(np.eye(4)[None], 4, axis=0).astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        solve_opencv_eye_to_hand(data, data)


def test_method_metadata_matches_opencv_enum():
    import cv2

    from kinematic_calibration.handeye.opencv import OPENCV_ROBOT_WORLD_METHODS

    assert OPENCV_ROBOT_WORLD_METHODS == {
        'shah': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        'li': cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
    }
