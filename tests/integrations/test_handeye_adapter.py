from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")

from kinematic_calibration.handeye import (
    camera_observations_to_base_ee,
    solve_eye_to_hand,
    solve_toolkit_eye_to_hand,
)


def pose(rotvec, xyz):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = xyz
    return value


@pytest.fixture
def observations():
    camera = pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    mount = pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    robot = np.stack(
        [
            pose(
                [0.08 * i, -0.05 * i**2, 0.04 * (i + 1)],
                [0.1 * np.sin(i), 0.08 * np.cos(i), 0.35 + 0.03 * i],
            )
            for i in range(1, 10)
        ]
    )
    board = np.linalg.inv(camera) @ robot @ mount
    return robot, board, camera, mount


@pytest.mark.parametrize("method", ["shah", "li"])
def test_unified_opencv_solves_exact_dataset(observations, method):
    robot, board, camera, mount = observations
    result = solve_eye_to_hand(robot, board, method=method)
    np.testing.assert_allclose(result.base_T_camera, camera, atol=1e-7)
    np.testing.assert_allclose(result.ee_T_board, mount, atol=1e-7)
    assert result.backend == "opencv"
    assert result.position_error_max_m < 1e-8
    assert result.orientation_error_max_rad < 1e-8
    np.testing.assert_allclose(
        camera_observations_to_base_ee(board, camera, mount), robot, atol=1e-12
    )


def test_toolkit_adapter_respects_input_order_and_frame_direction(observations):
    robot, board, camera, mount = observations

    def native_solver(camera_poses, robot_poses, eye_in_hand):
        np.testing.assert_allclose(camera_poses, board)
        np.testing.assert_allclose(robot_poses, robot)
        assert eye_in_hand is False
        return SimpleNamespace(status=0, pose_calibrate=camera)

    result = solve_toolkit_eye_to_hand(robot, board, solver=native_solver)
    np.testing.assert_allclose(result.ee_T_board, mount, atol=1e-12)
    assert result.orientation_error_max_rad < 1e-12
    assert result.method == "axxb"


def test_toolkit_failure_and_corrupt_result_are_rejected(observations):
    robot, board, _, _ = observations
    with pytest.raises(RuntimeError, match="status"):
        solve_toolkit_eye_to_hand(
            robot, board, solver=lambda *a: SimpleNamespace(status=1)
        )
    with pytest.raises(ValueError, match="rigid transforms"):
        solve_toolkit_eye_to_hand(
            robot,
            board,
            solver=lambda *a: SimpleNamespace(
                status=0, pose_calibrate=np.full((4, 4), np.nan)
            ),
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j])
def test_non_real_or_nonfinite_data_is_rejected(observations, bad):
    robot, board, _, _ = observations
    robot = robot.astype(complex if isinstance(bad, complex) else float)
    robot[0, 0, 3] = bad
    with pytest.raises(ValueError):
        solve_eye_to_hand(robot, board)


def test_degenerate_toolkit_dataset_rejected_before_native_call():
    data = np.repeat(np.eye(4)[None], 4, axis=0)
    with pytest.raises(ValueError, match="independent rotation"):
        solve_toolkit_eye_to_hand(
            data, data, solver=lambda *a: pytest.fail("must reject before native call")
        )


def test_optional_native_toolkit_recovers_known_camera(observations):
    pytest.importorskip("handeye_calibration")
    robot, board, camera, mount = observations
    result = solve_eye_to_hand(robot, board, backend="toolkit")
    np.testing.assert_allclose(result.base_T_camera, camera, atol=2e-7)
    np.testing.assert_allclose(result.ee_T_board, mount, atol=2e-7)
