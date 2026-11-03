from itertools import permutations

import numpy as np
import pytest

pytest.importorskip("cv2")

from kinematic_calibration.handeye.opencv import evaluate_eye_to_hand_solution


def _pose(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


@pytest.mark.parametrize("strided", [False, True])
def test_closure_residuals_match_independent_errors_and_preserve_inputs(strided):
    # Exact signed-permutation rotations keep frame construction independent of
    # a calibration solver and make the duplicated worst observations identical.
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    base_T_camera = _pose(ry, [0.5, -0.25, 1.0])
    ee_T_board = _pose(rz, [0.125, -0.25, 0.0625])
    camera_T_base = _pose(ry.T, -ry.T @ base_T_camera[:3, 3])
    base_T_ee = np.stack(
        [
            _pose(rz, [0.25, 0.5, -0.125]),
            _pose(ry, [-0.125, 0.25, 0.5]),
            _pose(rx, [0.5, -0.125, 0.25]),
            _pose(rx, [0.5, -0.125, 0.25]),
        ]
    )
    angles = [0.25, 0.125, 0.5, 0.5]
    translations = np.array([[3, 4, 0], [0, 0, 0], [0, -5, 12], [0, -5, 12]]) / 64
    closure_errors = np.stack(
        [
            _pose(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                translation,
            )
            for angle, translation in zip(angles, translations, strict=True)
        ]
    )
    camera_T_board = camera_T_base @ base_T_ee @ ee_T_board @ closure_errors
    if strided:
        robot_storage = np.empty((2 * len(base_T_ee), 4, 4))
        board_storage = np.empty_like(robot_storage)
        robot_storage[::2] = base_T_ee
        board_storage[::2] = camera_T_board
        base_T_ee = robot_storage[::2]
        camera_T_board = board_storage[::2]
        assert not base_T_ee.flags.c_contiguous
        assert not camera_T_board.flags.c_contiguous
    inputs = (base_T_ee, camera_T_board, base_T_camera, ee_T_board)
    snapshots = [value.copy() for value in inputs]
    for value in inputs:
        value.setflags(write=False)

    result = evaluate_eye_to_hand_solution(*inputs)

    assert set(result) == {
        "position_errors_m",
        "orientation_errors_rad",
        "position_error_mean_m",
        "position_error_max_m",
        "orientation_error_mean_rad",
        "orientation_error_max_rad",
    }
    expected_positions = np.array([5, 0, 13, 13]) / 64
    np.testing.assert_allclose(
        result["position_errors_m"], expected_positions, rtol=0, atol=1e-15
    )
    # Trace/arccos magnifies library rounding at small angles (about 3e-15 rad
    # on NumPy 1.22 for the 0.125-rad sample, including the scalar evaluator).
    np.testing.assert_allclose(
        result["orientation_errors_rad"], angles, rtol=0, atol=1e-14
    )
    assert result["position_error_mean_m"] == pytest.approx(31 / 256, abs=1e-15)
    assert result["position_error_max_m"] == pytest.approx(13 / 64, abs=1e-15)
    assert result["orientation_error_mean_rad"] == pytest.approx(0.34375, abs=1e-14)
    assert result["orientation_error_max_rad"] == pytest.approx(0.5, abs=1e-14)
    for key in ("position_errors_m", "orientation_errors_rad"):
        values = result[key]
        assert values.dtype == np.float64
        assert values.shape == (4,)
        np.testing.assert_array_equal(np.flatnonzero(values == np.max(values)), [2, 3])
        assert np.argmax(values) == 2
    for actual, expected in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable


def test_accepted_near_rigid_transforms_keep_general_inverse_residuals():
    # The public tolerances accept these tiny scale deviations. Their inverse
    # is reciprocal scaling, so replacing it with a rigid transpose is incorrect.
    robot_scale, mount_scale = 1 + 2e-7, 1 - 1e-7
    camera_scale, board_scale = 1 + 1e-7, 1 - 2e-7
    base_T_ee = np.repeat(np.eye(4)[None], 3, axis=0)
    camera_T_board = base_T_ee.copy()
    ee_T_board = np.eye(4)
    base_T_camera = np.eye(4)
    base_T_ee[:, 0, 0] = robot_scale
    camera_T_board[:, 0, 0] = board_scale
    ee_T_board[0, 0] = mount_scale
    base_T_camera[0, 0] = camera_scale
    robot_x = np.array([0.125, 0.25, -0.375])
    board_x = np.array([0.25, -0.125, 0.5])
    mount_x, camera_x = 0.0625, -0.125
    base_T_ee[:, 0, 3] = robot_x
    camera_T_board[:, 0, 3] = board_x
    ee_T_board[0, 3] = mount_x
    base_T_camera[0, 3] = camera_x
    lhs_scale = robot_scale * mount_scale
    expected_positions = np.abs(
        (camera_x + camera_scale * board_x - robot_x - robot_scale * mount_x)
        / lhs_scale
    )
    residual_scale = camera_scale * board_scale / lhs_scale
    expected_angle = np.arccos((residual_scale + 1) / 2)

    result = evaluate_eye_to_hand_solution(
        base_T_ee, camera_T_board, base_T_camera, ee_T_board
    )

    np.testing.assert_allclose(
        result["position_errors_m"], expected_positions, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        result["orientation_errors_rad"], expected_angle, rtol=0, atol=1e-11
    )
    assert result["position_error_max_m"] == pytest.approx(
        expected_positions[2], abs=1e-15
    )
    assert np.argmax(result["position_errors_m"]) == 2


def test_scalar_distances_preserve_order_of_nearly_tied_worst_samples():
    translations = np.array(
        list(
            permutations(
                [-4.180422279378624, -3.3593128893899204e-5, -4.104663570938061e-5]
            )
        )
    )
    base_T_ee = np.repeat(np.eye(4)[None], len(translations), axis=0)
    camera_T_board = base_T_ee.copy()
    camera_T_board[:, :3, 3] = translations
    # These permutations have equal analytic lengths. The established scalar
    # norm can distinguish them by one ulp on some NumPy/BLAS builds.
    expected = np.asarray([np.linalg.norm(row) for row in translations])

    result = evaluate_eye_to_hand_solution(
        base_T_ee, camera_T_board, np.eye(4), np.eye(4)
    )

    np.testing.assert_array_equal(result["position_errors_m"], expected)
    assert np.argmax(result["position_errors_m"]) == np.argmax(expected)
    np.testing.assert_array_equal(
        np.flatnonzero(
            result["position_errors_m"] == np.max(result["position_errors_m"])
        ),
        np.flatnonzero(expected == np.max(expected)),
    )


def test_residual_validation_keeps_error_precedence():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    robot = poses.astype(complex)
    board = poses.copy()
    board[0, 0, 3] = np.nan
    camera = np.eye(3)
    mount = np.full((4, 4), np.nan)
    with pytest.raises(ValueError, match="base_T_ee must contain real numeric"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
    robot = poses
    with pytest.raises(ValueError, match="camera_T_board contains non-finite"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
    board = np.repeat(np.eye(4)[None], 4, axis=0)
    with pytest.raises(ValueError, match="sample counts must match"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
    board = poses
    with pytest.raises(ValueError, match="Base_T_Camera must be a 4x4"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
    camera = np.eye(4)
    camera[3, 0] = 0.01
    with pytest.raises(ValueError, match="Base_T_Camera is not a valid rigid"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
    camera = np.eye(4)
    with pytest.raises(ValueError, match="EE_T_Board is not a valid rigid"):
        evaluate_eye_to_hand_solution(robot, board, camera, mount)
