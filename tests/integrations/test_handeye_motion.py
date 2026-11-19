"""Motion excitation keeps OpenCV's conventions and scalar diagnostics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calibration.handeye import opencv

cv = pytest.importorskip("cv2")


def scalar_motion(poses):
    """Retained per-pose Rodrigues calculation and independent NumPy rank."""
    poses = np.asarray(poses, dtype=np.float64)
    vectors = np.asarray(
        [
            cv.Rodrigues(poses[0, :3, :3].T @ pose[:3, :3])[0].reshape(3)
            for pose in poses[1:]
        ]
    )
    singular_values = np.linalg.svd(vectors, compute_uv=False)
    largest, second = map(float, singular_values[:2])
    span = poses[:, :3, 3].max(axis=0) - poses[:, :3, 3].min(axis=0)
    return {
        "motion_rotation_rank": int(np.linalg.matrix_rank(vectors)),
        "rotation_singular_value_1_rad": largest,
        "rotation_singular_value_2_rad": second,
        "rotation_direction_ratio": second / largest if largest > 0 else 0.0,
        "max_rotation_from_first_rad": float(np.linalg.norm(vectors, axis=1).max()),
        "translation_span_x_m": float(span[0]),
        "translation_span_y_m": float(span[1]),
        "translation_span_z_m": float(span[2]),
        "translation_span_norm_m": float(np.linalg.norm(span)),
    }


def layout_view(poses, layout):
    if layout == "F":
        return np.asfortranarray(poses)
    if layout == "strided":
        storage = np.empty((2 * len(poses), 4, 4))
        storage[::2] = poses
        return storage[::2]
    if layout == "negative":
        return poses[::-1].copy()[::-1]
    return poses.copy()


@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_motion_has_analytic_excitation_and_preserves_readonly_inputs(layout):
    quarter_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    quarter_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    reference = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, :3] = reference @ np.stack(
        [np.eye(3), quarter_x, quarter_y, quarter_x]
    )
    poses[:, :3, 3] = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 1], [0, 1, -2]])
    poses = layout_view(poses, layout)
    before = poses.copy()
    poses.setflags(write=False)

    actual = opencv.measure_handeye_motion(poses)

    assert actual == scalar_motion(poses)
    assert actual["motion_rotation_rank"] == 2
    assert actual["rotation_singular_value_1_rad"] == pytest.approx(
        np.pi / np.sqrt(2), abs=1e-14
    )
    assert actual["rotation_singular_value_2_rad"] == pytest.approx(
        np.pi / 2, abs=1e-14
    )
    assert actual["rotation_direction_ratio"] == pytest.approx(1 / np.sqrt(2))
    assert actual["max_rotation_from_first_rad"] == pytest.approx(np.pi / 2)
    assert [actual[f"translation_span_{axis}_m"] for axis in "xyz"] == [2, 2, 5]
    assert actual["translation_span_norm_m"] == pytest.approx(np.sqrt(33))
    np.testing.assert_array_equal(poses, before)
    assert not poses.flags.writeable


@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_motion_matches_scalar_near_zero_pi_and_for_duplicate_rows(layout):
    angles = np.array([0, 1e-8, 1e-4, 0.3, np.pi - 1e-8, np.pi, np.pi + 1e-8])
    axes = np.array([[1, 0, 0], [0, 1, 0], [1, 1, -1]], dtype=float)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    rotvecs = (angles[:, None, None] * axes).reshape(-1, 3)
    rotations = Rotation.from_rotvec(rotvecs).as_matrix()
    poses = np.repeat(np.eye(4)[None], len(rotations) + 2, axis=0)
    poses[1:-1, :3, :3] = rotations
    poses[-1] = poses[5]
    poses = layout_view(poses, layout)

    assert opencv.measure_handeye_motion(poses) == scalar_motion(poses)


def test_tiny_rotations_retain_opencv_zero_angle_convention():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1:, :3, :3] = Rotation.from_rotvec([[1e-7, 0, 0], [0, 1e-7, 0]]).as_matrix()

    actual = opencv.measure_handeye_motion(poses)

    assert actual == scalar_motion(poses)
    assert actual["motion_rotation_rank"] == 0
    assert actual["rotation_singular_value_1_rad"] == 0
    assert actual["rotation_singular_value_2_rad"] == 0
    assert actual["max_rotation_from_first_rad"] == 0


def test_motion_derives_rank_and_spectrum_from_one_factorization(monkeypatch):
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[1:, :3, :3] = Rotation.from_rotvec(np.eye(3) * 0.2).as_matrix()
    factorization = np.linalg.svd
    factored_shapes = []

    def tracked_svd(matrix, *args, **kwargs):
        factored_shapes.append(matrix.shape)
        return factorization(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    actual = opencv.measure_handeye_motion(poses)

    assert actual["motion_rotation_rank"] == 3
    assert factored_shapes == [(3, 3)]


@pytest.mark.parametrize("method", ["shah", "li"])
@pytest.mark.parametrize("gate", ["span", "secondary", "ratio"])
@pytest.mark.parametrize("factor", [0.99, 1.01])
def test_excitation_gates_keep_the_same_acceptance_boundary(
    monkeypatch, method, gate, factor
):
    largest = 0.2 if gate == "ratio" else 0.05
    if gate == "span":
        largest = 0.001 * factor
        second = largest
    else:
        second = (0.002 if gate == "ratio" else 0.001) * factor
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1:, :3, :3] = Rotation.from_rotvec(
        [[largest, 0, 0], [0, second, 0]]
    ).as_matrix()
    camera = np.eye(4)
    camera[:3, 3] = [0.3, -0.2, 0.8]
    target = np.eye(4)
    target[:3, 3] = [0.03, -0.02, 0.1]
    board = np.linalg.inv(camera) @ poses @ target
    solver_calls = []

    def known_solution(*args, **kwargs):
        # Isolate the excitation gate from native-method conditioning at the
        # three-pose minimum. General SHAH/LI solves are tested independently.
        solver_calls.append(kwargs["method"])
        return target[:3, :3], target[:3, 3], camera[:3, :3], camera[:3, 3]

    monkeypatch.setattr(cv, "calibrateRobotWorldHandEye", known_solution)

    if factor < 1:
        message = (
            "independent rotation directions"
            if gate == "span"
            else "nearly single-axis"
        )
        with pytest.raises(ValueError, match=message):
            opencv.solve_opencv_eye_to_hand(poses, board, method=method)
        assert not solver_calls
    else:
        result = opencv.solve_opencv_eye_to_hand(poses, board, method=method)
        assert result.motion_rotation_rank == 2
        np.testing.assert_allclose(result.base_T_camera, camera, atol=1e-7)
        np.testing.assert_allclose(result.ee_T_board, target, atol=1e-7)
        assert solver_calls == [opencv.OPENCV_ROBOT_WORLD_METHODS[method]]


def test_solver_validates_each_observation_array_once(monkeypatch):
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(
        np.random.default_rng(19).normal(0, 0.3, (8, 3))
    ).as_matrix()
    validator = opencv._validate_pose_batch
    checked_names = []

    def tracked_validation(name, values):
        checked_names.append(name)
        return validator(name, values)

    monkeypatch.setattr(opencv, "_validate_pose_batch", tracked_validation)
    result = opencv.solve_opencv_eye_to_hand(poses, poses)

    assert result.motion_rotation_rank == 3
    assert checked_names == ["base_T_ee", "camera_T_board"]


@pytest.mark.parametrize("method", ["shah", "li"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_array_inputs_match_list_inputs_for_readonly_pose_views(
    method, layout, dtype
):
    rng = np.random.default_rng(20261119)
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.4, (20, 3))).as_matrix()
    poses[:, :3, 3] = rng.uniform(-0.4, 0.4, (20, 3))
    camera, target = np.eye(4), np.eye(4)
    camera[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    camera[:3, 3] = [0.3, -0.2, 0.8]
    target[:3, 3] = [0.03, -0.02, 0.1]
    board = np.linalg.inv(camera) @ poses @ target
    poses, board = [
        layout_view(value.astype(dtype), layout) for value in (poses, board)
    ]
    originals = [value.copy() for value in (poses, board)]
    for value in (poses, board):
        value.setflags(write=False)
    converted = [np.asarray(value, dtype=float) for value in (poses, board)]
    rotations_and_translations = [
        [pose[:3, :3] for pose in converted[0]],
        [pose[:3, 3] for pose in converted[0]],
        [pose[:3, :3] for pose in converted[1]],
        [pose[:3, 3] for pose in converted[1]],
    ]
    old = cv.calibrateRobotWorldHandEye(
        *rotations_and_translations, method=opencv.OPENCV_ROBOT_WORLD_METHODS[method]
    )

    actual = opencv.solve_opencv_eye_to_hand(poses, board, method=method)

    np.testing.assert_array_equal(actual.ee_T_board[:3, :3], old[0])
    np.testing.assert_array_equal(actual.ee_T_board[:3, 3], old[1].reshape(3))
    np.testing.assert_array_equal(actual.base_T_camera[:3, :3], old[2])
    np.testing.assert_array_equal(actual.base_T_camera[:3, 3], old[3].reshape(3))
    expected_motion = scalar_motion(poses)
    for key in expected_motion:
        if hasattr(actual, key):
            assert getattr(actual, key) == expected_motion[key]
    for value, original in zip((poses, board), originals, strict=True):
        np.testing.assert_array_equal(value, original)
        assert not value.flags.writeable


def test_solve_report_reuses_the_solver_motion_diagnostics(tmp_path, monkeypatch):
    import json

    from kinematic_calibration import handeye_cli

    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(
        np.random.default_rng(33).normal(0, 0.4, (8, 3))
    ).as_matrix()
    source, output = tmp_path / "observations.npz", tmp_path / "result.json"
    np.savez(
        source,
        schema_version=1,
        archive_type="eye-to-hand-observations",
        base_T_ee=poses,
        camera_T_board=poses,
        candidate_indices=np.arange(8),
        translation_candidate_count=4,
        rotation_candidate_count=4,
    )
    monkeypatch.setattr(
        handeye_cli,
        "measure_handeye_motion",
        lambda *args: pytest.fail("solve report repeated motion analysis"),
    )

    assert (
        handeye_cli.main(["solve", "--samples", str(source), "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text())
    expected = scalar_motion(poses)
    for key in expected:
        if key in report:
            assert report[key] == expected[key]
