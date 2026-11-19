"""Leave-one-out diagnostics reuse rotations while retaining fresh subset solves."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calibration import handeye_cli
from kinematic_calibration.handeye import opencv

cv = pytest.importorskip("cv2")


def poses(count, kind):
    rng = np.random.default_rng(20261119)
    values = np.repeat(np.eye(4)[None], count, axis=0)
    vectors = rng.normal(0, 0.5, (count, 3))
    if kind == "pi":
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        vectors *= np.pi - 1e-9
    elif kind == "critical":
        vectors[:, 1:] = 0
        vectors[-1] = [0, 0.4, 0]
    values[:, :3, :3] = Rotation.from_rotvec(vectors).as_matrix()
    values[:, :3, 3] = rng.normal(0, 0.3, (count, 3))
    return values


def readonly_view(values, layout):
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        storage = np.empty((len(values) * 2, 4, 4))
        storage[::2] = values
        values = storage[::2]
    elif layout == "negative":
        values = values[::-1].copy()[::-1]
    values.setflags(write=False)
    return values


@pytest.mark.parametrize("count", [4, 17])
@pytest.mark.parametrize("kind", ["random", "pi", "critical"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "negative"])
def test_cached_subset_motion_equals_fresh_motion_including_reference_removal(
    count, kind, layout
):
    values = readonly_view(poses(count, kind), layout)
    before = values.copy()
    vectors = opencv._relative_rotation_vectors(values[np.arange(count)], cv)
    saved_vectors = vectors.copy()
    for omitted in range(count):
        keep = np.arange(count) != omitted
        subset = values[keep]
        actual = opencv._measure_validated_motion(
            subset,
            cv,
            relative_rotation_vectors=vectors[keep[1:]] if omitted else None,
        )
        assert actual == opencv.measure_handeye_motion(subset)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(vectors, saved_vectors)
    assert not values.flags.writeable


@pytest.mark.parametrize("method", ["shah", "li"])
@pytest.mark.parametrize("kind", ["random", "critical"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_influence_reports_match_fresh_public_solves_and_use_linear_rodrigues_calls(
    tmp_path, monkeypatch, method, kind, layout
):
    count = 17
    values = readonly_view(poses(count, kind), layout)
    source, output = tmp_path / "observations.npz", tmp_path / "influence.json"
    np.savez(
        source,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=values,
        camera_T_board=values.copy(),
        candidate_indices=np.arange(count) + 100,
        translation_candidate_count=np.asarray(count + 100),
        rotation_candidate_count=np.asarray(0),
    )
    source_bytes = source.read_bytes()
    argv = [
        "influence", "--samples", str(source), "--output", str(output),
        "--method", method, "--force",
    ]
    native_subset_solver = handeye_cli._solve_validated_poses

    def fresh_subset_solver(ee, board, *, method, cv, excitation):
        return opencv.solve_opencv_eye_to_hand(ee, board, method=method)

    monkeypatch.setattr(handeye_cli, "_solve_validated_poses", fresh_subset_solver)
    assert handeye_cli.main(argv) == 0
    expected = json.loads(output.read_text())
    expected.pop("provenance")
    monkeypatch.setattr(handeye_cli, "_solve_validated_poses", native_subset_solver)
    rodrigues, validator = cv.Rodrigues, opencv._validate_pose_batch
    rotation_calls, validations = [], []

    def tracked_rodrigues(*args, **kwargs):
        rotation_calls.append(1)
        return rodrigues(*args, **kwargs)

    def tracked_validation(name, array):
        validations.append(name)
        return validator(name, array)

    monkeypatch.setattr(cv, "Rodrigues", tracked_rodrigues)
    monkeypatch.setattr(opencv, "_validate_pose_batch", tracked_validation)
    assert handeye_cli.main(argv) == 0
    actual = json.loads(output.read_text())
    actual.pop("provenance")
    assert actual == expected
    assert len(rotation_calls) == 3 * count - 4
    assert validations == ["base_T_ee", "camera_T_board"] * 2
    assert source.read_bytes() == source_bytes
    if kind == "critical":
        rejected = [
            row for row in actual["sample_influence"]
            if not row["reduced_solve_succeeded"]
        ]
        assert [row["sample_index"] for row in rejected] == [count - 1]
        assert "independent rotation directions" in rejected[0]["failure"]
