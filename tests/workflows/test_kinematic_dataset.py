from pathlib import Path

import numpy as np
import pytest

from kinematic_calib.cli import main
from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    camera_target_to_base_ee,
    minimum_full_pose_samples,
    read_dataset_schema_version,
)


def make_transform(translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


def test_dataset_round_trip(tmp_path: Path) -> None:
    poses = np.stack([make_transform(), make_transform((0.1, 0.2, 0.3))])
    dataset = KinematicDataset(
        np.zeros((2, 6)), poses, ObservationFrame.CAMERA_T_TARGET
    )
    path = tmp_path / "observations.npz"

    dataset.save(path)
    loaded = KinematicDataset.load(path)

    np.testing.assert_array_equal(loaded.joint_positions, dataset.joint_positions)
    np.testing.assert_array_equal(loaded.observed_poses, dataset.observed_poses)
    np.testing.assert_array_equal(loaded.sample_indices, [0, 1])
    assert loaded.observation_frame == ObservationFrame.CAMERA_T_TARGET


def test_dataset_preserves_source_sample_indices(tmp_path: Path) -> None:
    path = tmp_path / "observations.npz"
    dataset = KinematicDataset(
        np.zeros((2, 1)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
        sample_indices=np.array([7, 12]),
    )

    dataset.save(path)

    np.testing.assert_array_equal(KinematicDataset.load(path).sample_indices, [7, 12])


def test_dataset_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(999),
        joint_positions=np.zeros((2, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=np.asarray("camera_T_target"),
    )

    with pytest.raises(ValueError, match="schema version"):
        KinematicDataset.load(path)


@pytest.mark.parametrize("schema_version", [None, 1, 2], ids=["legacy", "v1", "v2"])
def test_dataset_loads_supported_schemas_without_sample_indices(
    tmp_path: Path, schema_version: int | None
) -> None:
    path = tmp_path / "observations.npz"
    metadata = (
        {} if schema_version is None else {"schema_version": np.asarray(schema_version)}
    )
    np.savez_compressed(
        path,
        joint_positions=np.zeros((2, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=np.asarray("camera_T_target"),
        **metadata,
    )
    original = path.read_bytes()

    assert read_dataset_schema_version(path) == schema_version
    loaded = KinematicDataset.load(path)

    np.testing.assert_array_equal(loaded.sample_indices, [0, 1])
    assert loaded.observation_frame == ObservationFrame.CAMERA_T_TARGET
    assert path.read_bytes() == original


@pytest.mark.parametrize("entrypoint", ["schema", "dataset", "inspect"])
@pytest.mark.parametrize(
    "case", ["raw_npy", "empty", "truncated", "bad_crc", "positive_inf", "negative_inf"]
)
def test_invalid_dataset_reports_input_error_without_modifying_source(
    tmp_path: Path, caplog, capsys, case: str, entrypoint: str
) -> None:
    path = tmp_path / f"{case}.npz"
    if case == "raw_npy":
        with path.open("wb") as stream:
            np.save(stream, np.zeros((2, 1)), allow_pickle=False)
    elif case == "empty":
        path.write_bytes(b"")
    elif case == "truncated":
        path.write_bytes(b"PK\x03\x04")
    elif case == "bad_crc":
        np.savez(path, schema_version=np.asarray(2))
        content = bytearray(path.read_bytes())
        # Alter stored array bytes while retaining the original ZIP checksum.
        content[content.index(b"\x93NUMPY") + 6] ^= 1
        path.write_bytes(content)
    else:
        np.savez_compressed(
            path,
            schema_version=np.asarray(np.inf if case == "positive_inf" else -np.inf),
        )
    original = path.read_bytes()

    try:
        if entrypoint == "inspect":
            assert main(["inspect", "--dataset", str(path)]) == 2
            captured = capsys.readouterr()
            assert captured.out == ""
            message = caplog.text + captured.err
            assert "Traceback" not in message
        else:
            reader = (
                read_dataset_schema_version
                if entrypoint == "schema"
                else KinematicDataset.load
            )
            with pytest.raises(ValueError) as error:
                reader(path)
            message = str(error.value)

        if case in {"positive_inf", "negative_inf"}:
            assert "dataset schema_version must be an integer" in message
        else:
            assert "NPZ" in message
            assert str(path) in message
            if case == "bad_crc":
                assert "CRC" in message
    finally:
        assert path.read_bytes() == original


def test_empty_npz_retains_legacy_schema_and_missing_fields_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty_archive.npz"
    np.savez_compressed(path)
    original = path.read_bytes()

    assert read_dataset_schema_version(path) is None
    with pytest.raises(ValueError, match="dataset is missing fields"):
        KinematicDataset.load(path)
    assert path.read_bytes() == original


def test_camera_target_conversion_uses_both_extrinsics() -> None:
    camera_t_target = np.stack([make_transform((1.0, 2.0, 3.0))])
    base_t_camera = make_transform((0.5, 0.0, 0.0))
    ee_t_target = make_transform((0.0, 0.0, 0.2))

    converted = camera_target_to_base_ee(camera_t_target, base_t_camera, ee_t_target)

    np.testing.assert_allclose(converted[0, :3, 3], [1.5, 2.0, 2.8])


def test_batch_camera_conversion_preserves_frame_equation_near_pi() -> None:
    from scipy.spatial.transform import Rotation

    expected = np.repeat(np.eye(4)[None], 4, axis=0)
    expected[:, :3, :3] = Rotation.from_rotvec(
        [[0, 0, 0], [np.pi - 1e-10, 0, 0], [0, -np.pi, 0], [0.2, 0.3, -0.4]]
    ).as_matrix()
    expected[:, :3, 3] = np.arange(12).reshape(4, 3) / 10
    base_t_camera = make_transform((0.3, -0.4, 0.2))
    base_t_camera[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.1, 0.5]).as_matrix()
    ee_t_target = make_transform((0.05, 0.02, -0.1))
    ee_t_target[:3, :3] = Rotation.from_euler("xyz", [-0.3, 0.2, 0.1]).as_matrix()
    observed = np.linalg.inv(base_t_camera) @ expected @ ee_t_target
    original = observed.copy()

    recovered = camera_target_to_base_ee(observed, base_t_camera, ee_t_target)

    np.testing.assert_allclose(recovered, expected, atol=2e-15)
    np.testing.assert_allclose(
        recovered @ ee_t_target, base_t_camera @ observed, atol=2e-15
    )
    np.testing.assert_array_equal(observed, original)


def test_dataset_rejects_invalid_rotation() -> None:
    pose = np.eye(4)
    pose[0, 0] = 2.0

    with pytest.raises(ValueError, match="non-orthonormal"):
        KinematicDataset(np.zeros((1, 6)), pose[None], "base_T_ee")


def test_dataset_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError, match="sample counts"):
        KinematicDataset(np.zeros((2, 6)), np.eye(4)[None], ObservationFrame.BASE_T_EE)


def test_minimum_sample_count_matches_robotkinecal_parameterization() -> None:
    assert minimum_full_pose_samples(6) == 5
    assert minimum_full_pose_samples(7) == 6


def test_solver_preflight_rejects_too_few_samples() -> None:
    dataset = KinematicDataset(
        np.zeros((4, 6)),
        np.repeat(np.eye(4)[None], 4, axis=0),
        ObservationFrame.BASE_T_EE,
    )

    with pytest.raises(ValueError, match="at least 5"):
        dataset.validate_solver_sample_count()


def test_solver_preflight_does_not_count_duplicate_rows() -> None:
    joints = np.zeros((5, 6))
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[1:, 0, 3] = np.arange(1, 5) * 0.01
    poses[4] = poses[3]
    dataset = KinematicDataset(joints, poses, ObservationFrame.BASE_T_EE)

    with pytest.raises(ValueError, match="got 4 unique samples from 5 rows"):
        dataset.validate_solver_sample_count()


def test_dataset_summary_reports_joint_and_pose_coverage() -> None:
    poses = np.stack([make_transform(), make_transform((0.1, 0.2, 0.3))])
    joints = np.asarray([[0.0, -0.5], [1.0, 0.5]])
    dataset = KinematicDataset(joints, poses, ObservationFrame.BASE_T_EE)

    summary = dataset.get_summary()

    assert summary["samples"] == 2
    assert summary["dof"] == 2
    assert summary["joint_span_rad"] == [1.0, 1.0]
    assert summary["translation_span_m"] == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize("dof", [1, 2, 3, 6, 7, 8, 12])
def test_dataset_contract_supports_arbitrary_positive_dof(dof: int) -> None:
    sample_count = minimum_full_pose_samples(dof)
    joints = np.zeros((sample_count, dof))
    joints[:, 0] = np.arange(sample_count) * 0.01
    dataset = KinematicDataset(
        joints,
        np.repeat(np.eye(4)[None], sample_count, axis=0),
        ObservationFrame.BASE_T_EE,
    )

    dataset.validate_solver_sample_count()
    assert dataset.joint_positions.shape == (sample_count, dof)
