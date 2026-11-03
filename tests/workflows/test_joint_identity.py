"""Joint-column identity survives interchange and is checked before fitting."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib import FitOptions, KinematicDataset, ObservationFrame, fit_dataset
from kinematic_calib.cli import main


def _dataset(names=("shoulder", "elbow"), count=12):
    q = np.random.default_rng(17).uniform(-1, 1, (count, 2))
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = Rotation.from_euler("z", np.linspace(0, 1, count)).as_matrix()
    poses[:, 0, 3] = np.linspace(0, 0.3, count)
    return KinematicDataset(
        q, poses, ObservationFrame.CAMERA_T_TARGET, np.arange(100, 100 + count), names
    )


def test_named_dataset_schema_two_round_trip_and_frame_conversion(tmp_path):
    original = _dataset()
    path = tmp_path / "named.npz"
    original.save(path)
    loaded = KinematicDataset.load(path)
    assert loaded.joint_names == ("shoulder", "elbow")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == 2
        assert archive["joint_names"].dtype.kind == "U"
    body = loaded.as_robot_body(np.eye(4), np.eye(4))
    assert body.joint_names == original.joint_names
    assert body.get_summary()["joint_names"] == list(original.joint_names)
    assert body.get_summary()["joint_names_present"]


@pytest.mark.parametrize(
    "names",
    [
        "shoulder",
        ["one"],
        ["same", "same"],
        ["", "elbow"],
        [" ", "elbow"],
        [1, 2],
        [True, False],
        [b"shoulder", b"elbow"],
        [["shoulder", "elbow"]],
    ],
)
def test_invalid_joint_column_names_are_rejected(names):
    with pytest.raises(ValueError, match="joint_names"):
        _dataset(names)


@pytest.mark.parametrize("version", [None, 1])
def test_legacy_unnamed_dataset_remains_loadable_and_explicitly_unverified(
    tmp_path, version
):
    path = tmp_path / "legacy.npz"
    data = _dataset()
    arrays = dict(
        joint_positions=data.joint_positions,
        observed_poses=data.observed_poses,
        observation_frame=np.asarray(data.observation_frame.value),
        sample_indices=data.sample_indices,
    )
    if version is not None:
        arrays["schema_version"] = np.asarray(version)
    np.savez_compressed(path, **arrays)
    loaded = KinematicDataset.load(path)
    assert loaded.joint_names is None
    diagnostic = loaded.check_joint_identity(["shoulder", "elbow"])
    assert not diagnostic["verified"]
    assert "unverified" in diagnostic["reason"]
    # Upgrading records what is known and does not invent joint labels.
    loaded.save(tmp_path / "upgraded.npz")
    assert KinematicDataset.load(tmp_path / "upgraded.npz").joint_names is None


def test_original_four_positional_arguments_keep_sample_index_meaning():
    source = _dataset()
    legacy = KinematicDataset(
        source.joint_positions,
        source.observed_poses,
        source.observation_frame,
        [8 + i for i in range(12)],
    )
    assert legacy.joint_names is None
    assert legacy.sample_indices[0] == 8


def test_named_dataset_never_reorders_columns_to_match_a_model():
    data = _dataset()
    with pytest.raises(ValueError, match="never automatically reordered"):
        data.check_joint_identity(["elbow", "shoulder"])
    with pytest.raises(ValueError, match="requires model joint_names"):
        data.check_joint_identity(None)
    assert data.check_joint_identity(["shoulder", "elbow"])["verified"]


def test_split_dedup_and_upgrade_preserve_names(tmp_path):
    data = _dataset()
    source = tmp_path / "source.npz"
    data.save(source)
    train, validation, split = (
        tmp_path / "train.npz",
        tmp_path / "validation.npz",
        tmp_path / "split.json",
    )
    assert (
        main(
            [
                "split-dataset",
                "--dataset",
                str(source),
                "--train-output",
                str(train),
                "--validation-output",
                str(validation),
                "--manifest",
                str(split),
            ]
        )
        == 0
    )
    for path in (train, validation):
        assert KinematicDataset.load(path).joint_names == data.joint_names
    assert (
        main(
            [
                "verify-split",
                "--source",
                str(source),
                "--train",
                str(train),
                "--validation",
                str(validation),
                "--manifest",
                str(split),
            ]
        )
        == 0
    )
    duplicate = tmp_path / "duplicates.npz"
    KinematicDataset(
        np.vstack([data.joint_positions, data.joint_positions[:1]]),
        np.concatenate([data.observed_poses, data.observed_poses[:1]]),
        data.observation_frame,
        joint_names=data.joint_names,
    ).save(duplicate)
    cleaned, manifest = tmp_path / "cleaned.npz", tmp_path / "dedup.json"
    assert (
        main(
            [
                "deduplicate-dataset",
                "--dataset",
                str(duplicate),
                "--output",
                str(cleaned),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert KinematicDataset.load(cleaned).joint_names == data.joint_names
    upgraded = tmp_path / "upgraded.npz"
    assert (
        main(
            [
                "upgrade-dataset",
                "--dataset",
                str(cleaned),
                "--output",
                str(upgraded),
                "--manifest",
                str(tmp_path / "upgrade.json"),
            ]
        )
        == 0
    )
    assert KinematicDataset.load(upgraded).joint_names == data.joint_names


def test_service_rejects_wrong_order_before_mutating_the_calibrator():
    from robotkinecal import SerialRobotKineCal

    axes = np.array([[0, 0, 0, 0, 0, 1], [-0.2, 0, 0.1, 0, 1, 0]], float)
    calibrator = SerialRobotKineCal.from_screw_axes(
        axes, np.eye(4), joint_names=["elbow", "shoulder"]
    )
    data = _dataset().as_robot_body(np.eye(4), np.eye(4))
    with pytest.raises(ValueError, match="joint_names order"):
        fit_dataset(data, calibrator, options=FitOptions())
    np.testing.assert_array_equal(calibrator.joint_screw_axis, axes)
    np.testing.assert_array_equal(calibrator.zero_conf_EE_pose.A, np.eye(4))


def test_numpy_solve_binds_named_training_and_validation_column_contract(tmp_path):
    from kinematic_calibration.backends import NumpyBackend
    from kinematic_calibration.experiment import example_urdf

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=81, translation_std_m=0.001, rotation_std_rad=0.001)
    q = np.random.default_rng(62).uniform(-1.5, 1.5, (42, 6))
    poses = np.asarray([truth.forward(row) for row in q])
    train, validation, parameters = (
        tmp_path / "train.npz",
        tmp_path / "validation.npz",
        tmp_path / "fit.npz",
    )
    KinematicDataset(
        q[:30], poses[:30], "base_T_ee", joint_names=nominal.joint_names
    ).save(train)
    KinematicDataset(
        q[30:], poses[30:], "base_T_ee", joint_names=nominal.joint_names
    ).save(validation)
    solve = [
        "solve",
        "--backend",
        "numpy",
        "--dataset",
        str(train),
        "--urdf",
        str(example_urdf()),
        "--ee-name",
        "ee_link",
        "--model-mode",
        "robot-body",
        "--output",
        str(parameters),
        "--quiet",
    ]
    assert main(solve) == 0
    report = json.loads(parameters.with_suffix(".json").read_text())
    assert report["joint_identity_verified"]
    assert report["joint_identity"]["dataset_joint_names"] == list(nominal.joint_names)
    validation_report = tmp_path / "validation.json"
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(validation),
                "--output",
                str(validation_report),
            ]
        )
        == 0
    )
    assert json.loads(validation_report.read_text())["joint_identity_verified"]
    # A consistent permutation of encoder values and their names is still an
    # explicit contract change. The workflow must not silently undo it.
    KinematicDataset(
        q[30:, ::-1], poses[30:], "base_T_ee", joint_names=nominal.joint_names[::-1]
    ).save(validation)
    assert (
        main(
            [
                "validate-result",
                "--parameters",
                str(parameters),
                "--dataset",
                str(validation),
            ]
        )
        == 2
    )
    KinematicDataset(
        q[:30, ::-1], poses[:30], "base_T_ee", joint_names=nominal.joint_names[::-1]
    ).save(train)
    assert main(solve + ["--force"]) == 2


def test_legacy_numpy_solve_logs_and_records_unverified_column_identity(
    tmp_path, caplog
):
    from kinematic_calibration.backends import NumpyBackend
    from kinematic_calibration.experiment import example_urdf

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    q = np.random.default_rng(13).uniform(-1.5, 1.5, (20, 6))
    data = tmp_path / "legacy.npz"
    KinematicDataset(
        q, np.asarray([nominal.forward(row) for row in q]), "base_T_ee"
    ).save(data)
    output = tmp_path / "fit.npz"
    assert (
        main(
            [
                "solve",
                "--backend",
                "numpy",
                "--dataset",
                str(data),
                "--urdf",
                str(example_urdf()),
                "--ee-name",
                "ee_link",
                "--model-mode",
                "robot-body",
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    report = json.loads(output.with_suffix(".json").read_text())
    assert not report["joint_identity_verified"]
    assert "unverified" in caplog.text
