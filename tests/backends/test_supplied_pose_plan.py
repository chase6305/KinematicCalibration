"""Supplied joint configurations retain their model and source-row identities."""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset, ObservationFrame
from kinematic_calibration import planning
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.backends.playback import ExperimentPlayback
from kinematic_calibration.experiment import example_urdf, run_experiment
from kinematic_calibration.planning import (
    load_pose_plan,
    save_pose_plan,
    select_informative_poses,
)
from kinematic_calibration.simulation import MeasurementNoise


@pytest.fixture
def robot():
    return NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")


def rewrite_plan(path, *, arrays=None, metadata=None, remove_metadata=()):
    with np.load(path, allow_pickle=False) as archive:
        contents = {key: archive[key] for key in archive.files}
    contents.update(arrays or {})
    report = json.loads(contents["metadata_json"].item())
    report.update(metadata or {})
    for key in remove_metadata:
        report.pop(key)
    contents["metadata_json"] = np.asarray(json.dumps(report))
    np.savez_compressed(path, **contents)


@pytest.mark.parametrize("count", [None, 5, np.int64(5)])
def test_npy_default_keeps_every_row_and_skips_greedy_selection(
    tmp_path, monkeypatch, count
):
    rows = np.zeros((5, 6))
    rows[:, 0] = [0, 2, 1, -1, 2]
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, rows)
    source_bytes = source.read_bytes()
    monkeypatch.setattr(
        planning,
        "select_informative_poses",
        lambda *args, **kwargs: pytest.fail("all-input planning ran greedy selection"),
    )
    original_svd = np.linalg.svd
    diagnostic_calls = []

    def tracked_svd(matrix, *args, **kwargs):
        if kwargs.get("compute_uv") is False:
            diagnostic_calls.append(matrix.shape)
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    report = save_pose_plan(output, input_joints=source, count=count)
    plan = load_pose_plan(output)
    np.testing.assert_array_equal(plan.selected_candidate_indices, np.arange(5))
    np.testing.assert_array_equal(plan.candidate_indices, plan.visit_order)
    np.testing.assert_array_equal(
        plan.planned_joint_positions, rows[plan.candidate_indices]
    )
    np.testing.assert_array_equal(np.sort(plan.candidate_indices), np.arange(5))
    assert diagnostic_calls == [(30, 30)]
    assert source.read_bytes() == source_bytes
    assert report["candidate_count"] == report["selected_count"] == 5
    assert report["selection_method"] == "all-input"
    assert report["input_joints_format"] == "npy"
    assert report["input_joints_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert report["input_joint_identity_verified"] is False
    assert report["input_joint_names"] is None
    assert report["generation_seed_used"] is False
    assert plan.source_sample_indices is None
    assert report["total_distance_rad"] < report["selection_order_distance_rad"]
    for flag in (
        "measurements_acquired",
        "collision_checked",
        "visibility_checked",
        "trajectory_generated",
    ):
        assert report[flag] is False


@pytest.mark.parametrize("rows", [np.zeros((1, 6)), np.zeros((3, 6))])
def test_small_repeated_zero_configurations_are_valid_rank_deficient_plans(
    tmp_path, rows
):
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, rows)
    report = save_pose_plan(output, input_joints=source)
    plan = load_pose_plan(output)
    assert report["selected_count"] == len(rows)
    assert report["full_rank"] is False
    assert report["condition_number"] is None
    assert report["total_distance_rad"] == 0
    np.testing.assert_array_equal(plan.planned_joint_positions, rows)
    np.testing.assert_array_equal(plan.candidate_indices, np.arange(len(rows)))


@pytest.mark.parametrize("named", [True, False])
def test_npz_retains_sparse_sample_ids_and_never_reads_observation_arrays(
    tmp_path, robot, named
):
    rows = np.zeros((4, robot.dof))
    rows[:, 0] = [0, 2, 1, -1]
    ids = np.array([0, 900, 31, np.iinfo(np.int64).max], dtype=np.int64)
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    names = {"joint_names": np.asarray(robot.joint_names)} if named else {}
    np.savez_compressed(
        source,
        joint_positions=rows,
        sample_indices=ids,
        observed_poses=np.asarray([{"must": "remain unread"}], dtype=object),
        **names,
    )
    report = save_pose_plan(
        output,
        input_joints=source,
        start_configuration=np.zeros(robot.dof),
        return_to_start=True,
    )
    source.unlink()
    plan = load_pose_plan(output, backend=robot)
    assert plan.metadata == report
    assert report["input_joint_identity_verified"] is named
    assert report["joint_names"] == list(robot.joint_names)
    assert report["input_joint_names"] == (list(robot.joint_names) if named else None)
    np.testing.assert_array_equal(plan.source_sample_indices, ids)
    np.testing.assert_array_equal(
        plan.planned_joint_positions, rows[plan.candidate_indices]
    )
    np.testing.assert_array_equal(
        plan.source_sample_indices[plan.candidate_indices], ids[plan.visit_order]
    )
    assert plan.segment_distances_rad.shape == (5,)


def test_existing_dataset_archive_is_a_configuration_source(tmp_path, robot):
    source, output = tmp_path / "recorded.npz", tmp_path / "plan.npz"
    dataset = KinematicDataset(
        np.zeros((2, robot.dof)),
        np.repeat(np.eye(4)[None], 2, axis=0),
        ObservationFrame.CAMERA_T_TARGET,
        np.array([83, 12]),
        robot.joint_names,
    )
    dataset.save(source)
    save_pose_plan(output, input_joints=source)
    plan = load_pose_plan(output)
    np.testing.assert_array_equal(plan.planned_joint_positions, dataset.joint_positions)
    np.testing.assert_array_equal(plan.source_sample_indices, dataset.sample_indices)


def test_explicit_subset_uses_supplied_pool_and_keeps_row_offsets(tmp_path, robot):
    rows = np.random.default_rng(9).uniform(-2, 2, (20, robot.dof))
    ids = np.arange(20, dtype=np.int64) * 30 + 2000
    source, output = tmp_path / "pool.npz", tmp_path / "plan.npz"
    np.savez_compressed(source, joint_positions=rows, sample_indices=ids)
    expected = select_informative_poses(robot, rows, count=8)
    report = save_pose_plan(output, input_joints=source, count=8, seed=99)
    plan = load_pose_plan(output)
    assert report["candidate_count"] == 20
    assert report["selected_count"] == 8
    assert report["selection_method"] == "d-optimal"
    assert report["generation_seed_used"] is False
    np.testing.assert_array_equal(
        plan.selected_candidate_indices, expected.candidate_indices
    )
    np.testing.assert_array_equal(
        plan.planned_joint_positions, rows[plan.candidate_indices]
    )
    np.testing.assert_array_equal(plan.source_sample_indices, ids)
    np.testing.assert_array_equal(plan.singular_values, expected.singular_values)


@pytest.mark.parametrize(
    "rows",
    [
        np.empty((0, 6)),
        np.empty((2, 0)),
        np.zeros((2, 5)),
        np.zeros(6),
        np.zeros((2, 6), dtype=bool),
        np.zeros((2, 6), dtype=complex),
        np.full((2, 6), "zero"),
        np.full((2, 6), np.nan),
        np.full((2, 6), np.inf),
        np.full((2, 6), 10),
        np.full((2, 6), np.finfo(np.longdouble).max, dtype=np.longdouble),
    ],
)
def test_invalid_configurations_preserve_existing_output(tmp_path, rows):
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, rows)
    output.write_bytes(b"previous plan")
    with pytest.raises(ValueError):
        save_pose_plan(output, input_joints=source, force=True)
    assert output.read_bytes() == b"previous plan"


@pytest.mark.parametrize(
    "indices",
    [
        [-1, 0],
        [3, 3],
        [0.5, 1],
        [True, False],
        [0, np.inf],
        np.array([0, 2**63], dtype=np.uint64),
        [0.0, float(2**63)],
        ["0", "1"],
        [0j, 1j],
        [[0, 1]],
    ],
)
def test_invalid_input_sample_ids_are_rejected_before_writing(tmp_path, indices):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    np.savez_compressed(
        source, joint_positions=np.zeros((2, 6)), sample_indices=indices
    )
    with pytest.raises(ValueError, match="sample_indices"):
        save_pose_plan(output, input_joints=source)
    assert not output.exists()


@pytest.mark.parametrize("indices", [np.array([0, 20], dtype=np.uint64), [0.0, 20.0]])
def test_integral_sample_ids_normalize_to_int64(tmp_path, indices):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    np.savez_compressed(
        source, joint_positions=np.zeros((2, 6)), sample_indices=indices
    )
    save_pose_plan(output, input_joints=source)
    plan = load_pose_plan(output)
    assert plan.source_sample_indices.dtype == np.int64
    np.testing.assert_array_equal(plan.source_sample_indices, [0, 20])


@pytest.mark.parametrize("count", [0, -1, 3, True, 1.5])
def test_invalid_supplied_count_preserves_existing_output(tmp_path, count):
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, np.zeros((2, 6)))
    output.write_bytes(b"previous plan")
    with pytest.raises(ValueError, match="count"):
        save_pose_plan(output, input_joints=source, count=count, force=True)
    assert output.read_bytes() == b"previous plan"


def test_named_input_rejects_reordered_model_columns(tmp_path, robot):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    np.savez_compressed(
        source,
        joint_positions=np.zeros((2, robot.dof)),
        joint_names=np.asarray(robot.joint_names[::-1]),
    )
    with pytest.raises(ValueError, match="joint_names order does not match"):
        save_pose_plan(output, input_joints=source)
    assert not output.exists()


def test_npz_requires_configuration_key_and_never_creates_output(tmp_path):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    np.savez_compressed(source, positions=np.zeros((2, 6)))
    with pytest.raises(ValueError, match="missing joint_positions"):
        save_pose_plan(output, input_joints=source)
    assert not output.exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_force_cannot_overwrite_input_file_or_alias(tmp_path, alias):
    source = tmp_path / "chosen.npy"
    np.save(source, np.zeros((2, 6)))
    original = source.read_bytes()
    output = source if alias == "same" else tmp_path / "plan.npz"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="cannot replace its input"):
        save_pose_plan(output, input_joints=source, force=True)
    assert source.read_bytes() == original
    assert output.read_bytes() == original


@pytest.fixture
def supplied_plan(tmp_path, robot):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    np.savez_compressed(
        source,
        joint_positions=np.zeros((2, robot.dof)),
        sample_indices=[5, 90],
        joint_names=np.asarray(robot.joint_names),
    )
    save_pose_plan(output, input_joints=source)
    return output


@pytest.mark.parametrize(
    "indices",
    [[0], [-1, 2], [3, 3], [0, 1.5], np.array([0, 2**63], dtype=np.uint64)],
)
def test_loader_validates_optional_source_id_mapping(supplied_plan, indices):
    rewrite_plan(supplied_plan, arrays={"source_sample_indices": np.asarray(indices)})
    with pytest.raises(ValueError, match="source_sample_indices"):
        load_pose_plan(supplied_plan)


@pytest.mark.parametrize(
    "metadata",
    [
        {"input_joints_sha256": "bad"},
        {"input_joints_format": "csv"},
        {"input_joint_identity_verified": 1},
        {"input_joint_identity_verified": False},
        {"input_joint_names": None},
        {"generation_seed_used": True},
        {"generation_seed_used": "false"},
        {"selection_method": "unknown"},
        {"input_joints_format": "npy"},
        {"input_joint_names": ["wrong"] * 6},
    ],
)
def test_loader_rejects_inconsistent_supplied_provenance(supplied_plan, metadata):
    rewrite_plan(supplied_plan, metadata=metadata)
    with pytest.raises(ValueError):
        load_pose_plan(supplied_plan)


def test_loader_requires_complete_optional_provenance(supplied_plan):
    rewrite_plan(supplied_plan, remove_metadata=("input_joints_sha256",))
    with pytest.raises(ValueError, match="missing input joint provenance"):
        load_pose_plan(supplied_plan)


def test_loader_checks_all_input_selection_order(supplied_plan):
    plan = load_pose_plan(supplied_plan)
    reversed_selection = plan.selected_candidate_indices[::-1]
    rewrite_plan(
        supplied_plan,
        arrays={
            "selected_candidate_indices": reversed_selection,
            "candidate_indices": reversed_selection[plan.visit_order],
        },
    )
    with pytest.raises(ValueError, match="retain every source row in order"):
        load_pose_plan(supplied_plan)


def test_generated_defaults_and_legacy_schema_remain_compatible(tmp_path, robot):
    output = tmp_path / "generated.npz"
    report = save_pose_plan(output)
    plan = load_pose_plan(output)
    proposals = np.random.default_rng(42).uniform(-np.pi, np.pi, (400, robot.dof))
    expected = select_informative_poses(robot, proposals, count=40)
    np.testing.assert_array_equal(
        plan.selected_candidate_indices, expected.candidate_indices
    )
    np.testing.assert_array_equal(
        plan.planned_joint_positions, proposals[plan.candidate_indices]
    )
    assert report["selected_count"] == 40
    assert report["generation_seed_used"] is True
    assert report["selection_method"] == "d-optimal"
    rewrite_plan(output, remove_metadata=("generation_seed_used", "selection_method"))
    legacy = load_pose_plan(output)
    assert legacy.source_sample_indices is None
    np.testing.assert_array_equal(legacy.candidate_indices, plan.candidate_indices)


def test_supplied_planning_and_loading_keep_optional_engines_unimported(tmp_path):
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, np.zeros((1, 6)))
    script = """
import sys
from unittest.mock import patch
blocked = {name: None for name in ('roboticstoolbox', 'dexsim', 'viser', 'cv2')}
with patch.dict(sys.modules, blocked):
    from kinematic_calibration.planning import save_pose_plan, load_pose_plan
    save_pose_plan(sys.argv[2], input_joints=sys.argv[1])
    plan = load_pose_plan(sys.argv[2])
    assert plan.planned_joint_positions.shape == (1, 6)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", script, str(source), str(output)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_supplied_plan_reaches_measured_experiment_and_training_playback(
    tmp_path, robot
):
    source, output = tmp_path / "chosen.npz", tmp_path / "plan.npz"
    rows = np.random.default_rng(17).uniform(-2, 2, (12, robot.dof))
    ids = np.arange(12, dtype=np.int64) * 7 + 100
    np.savez_compressed(
        source,
        joint_positions=rows,
        sample_indices=ids,
        joint_names=np.asarray(robot.joint_names),
    )
    save_pose_plan(output, input_joints=source)
    plan = load_pose_plan(output)
    experiment = tmp_path / "experiment"
    report = run_experiment(
        experiment,
        plan=output,
        validation_samples=8,
        seed=19,
        noise=MeasurementNoise(0, 0),
    )
    assert report["accepted"]
    playback = ExperimentPlayback(experiment, split="training")
    np.testing.assert_array_equal(
        playback.dataset.joint_positions, rows[plan.candidate_indices]
    )
    saved_plan = load_pose_plan(experiment / "plan.npz")
    np.testing.assert_array_equal(saved_plan.source_sample_indices, ids)
    state = playback.sample(4)
    np.testing.assert_array_equal(
        state.joint_positions, plan.planned_joint_positions[4]
    )
    assert state.error_reference == "training measurement"
    assert (
        report["training_plan"]["candidate_indices"] == plan.candidate_indices.tolist()
    )
