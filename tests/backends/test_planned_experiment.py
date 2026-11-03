"""Planned synthetic training preserves visit rows and archived provenance."""

import hashlib
import json
import shutil

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import (
    example_urdf,
    load_experiment,
    run_experiment,
)
from kinematic_calibration.planning import load_pose_plan, save_pose_plan
from kinematic_calibration.simulation import MeasurementNoise, sample_measurements


@pytest.fixture(scope="module")
def planned_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("planned-experiment")
    plan = root / "source-plan.npz"
    save_pose_plan(
        plan,
        count=12,
        candidates=40,
        seed=7,
        start_configuration=np.zeros(6),
        return_to_start=True,
    )
    output = root / "experiment"
    run_experiment(
        output,
        plan=plan,
        samples=1,
        validation_samples=8,
        seed=17,
        noise=MeasurementNoise(0, 0),
    )
    return plan, output


def copy_experiment(snapshot, tmp_path):
    source, experiment = snapshot
    output = tmp_path / "experiment"
    shutil.copytree(experiment, output)
    return source, output


def refresh_manifest(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = hashlib.sha256(
            (path / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def rewrite_plan(path, *, arrays=None, metadata=None):
    with np.load(path, allow_pickle=False) as archive:
        contents = {key: archive[key] for key in archive.files}
    if arrays:
        contents.update(arrays)
    if metadata:
        report = json.loads(contents["metadata_json"].item())
        report.update(metadata)
        contents["metadata_json"] = np.asarray(json.dumps(report))
    np.savez_compressed(path, **contents)


def test_plan_run_load_measures_visit_rows_and_preserves_candidate_mapping(
    planned_experiment,
):
    source, output = planned_experiment
    plan = load_pose_plan(source)
    report, validation, _ = load_experiment(output)
    train = KinematicDataset.load(output / "train.npz")
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=18)

    assert report["schema_version"] == 2
    assert report["training_samples"] == 12
    assert report["sample_id_namespace"] == "experiment"
    assert (output / "plan.npz").read_bytes() == source.read_bytes()
    assert report["training_plan"] == {
        "artifact": "plan.npz",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_indices": plan.candidate_indices.tolist(),
    }
    assert report["training_plan"]["candidate_indices"] != list(range(12))
    np.testing.assert_array_equal(train.joint_positions, plan.planned_joint_positions)
    np.testing.assert_array_equal(train.sample_indices, np.arange(12))
    np.testing.assert_array_equal(validation.sample_indices, np.arange(12, 20))
    np.testing.assert_allclose(
        train.observed_poses,
        np.asarray([truth.forward(q) for q in plan.planned_joint_positions]),
        atol=1e-15,
        rtol=0,
    )
    assert report["calibrated_vs_measurements"]["position_rmse_m"] < 1e-6
    assert report["nominal_vs_measurements"]["position_rmse_m"] > 1e-3


def test_planned_bundle_remains_loadable_after_move(planned_experiment, tmp_path):
    _, output = copy_experiment(planned_experiment, tmp_path)
    moved = tmp_path / "moved"
    shutil.move(output, moved)
    report, _, _ = load_experiment(moved)
    assert report["training_plan"]["artifact"] == "plan.npz"


def test_large_unsigned_candidate_identity_is_compared_as_exact_integers(
    planned_experiment, tmp_path
):
    _, output = copy_experiment(planned_experiment, tmp_path)
    plan = load_pose_plan(output / "plan.npz")
    selected = np.arange(len(plan.visit_order), dtype=np.uint64)
    selected[7] = 2**63 + 1
    candidates = selected[plan.visit_order]
    rewrite_plan(
        output / "plan.npz",
        arrays={
            "selected_candidate_indices": selected,
            "candidate_indices": candidates,
        },
        metadata={"candidate_count": 2**63 + 100},
    )
    report_path = output / "metrics.json"
    report = json.loads(report_path.read_text())
    report["training_plan"]["sha256"] = hashlib.sha256(
        (output / "plan.npz").read_bytes()
    ).hexdigest()
    report["training_plan"]["candidate_indices"] = candidates.tolist()
    report_path.write_text(json.dumps(report))
    refresh_manifest(output)
    load_experiment(output)

    changed_row = int(np.flatnonzero(plan.visit_order == 7)[0])
    report["training_plan"]["candidate_indices"][changed_row] += 1
    report_path.write_text(json.dumps(report))
    refresh_manifest(output)
    with pytest.raises(ValueError, match="candidate_indices.*archived pose plan"):
        load_experiment(output)


def test_plan_sharing_validation_seed_still_produces_disjoint_held_out_rows(tmp_path):
    plan_path = tmp_path / "plan.npz"
    save_pose_plan(plan_path, count=16, candidates=20, seed=44)
    plan = load_pose_plan(plan_path)
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    original_validation = sample_measurements(
        nominal.perturbed(seed=43), count=8, seed=44
    )
    training_rows = set(map(tuple, plan.planned_joint_positions))
    assert training_rows & set(map(tuple, original_validation.joint_positions))

    output = tmp_path / "experiment"
    run_experiment(output, plan=plan_path, seed=42, validation_samples=8)
    report, validation, _ = load_experiment(output)

    assert len(validation.joint_positions) == 8
    assert report["validation_seed"] == 44
    assert not training_rows & set(map(tuple, validation.joint_positions))
    np.testing.assert_array_equal(validation.sample_indices, np.arange(16, 24))


@pytest.mark.parametrize("mismatch", ["urdf", "joint_order", "chain", "limits"])
def test_plan_must_match_the_experiment_model(planned_experiment, tmp_path, mismatch):
    source, _ = planned_experiment
    plan = tmp_path / "source.npz"
    shutil.copyfile(source, plan)
    urdf = example_urdf()
    message = ""
    if mismatch == "urdf":
        urdf = tmp_path / "different.urdf"
        urdf.write_bytes(example_urdf().read_bytes() + b"\n")
        message = "source URDF SHA-256"
    elif mismatch == "joint_order":
        names = load_pose_plan(plan).joint_names[::-1]
        rewrite_plan(
            plan,
            arrays={"joint_names": np.asarray(names)},
            metadata={"joint_names": list(names)},
        )
        message = "joint_names"
    elif mismatch == "chain":
        rewrite_plan(plan, metadata={"end_link": "other_link"})
        message = "end_link"
    else:
        limits = load_pose_plan(plan).joint_limits.copy()
        limits[:, 0] -= 0.1
        rewrite_plan(plan, arrays={"joint_limits": limits})
        message = "joint_limits"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        run_experiment(output, urdf=urdf, plan=plan)
    assert not output.exists()


def test_small_preview_plan_still_requires_the_solver_sample_minimum(tmp_path):
    plan = tmp_path / "one.npz"
    save_pose_plan(plan, count=1, candidates=2)
    with pytest.raises(ValueError, match="samples must be >= 5"):
        run_experiment(tmp_path / "output", plan=plan)


def test_load_rejects_reordered_training_rows_after_rehash(
    planned_experiment, tmp_path
):
    _, output = copy_experiment(planned_experiment, tmp_path)
    train = KinematicDataset.load(output / "train.npz")
    order = np.roll(np.arange(len(train.joint_positions)), 1)
    KinematicDataset(
        train.joint_positions[order],
        train.observed_poses[order],
        train.observation_frame,
        train.sample_indices,
        train.joint_names,
    ).save(output / "train.npz")
    refresh_manifest(output)
    with pytest.raises(ValueError, match="training joint rows.*archived pose plan"):
        load_experiment(output)


def test_load_rejects_changed_plan_candidate_mapping_after_rehash(
    planned_experiment, tmp_path
):
    _, output = copy_experiment(planned_experiment, tmp_path)
    plan_path = output / "plan.npz"
    plan = load_pose_plan(plan_path)
    selected = np.roll(plan.selected_candidate_indices, 1)
    rewrite_plan(
        plan_path,
        arrays={
            "selected_candidate_indices": selected,
            "candidate_indices": selected[plan.visit_order],
        },
    )
    report_path = output / "metrics.json"
    report = json.loads(report_path.read_text())
    report["training_plan"]["sha256"] = hashlib.sha256(
        plan_path.read_bytes()
    ).hexdigest()
    report_path.write_text(json.dumps(report))
    refresh_manifest(output)
    with pytest.raises(ValueError, match="candidate_indices.*archived pose plan"):
        load_experiment(output)


@pytest.mark.parametrize(
    "change",
    [
        "missing_file",
        "missing_hash",
        "missing_metadata",
        "extra_metadata",
        "wrong_sha",
        "wrong_artifact",
        "boolean_candidate",
    ],
)
def test_optional_plan_artifact_manifest_and_report_must_agree(
    planned_experiment, tmp_path, change
):
    _, output = copy_experiment(planned_experiment, tmp_path)
    manifest_path = output / "manifest.json"
    report_path = output / "metrics.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if change == "missing_file":
        (output / "plan.npz").unlink()
    elif change == "missing_hash":
        del manifest["files_sha256"]["plan.npz"]
    elif change == "missing_metadata":
        del report["training_plan"]
    elif change == "extra_metadata":
        (output / "plan.npz").unlink()
        del manifest["files_sha256"]["plan.npz"]
    elif change == "wrong_sha":
        report["training_plan"]["sha256"] = "0" * 64
    elif change == "wrong_artifact":
        report["training_plan"]["artifact"] = "other.npz"
    else:
        report["training_plan"]["candidate_indices"][0] = True
    report_path.write_text(json.dumps(report))
    manifest["files_sha256"]["metrics.json"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|training_plan"):
        load_experiment(output)


def test_optional_plan_hash_is_checked(planned_experiment, tmp_path):
    _, output = copy_experiment(planned_experiment, tmp_path)
    with (output / "plan.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch: plan.npz"):
        load_experiment(output)


def test_force_switches_between_planned_and_random_training(
    planned_experiment, tmp_path
):
    source, output = copy_experiment(planned_experiment, tmp_path)
    (output / "notes.txt").write_text("preserve unrelated files")
    run_experiment(output, samples=12, validation_samples=8, force=True)
    report, _, _ = load_experiment(output)
    assert "training_plan" not in report
    assert not (output / "plan.npz").exists()
    run_experiment(output, plan=source, validation_samples=8, force=True)
    report, _, _ = load_experiment(output)
    assert report["training_plan"]["artifact"] == "plan.npz"
    assert (output / "notes.txt").read_text() == "preserve unrelated files"


@pytest.mark.parametrize(
    "replacement_plan,failed_file",
    [(False, "metrics.json"), (True, "plan.npz"), (True, "manifest.json")],
)
def test_force_failure_restores_optional_plan_and_all_previous_artifacts(
    planned_experiment, tmp_path, monkeypatch, replacement_plan, failed_file
):
    import kinematic_calibration.experiment as experiment

    source, output = copy_experiment(planned_experiment, tmp_path)
    (output / "notes.txt").write_text("keep")
    before = {entry.name: entry.read_bytes() for entry in output.iterdir()}
    original_replace = experiment.os.replace
    armed = True

    def fail_one_install(source_path, destination):
        nonlocal armed
        if armed and destination == output / failed_file:
            armed = False
            raise OSError("injected planned publish failure")
        return original_replace(source_path, destination)

    monkeypatch.setattr(experiment.os, "replace", fail_one_install)
    with pytest.raises(OSError, match="injected planned publish failure"):
        run_experiment(
            output,
            plan=source if replacement_plan else None,
            samples=12,
            validation_samples=8,
            force=True,
        )
    assert {entry.name: entry.read_bytes() for entry in output.iterdir()} == before
    load_experiment(output)


def test_source_plan_inside_output_is_snapshotted_and_manifest_published_last(
    planned_experiment, tmp_path, monkeypatch
):
    import kinematic_calibration.experiment as experiment

    _, output = copy_experiment(planned_experiment, tmp_path)
    source = output / "plan.npz"
    before = source.read_bytes()
    original_replace = experiment.os.replace
    installed = []

    def record_install(source_path, destination):
        if destination.parent == output:
            installed.append(destination.name)
        return original_replace(source_path, destination)

    monkeypatch.setattr(experiment.os, "replace", record_install)
    run_experiment(output, plan=source, validation_samples=8, force=True)
    assert source.read_bytes() == before
    assert installed[-1] == "manifest.json"
    load_experiment(output)


def test_source_plan_cannot_be_replaced_through_another_reserved_path(
    planned_experiment, tmp_path
):
    source, _ = planned_experiment
    output = tmp_path / "output"
    output.mkdir()
    conflicting = output / "model.npz"
    shutil.copyfile(source, conflicting)
    before = conflicting.read_bytes()
    with pytest.raises(ValueError, match="source plan conflicts"):
        run_experiment(output, plan=conflicting, force=True)
    assert conflicting.read_bytes() == before


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_optional_plan_reserved_path_must_be_regular(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    plan = output / "plan.npz"
    if kind == "directory":
        plan.mkdir()
    else:
        plan.symlink_to(tmp_path / "missing-plan.npz")
    with pytest.raises(ValueError, match="reserved artifact path"):
        run_experiment(output, force=True)
