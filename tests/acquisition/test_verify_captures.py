"""Offline import receipt verification, independent of source-image detection."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kinematic_calib.cli import main
from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.acquisition import import_captures, verify_capture_report

pytest.importorskip("cv2")


@pytest.fixture
def imported(tmp_path):
    source = Path(__file__).parents[2] / "examples" / "captures.example.json"
    manifest = tmp_path / "captures.json"
    manifest.write_bytes(source.read_bytes())
    output = tmp_path / "observations.npz"
    dataset, report = import_captures(manifest, output)
    return manifest, output, dataset, report


def _write_report(output, report):
    path = output.with_suffix(".report.json")
    path.write_text(json.dumps(report))
    return path


def test_imported_receipt_verifies_through_shared_api_and_legacy_cli(
    imported, monkeypatch, capsys
):
    manifest, output, _, _ = imported
    # Verification is intentionally usable after source files have moved away,
    # and does not import either image detection dependency.
    manifest.unlink()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "handeye_calibration", None)
    report_path = output.with_suffix(".report.json")
    summary = verify_capture_report(output, report_path)
    assert summary["integrity_verified"]
    assert summary["scope"] == "dataset_and_import_report"
    assert summary["accepted_samples"] == summary["total_samples"] == 2
    assert (
        main(
            [
                "verify-collection",
                "--dataset",
                str(output),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == summary


def test_partial_receipt_preserves_accepted_ids_without_hiding_rejections(imported):
    manifest, output, _, _ = imported
    source = json.loads(manifest.read_text())
    source["samples"][0]["image_timestamp_s"] += 1
    manifest.write_text(json.dumps(source))
    dataset, report = import_captures(manifest, output, allow_partial=True, force=True)
    assert dataset.sample_indices.tolist() == [source["samples"][1]["sample_id"]]
    summary = verify_capture_report(output, _write_report(output, report))
    assert summary["accepted_samples"] == summary["rejected_samples"] == 1
    assert summary["total_samples"] == 2
    report["allow_partial"] = False
    with pytest.raises(ValueError, match="acceptance"):
        verify_capture_report(output, _write_report(output, report))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r.update(dataset_written=False),
        lambda r: r.update(dataset_sha256="0" * 64),
        lambda r: r.update(dof=True),
        lambda r: r.update(joint_unit="deg"),
        lambda r: r.update(observation_frame="base_T_ee"),
        lambda r: r["joint_names"].reverse(),
        lambda r: r.update(camera_matrix=np.eye(2).tolist()),
        lambda r: r.update(distortion=None),
        lambda r: r.update(board_points_m=[[0, 0, 0]] * 6),
        lambda r: r.update(frames=[]),
        lambda r: r["frames"].reverse(),
        lambda r: r["frames"][0].update(sample_id=r["frames"][1]["sample_id"]),
        lambda r: r["frames"][0].update(dataset_row=True),
        lambda r: r["frames"][0].update(source=[]),
        lambda r: r["frames"][0].update(rejected="false"),
        lambda r: r["frames"][0].update(detected=False),
        lambda r: r["frames"][0].update(reprojection_rms_px=3.0),
        lambda r: r["frames"][0].update(sync_error_s=0),
        lambda r: r["frames"][0]["pose_diagnostics"].update(ambiguous=True),
        lambda r: r["frames"][0]["camera_T_target"][0].__setitem__(3, 0.2),
        lambda r: r.update(accepted_samples=True),
        lambda r: r.update(rejected_samples=1),
        lambda r: r.update(partial="false"),
        lambda r: r["coverage"].update(samples=99),
    ],
)
def test_report_inconsistencies_are_rejected(imported, change):
    _, output, _, report = imported
    change(report)
    with pytest.raises(ValueError):
        verify_capture_report(output, _write_report(output, report))


@pytest.mark.parametrize("field", ["observed_poses", "sample_indices"])
def test_rehashed_dataset_edits_still_fail_receipt_pose_or_identity_checks(
    imported, field
):
    _, output, _, report = imported
    with np.load(output, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if field == "observed_poses":
        arrays[field][0, 0, 3] += 0.01
    else:
        arrays[field][0] += 20
    np.savez_compressed(output, **arrays)
    report["dataset_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["coverage"] = KinematicDataset.load(output).get_summary()
    with pytest.raises(ValueError, match="pose|subset identity"):
        verify_capture_report(output, _write_report(output, report))
