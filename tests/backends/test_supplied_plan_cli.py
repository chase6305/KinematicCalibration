"""The CLI plans visits for already chosen poses without invented observations."""

import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from kinematic_calibration.cli import main
from kinematic_calibration.planning import load_pose_plan, save_pose_plan


def _run_plan_cli(source, output):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "kinematic_calibration.cli",
            "plan",
            "--input-joints",
            str(source),
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_small_supplied_pose_set_returns_success_and_retains_source_identity(
    tmp_path, capsys
):
    source = tmp_path / "chosen.npz"
    joints = np.array([[0.0] * 6, [0.4] * 6, [-0.2] * 6])
    sample_ids = np.array([80, 12, 901])
    np.savez_compressed(
        source,
        joint_positions=joints,
        sample_indices=sample_ids,
        joint_names=np.asarray([f"Joint{i}" for i in range(1, 7)]),
    )
    output = tmp_path / "plan.npz"

    assert main(["plan", "--input-joints", str(source), "--output", str(output)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["selected_count"] == 3
    assert report["selection_method"] == "all-input"
    assert not report["full_rank"]
    assert report["input_joint_identity_verified"]
    plan = load_pose_plan(output)
    np.testing.assert_array_equal(plan.source_sample_indices, sample_ids)
    np.testing.assert_array_equal(
        plan.planned_joint_positions, joints[plan.candidate_indices]
    )
    assert not report["measurements_acquired"]


def test_explicit_subset_preserves_existing_rank_exit_policy(tmp_path, capsys):
    source, output = tmp_path / "chosen.npy", tmp_path / "plan.npz"
    np.save(source, np.random.default_rng(3).uniform(-1, 1, (12, 6)))
    assert (
        main(
            [
                "plan",
                "--input-joints",
                str(source),
                "--count",
                "1",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["selection_method"] == "d-optimal"
    assert report["selected_count"] == 1
    assert not report["input_joint_identity_verified"]


def test_candidate_generation_and_supplied_input_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "plan",
                "--input-joints",
                str(tmp_path / "chosen.npy"),
                "--candidates",
                "20",
                "--output",
                str(tmp_path / "plan.npz"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "plan.npz").exists()


@pytest.mark.parametrize(
    "filename,payload",
    [("empty.npy", b""), ("truncated.npz", b"PK\x03\x04"), ("text.npy", b"not numpy")],
)
def test_unreadable_input_reports_cli_error_and_preserves_previous_plan(
    tmp_path, capsys, filename, payload
):
    source, output = tmp_path / filename, tmp_path / "plan.npz"
    source.write_bytes(payload)
    output.write_bytes(b"previous plan")
    assert (
        main(
            [
                "plan",
                "--input-joints",
                str(source),
                "--output",
                str(output),
                "--force",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "kincal:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert source.read_bytes() == payload
    assert output.read_bytes() == b"previous plan"


def test_deflate_input_joint_error_closes_archive_and_reports_cli_input_path(
    tmp_path, monkeypatch
):
    source = tmp_path / "chosen.npz"
    np.savez_compressed(
        source, joint_positions=np.array([[0.0] * 6, [0.2] * 6])
    )
    content = bytearray(source.read_bytes())
    with ZipFile(source) as archive:
        member = archive.getinfo("joint_positions.npy")
        assert member.compress_type == ZIP_DEFLATED
        name_size, extra_size = struct.unpack_from(
            "<HH", content, member.header_offset + 26
        )
        offset = member.header_offset + 30 + name_size + extra_size
    content[offset] = (content[offset] & ~0x06) | 0x06
    assert len(content) < 4096
    original = bytes(content)
    source.write_bytes(original)
    real_load = np.load
    archives = []

    def tracked_load(*args, **kwargs):
        archive = real_load(*args, **kwargs)
        archives.append(archive.zip)
        return archive

    monkeypatch.setattr("kinematic_calibration.planning.np.load", tracked_load)
    with pytest.raises(ValueError) as error:
        save_pose_plan(tmp_path / "api-plan.npz", input_joints=source)
    assert str(source) in str(error.value)
    assert isinstance(error.value.__cause__, zlib.error)
    assert len(archives) == 1 and archives[0].fp is None

    completed = _run_plan_cli(source, tmp_path / "cli-plan.npz")

    assert completed.returncode == 2, completed.stderr
    assert "Traceback" not in completed.stderr
    assert str(source) in completed.stderr
    assert completed.stdout == ""
    assert source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("extension", ["npy", "npz"])
def test_real_cli_preserves_valid_joint_input_format_and_exact_bytes_hash(
    tmp_path, extension
):
    source, output = tmp_path / f"chosen.{extension}", tmp_path / "plan.npz"
    joints = np.array([[0.0] * 6, [0.4] * 6, [-0.2] * 6])
    sample_ids = np.array([80, 12, 901])
    if extension == "npy":
        np.save(source, joints)
    else:
        np.savez_compressed(
            source,
            joint_positions=joints,
            sample_indices=sample_ids,
            joint_names=np.asarray([f"Joint{i}" for i in range(1, 7)]),
        )
    original = source.read_bytes()

    completed = _run_plan_cli(source, output)

    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr
    report = json.loads(completed.stdout)
    assert report["input_joints_format"] == extension
    assert report["input_joints_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["input_joint_identity_verified"] is (extension == "npz")
    assert report["selection_method"] == "all-input"
    plan = load_pose_plan(output)
    np.testing.assert_array_equal(
        plan.planned_joint_positions, joints[plan.candidate_indices]
    )
    if extension == "npz":
        np.testing.assert_array_equal(plan.source_sample_indices, sample_ids)
    else:
        assert plan.source_sample_indices is None
    assert source.read_bytes() == original
