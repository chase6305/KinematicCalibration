"""The CLI plans visits for already chosen poses without invented observations."""

import json

import numpy as np
import pytest

from kinematic_calibration.cli import main
from kinematic_calibration.planning import load_pose_plan


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
