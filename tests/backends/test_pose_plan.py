"""Saved multi-pose visit sequences and their command-line entry."""

import json

import numpy as np
import pytest

from kinematic_calibration.cli import main
from kinematic_calibration.planning import save_pose_plan


@pytest.mark.parametrize(
    "start,return_to_start",
    [(None, False), (np.zeros(6), False), (np.zeros(6), True)],
)
def test_plan_preserves_candidate_identity_and_accounts_for_every_leg(
    tmp_path, start, return_to_start
):
    output = tmp_path / "plan.npz"
    report = save_pose_plan(
        output,
        count=10,
        candidates=50,
        seed=7,
        start_configuration=start,
        return_to_start=return_to_start,
    )
    with np.load(output, allow_pickle=False) as archive:
        positions = archive["planned_joint_positions"]
        indices = archive["candidate_indices"]
        selected = archive["selected_candidate_indices"]
        visit_order = archive["visit_order"]
        segments = archive["segment_distances_rad"]
        assert json.loads(str(archive["metadata_json"])) == report
        assert archive["joint_names"].tolist() == report["joint_names"]
        assert archive["joint_limits"].shape == (6, 2)
        np.testing.assert_array_equal(np.sort(visit_order), np.arange(10))
        np.testing.assert_array_equal(indices, selected[visit_order])
        assert len(np.unique(indices)) == 10
        # Reconstruct seeded candidates independently of the routing algorithm.
        candidates = np.random.default_rng(7).uniform(-np.pi, np.pi, (50, 6))
        np.testing.assert_array_equal(positions, candidates[indices])
        path = positions
        if start is not None:
            path = np.vstack([start, path])
            if return_to_start:
                path = np.vstack([path, start])
        else:
            assert visit_order[0] == 0
        expected_segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
        np.testing.assert_allclose(segments, expected_segments)
        assert report["total_distance_rad"] == pytest.approx(expected_segments.sum())
    assert report["schema_version"] == 2
    assert report["full_rank"] and report["regressor_rank"] == 30
    assert report["total_distance_rad"] <= report["selection_order_distance_rad"]
    assert not report["measurements_acquired"]
    assert not report["trajectory_generated"]


def test_plan_metadata_normalizes_supported_numpy_scalars(tmp_path):
    report = save_pose_plan(
        tmp_path / "plan.npz",
        count=np.int64(5),
        candidates=8,
        characteristic_length_m=np.float32(0.8),
    )
    assert report["selected_count"] == 5
    assert isinstance(report["selected_count"], int)
    assert isinstance(report["characteristic_length_m"], float)


def test_plan_cli_writes_a_closed_route_and_refuses_implicit_replacement(
    tmp_path, capsys
):
    output = tmp_path / "plan.npz"
    arguments = [
        "plan",
        "--count",
        "10",
        "--candidates",
        "40",
        "--start-joints",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "--return-to-start",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["return_to_start"]
    assert report["start_configuration_rad"] == [0] * 6
    original = output.read_bytes()
    assert main(arguments) == 2
    assert "already exists" in capsys.readouterr().err
    assert output.read_bytes() == original
    assert main([*arguments, "--force"]) == 0


@pytest.mark.parametrize(
    "options,message",
    [
        ({"start_configuration": [0, 0]}, "start_configuration"),
        ({"start_configuration": [True] * 6}, "start_configuration"),
        ({"start_configuration": [np.inf] * 6}, "start_configuration"),
        ({"start_configuration": [10] * 6}, "joint limits"),
        ({"return_to_start": True}, "requires start_configuration"),
        ({"return_to_start": 1}, "boolean"),
        ({"count": 9}, "candidate count"),
    ],
)
def test_invalid_plan_does_not_replace_existing_output(tmp_path, options, message):
    output = tmp_path / "plan.npz"
    output.write_bytes(b"previous plan")
    with pytest.raises(ValueError, match=message):
        save_pose_plan(output, candidates=8, force=True, **options)
    assert output.read_bytes() == b"previous plan"


def test_return_to_start_cli_requires_an_explicit_start(tmp_path, capsys):
    output = tmp_path / "plan.npz"
    assert main(["plan", "--return-to-start", "--output", str(output)]) == 2
    assert "requires start_configuration" in capsys.readouterr().err
    assert not output.exists()


def test_plan_cli_respects_a_five_joint_subchain(tmp_path, capsys):
    output = tmp_path / "subchain.npz"
    assert (
        main(
            [
                "plan",
                "--base-link",
                "Link1",
                "--count",
                "10",
                "--candidates",
                "40",
                "--start-joints",
                "0",
                "0",
                "0",
                "0",
                "0",
                "--return-to-start",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["base_link"] == "Link1"
    assert report["joint_names"] == [f"Joint{index}" for index in range(2, 7)]
    assert report["parameter_count"] == report["regressor_rank"] == 26
    with np.load(output, allow_pickle=False) as archive:
        assert archive["planned_joint_positions"].shape == (10, 5)
        assert archive["segment_distances_rad"].shape == (11,)


def test_rank_deficient_plan_is_saved_with_a_nonzero_cli_status(tmp_path, capsys):
    output = tmp_path / "one-pose.npz"
    assert (
        main(
            [
                "plan",
                "--count",
                "1",
                "--candidates",
                "5",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert not report["full_rank"]
    assert report["regressor_rank"] == 6
    assert report["condition_number"] is None
    with np.load(output, allow_pickle=False) as archive:
        assert archive["planned_joint_positions"].shape == (1, 6)
        assert archive["segment_distances_rad"].shape == (0,)
