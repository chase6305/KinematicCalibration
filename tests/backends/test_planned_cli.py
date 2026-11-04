"""The saved plan reaches the experiment and its explicit training replay."""

import json

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.cli import main
from kinematic_calibration.planning import load_pose_plan, save_pose_plan


@pytest.fixture
def planned_experiment(tmp_path, capsys):
    plan = tmp_path / "route.npz"
    save_pose_plan(plan, count=12, candidates=60, seed=41)
    output = tmp_path / "experiment"
    assert (
        main(
            [
                "demo",
                "--plan",
                str(plan),
                "--validation-samples",
                "8",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    return plan, output, report


def test_demo_cli_consumes_plan_order_and_preserves_candidate_identity(
    planned_experiment, capsys
):
    plan_path, output, report = planned_experiment
    plan = load_pose_plan(plan_path)
    train = KinematicDataset.load(output / "train.npz")
    assert report["training_samples"] == 12
    assert report["validation_samples"] == 8
    np.testing.assert_array_equal(train.joint_positions, plan.planned_joint_positions)
    assert (
        report["training_plan"]["candidate_indices"] == plan.candidate_indices.tolist()
    )
    assert (output / "plan.npz").read_bytes() == plan_path.read_bytes()
    assert main(["inspect", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_demo_cli_rejects_ambiguous_training_count_with_plan(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "demo",
                "--plan",
                str(tmp_path / "route.npz"),
                "--samples",
                "12",
                "--output",
                str(tmp_path / "experiment"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "experiment").exists()


def test_training_view_cli_uses_real_server_and_stops_after_duration(
    planned_experiment, monkeypatch
):
    pytest.importorskip("viser")
    from kinematic_calibration.backends import viser as backend

    _, output, _ = planned_experiment
    create_viewer = backend.create_viewer
    splits, stopped = [], []

    def tracked_viewer(*args, **kwargs):
        splits.append(kwargs["split"])
        server = create_viewer(*args, **kwargs)
        stop = server.stop

        def tracked_stop():
            stopped.append(True)
            stop()

        monkeypatch.setattr(server, "stop", tracked_stop)
        return server

    monkeypatch.setattr(backend, "create_viewer", tracked_viewer)
    assert (
        main(
            [
                "view",
                str(output),
                "--split",
                "training",
                "--port",
                "0",
                "--duration",
                "0.01",
            ]
        )
        == 0
    )
    assert splits == ["training"]
    assert stopped == [True]
