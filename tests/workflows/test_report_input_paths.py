"""Report destinations must preserve source data, including path aliases."""

import os

import pytest

from kinematic_calib.cli import main
from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.simulation import sample_measurements


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
@pytest.mark.parametrize(
    "command,arguments",
    [
        ("inspect", ["--dataset", "SOURCE"]),
        (
            "validate-result",
            [
                "--parameters",
                "fit.npz",
                "--dataset",
                "data.npz",
                "--result",
                "SOURCE",
                "--force",
            ],
        ),
        (
            "zero-offset",
            [
                "--dataset",
                "data.npz",
                "--urdf",
                "robot.urdf",
                "--ee-name",
                "ee_link",
                "--model-mode",
                "robot-body",
                "--base-t-camera",
                "SOURCE",
                "--ee-t-target",
                "target.npy",
                "--force",
            ],
        ),
        ("export-urdf", ["--result", "SOURCE", "--urdf", "robot.urdf", "--force"]),
        (
            "compare-zero-offsets",
            ["--result", "SOURCE", "--result", "second.json", "--force"],
        ),
        ("benchmark", ["--urdf", "SOURCE", "--ee-name", "ee_link", "--force"]),
        (
            "validate-urdf-truth",
            [
                "--dataset",
                "data.npz",
                "--nominal-urdf",
                "nominal.urdf",
                "--calibrated-urdf",
                "calibrated.urdf",
                "--ground-truth-urdf",
                "truth.urdf",
                "--zero-offset-result",
                "SOURCE",
                "--force",
            ],
        ),
    ],
)
def test_report_rejects_input_alias_before_reading_or_replacing_source(
    tmp_path, caplog, alias, command, arguments
):
    source = tmp_path / "source.json"
    original = b"input must remain intact"
    source.write_bytes(original)
    output = source if alias == "same" else tmp_path / "report.json"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)
    arguments = [str(source) if value == "SOURCE" else value for value in arguments]

    assert main([command, *arguments, "--output", str(output)]) == 2

    assert "output conflicts with input" in caplog.text
    assert source.read_bytes() == original
    assert output.read_bytes() == original
    if alias == "symlink":
        assert output.is_symlink()


def test_inspection_cannot_replace_a_valid_dataset_with_its_json_report(tmp_path):
    source = tmp_path / "data.npz"
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link").perturbed(
        seed=41
    )
    sample_measurements(robot, count=8, seed=9).save(source)
    original = source.read_bytes()

    assert main(["inspect", "--dataset", str(source), "--output", str(source)]) == 2

    assert source.read_bytes() == original
    assert len(KinematicDataset.load(source).joint_positions) == 8
