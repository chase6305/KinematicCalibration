"""Interrupts may arrive after a real filesystem replacement has completed."""

import os
from pathlib import Path

import pytest

from kinematic_calibration import acquisition, experiment


@pytest.mark.parametrize("publisher", ["acquisition", "experiment"])
@pytest.mark.parametrize(
    "operation,previous", [("backup", True), ("install", True), ("install", False)]
)
@pytest.mark.parametrize("target", ["first", "last"])
def test_post_syscall_interruption_restores_complete_previous_output(
    tmp_path, monkeypatch, publisher, operation, previous, target
):
    stage, output = tmp_path / "stage", tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    names = (
        ("dataset.npz", "report.json")
        if publisher == "acquisition"
        else (*experiment._DATA_FILES, experiment._PLAN_FILE, "manifest.json")
    )
    for name in names:
        (stage / name).write_bytes(f"new {name}".encode())
        if previous:
            (output / name).write_bytes(f"old {name}".encode())
    (output / "notes.txt").write_bytes(b"unrelated notes")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    chosen = names[0] if target == "first" else names[-1]
    replace = os.replace
    injected = False

    def interrupt_after_replace(source, destination):
        nonlocal injected
        source, destination = Path(source), Path(destination)
        matches = (
            source == output / chosen
            if operation == "backup"
            else source == stage / chosen and destination == output / chosen
        )
        replace(source, destination)
        if matches and not injected:
            injected = True
            raise KeyboardInterrupt("after real replacement")

    monkeypatch.setattr(os, "replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt, match="after real replacement"):
        if publisher == "acquisition":
            acquisition._publish(stage, {name: output / name for name in names}, True)
        else:
            experiment._publish(stage, output, True)

    assert injected
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert set(tmp_path.iterdir()) == {stage, output}
