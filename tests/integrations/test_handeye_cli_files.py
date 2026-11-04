"""Offline hand-eye commands preserve inputs and complete output pairs."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")

from kinematic_calibration.handeye_cli import main

REPORT_COMMANDS = ("solve", "compare", "influence", "bootstrap")


def _pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    pose[:3, 3] = translation
    return pose


@pytest.fixture
def observations(tmp_path):
    base_T_camera = _pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    ee_T_board = _pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    base_T_ee = np.stack(
        [
            _pose(
                [0.08 * index, -0.05 * index**2, 0.04 * (index + 1)],
                [0.1 * np.sin(index), 0.08 * np.cos(index), 0.35 + 0.03 * index],
            )
            for index in range(1, 10)
        ]
    )
    camera_T_board = np.stack(
        [np.linalg.inv(base_T_camera) @ pose @ ee_T_board for pose in base_T_ee]
    )
    samples = tmp_path / "samples.npz"
    np.savez_compressed(
        samples,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=base_T_ee,
        camera_T_board=camera_T_board,
        candidate_indices=np.arange(9),
        translation_candidate_count=np.asarray(6),
        rotation_candidate_count=np.asarray(3),
    )
    return samples, base_T_camera, ee_T_board


@pytest.fixture
def accepted_result(observations):
    samples, _, _ = observations
    result = samples.with_suffix(".json")
    assert main(_report_arguments("solve", samples, result)) == 0
    return result


def _report_arguments(command, samples, output, *, force=False):
    arguments = [command, "--samples", str(samples), "--output", str(output)]
    if command == "bootstrap":
        arguments.extend(["--replicates", "2"])
    if force:
        arguments.append("--force")
    return arguments


def _export_arguments(samples, result, outputs, *, force=False):
    arguments = [
        "export",
        "--samples",
        str(samples),
        "--result",
        str(result),
        "--base-t-camera",
        str(outputs[0]),
        "--ee-t-target",
        str(outputs[1]),
    ]
    if force:
        arguments.append("--force")
    return arguments


@pytest.mark.parametrize("command", REPORT_COMMANDS)
@pytest.mark.parametrize("alias", ["input_symlink", "output_hardlink"])
def test_report_force_preserves_samples_aliased_by_output(
    observations, command, alias
):
    samples, _, _ = observations
    original = samples.read_bytes()
    output = samples.parent / "source.json"
    if alias == "input_symlink":
        samples.rename(output)
        samples.symlink_to(output.name)
    else:
        output.hardlink_to(samples)

    assert main(_report_arguments(command, samples, output, force=True)) == 2
    assert samples.read_bytes() == original
    assert output.read_bytes() == original


@pytest.mark.parametrize("command", REPORT_COMMANDS)
def test_report_requires_force_to_replace_existing_output(observations, command):
    samples, _, _ = observations
    output = samples.parent / "result.json"
    output.write_bytes(b"previous result")

    assert main(_report_arguments(command, samples, output)) == 2
    assert output.read_bytes() == b"previous result"


@pytest.mark.parametrize("command", REPORT_COMMANDS)
@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling_symlink"])
def test_report_force_rejects_nonregular_output(observations, command, kind):
    samples, _, _ = observations
    output = samples.parent / "result.json"
    target = samples.parent / "other.json"
    if kind == "directory":
        output.mkdir()
    else:
        if kind == "symlink":
            target.write_bytes(b"unrelated result")
        output.symlink_to(target.name)

    assert main(_report_arguments(command, samples, output, force=True)) == 2
    if kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()
        if kind == "symlink":
            assert target.read_bytes() == b"unrelated result"
        else:
            assert not target.exists()


@pytest.mark.parametrize("source", ["samples", "result"])
def test_export_force_preserves_input_hardlink(
    observations, accepted_result, source
):
    samples, _, _ = observations
    inputs = {"samples": samples, "result": accepted_result}
    original = {name: path.read_bytes() for name, path in inputs.items()}
    outputs = (samples.parent / "camera.npy", samples.parent / "target.npy")
    outputs[0].hardlink_to(inputs[source])
    outputs[1].write_bytes(b"previous target")

    assert main(_export_arguments(samples, accepted_result, outputs, force=True)) == 2
    assert {name: path.read_bytes() for name, path in inputs.items()} == original
    assert outputs[0].read_bytes() == original[source]
    assert outputs[1].read_bytes() == b"previous target"


def test_export_preflights_both_destinations(observations, accepted_result):
    samples, _, _ = observations
    outputs = (samples.parent / "camera.npy", samples.parent / "target.npy")
    outputs[0].write_bytes(b"previous camera")
    outputs[1].mkdir()

    assert main(_export_arguments(samples, accepted_result, outputs, force=True)) == 2
    assert outputs[0].read_bytes() == b"previous camera"
    assert outputs[1].is_dir()


def _previous_exports(samples):
    outputs = (
        samples.parent / "camera" / "transform.npy",
        samples.parent / "target" / "transform.npy",
    )
    previous = (b"previous camera", b"previous target")
    for output, content in zip(outputs, previous, strict=True):
        output.parent.mkdir()
        output.write_bytes(content)
    return outputs, previous


def test_export_restores_both_outputs_when_second_install_fails(
    observations, accepted_result, monkeypatch
):
    samples, _, _ = observations
    outputs, previous = _previous_exports(samples)
    replace = os.replace
    injected = False

    def fail_second_install(source, destination):
        nonlocal injected
        if Path(destination) == outputs[1] and not injected:
            injected = True
            raise OSError("second export install failed")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_install)

    assert main(_export_arguments(samples, accepted_result, outputs, force=True)) == 2
    assert injected
    assert tuple(path.read_bytes() for path in outputs) == previous
    assert all(set(path.parent.iterdir()) == {path} for path in outputs)


@pytest.mark.parametrize("operation", ["backup", "install"])
@pytest.mark.parametrize("index", [0, 1])
def test_export_restores_both_outputs_after_post_syscall_interrupt(
    observations, accepted_result, monkeypatch, operation, index
):
    samples, _, _ = observations
    outputs, previous = _previous_exports(samples)
    replace = os.replace
    injected = False

    def interrupt_after_replace(source, destination):
        nonlocal injected
        source, destination = Path(source), Path(destination)
        matches = (
            source == outputs[index]
            if operation == "backup"
            else destination == outputs[index]
        )
        replace(source, destination)
        if matches and not injected:
            injected = True
            raise KeyboardInterrupt("after real export replacement")

    monkeypatch.setattr(os, "replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt, match="after real export replacement"):
        main(_export_arguments(samples, accepted_result, outputs, force=True))

    assert injected
    assert tuple(path.read_bytes() for path in outputs) == previous
    assert all(set(path.parent.iterdir()) == {path} for path in outputs)


@pytest.mark.parametrize("existing", [False, True])
def test_export_supports_destinations_in_distinct_directories(
    observations, accepted_result, existing
):
    samples, base_T_camera, ee_T_board = observations
    if existing:
        outputs, _ = _previous_exports(samples)
    else:
        outputs = (
            samples.parent / "camera" / "transform.npy",
            samples.parent / "target" / "transform.npy",
        )

    assert (
        main(_export_arguments(samples, accepted_result, outputs, force=existing)) == 0
    )
    report = json.loads(accepted_result.read_text())
    for output, expected, field in zip(
        outputs,
        (base_T_camera, ee_T_board),
        ("Base_T_Camera", "EE_T_Board"),
        strict=True,
    ):
        actual = np.load(output, allow_pickle=False)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_array_equal(actual, np.asarray(report[field]))
    assert all(set(path.parent.iterdir()) == {path} for path in outputs)


@pytest.mark.parametrize("kind", ["empty", "truncated", "npy"])
def test_corrupt_samples_return_error_and_preserve_output(
    observations, caplog, capsys, kind
):
    samples, _, _ = observations
    if kind == "npy":
        with samples.open("wb") as stream:
            np.save(stream, np.eye(4), allow_pickle=False)
    else:
        samples.write_bytes(b"" if kind == "empty" else samples.read_bytes()[:80])
    output = samples.parent / "result.json"
    output.write_bytes(b"previous result")

    assert main(_report_arguments("solve", samples, output, force=True)) == 2
    assert output.read_bytes() == b"previous result"
    assert caplog.records
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert str(samples) in caplog.text
    assert "NPZ" in caplog.text
    assert "Traceback" not in capsys.readouterr().err
