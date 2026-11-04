"""Source preservation and recoverable publication of real fitted-model pairs."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from kinematic_calib import cli, io_utils
from kinematic_calib.dataset import KinematicDataset
from kinematic_calib.saved_model import load_fitted_model
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf, predict_poe


def _arguments(directory, *, dataset="first.npz", force=True):
    arguments = [
        "solve",
        "--backend",
        "numpy",
        "--dataset",
        str(directory / dataset),
        "--urdf",
        str(directory / "nominal.urdf"),
        "--ee-name",
        "ee_link",
        "--model-mode",
        "robot-body",
        "--output",
        str(directory / "fit.npz"),
        "--quiet",
    ]
    if force:
        arguments.append("--force")
    return cli.build_parser().parse_args(arguments)


def _hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _forbid_fit(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("output preflight must run before loading data or fitting")

    monkeypatch.setattr(cli.KinematicDataset, "load", unexpected)
    monkeypatch.setattr(cli, "fit_dataset", unexpected)


@pytest.mark.parametrize(
    "input_name", ["dataset", "urdf", "base_t_camera", "ee_t_target"]
)
@pytest.mark.parametrize("artifact", ["parameters", "report"])
@pytest.mark.parametrize("alias_kind", ["same-path", "hardlink"])
def test_solve_force_rejects_source_aliases_before_loading(
    tmp_path, monkeypatch, input_name, artifact, alias_kind
):
    args = _arguments(tmp_path)
    args.base_t_camera = tmp_path / "base_T_camera.npy"
    args.ee_t_target = tmp_path / "EE_T_target.npy"
    for name in ("dataset", "urdf", "base_t_camera", "ee_t_target"):
        getattr(args, name).write_bytes(f"source {name}".encode())
    destination = args.output
    if artifact == "report":
        destination = destination.with_suffix(".json")
    source = getattr(args, input_name)
    if alias_kind == "same-path":
        setattr(args, input_name, destination)
        destination.write_bytes(source.read_bytes())
    else:
        destination.hardlink_to(source)
    before = _hashes(tmp_path)
    _forbid_fit(monkeypatch)

    with pytest.raises(ValueError, match="solve output conflicts with input"):
        cli._solve(args)

    assert _hashes(tmp_path) == before


def test_solve_rejects_resolved_source_symlink_alias(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    report = args.output.with_suffix(".json")
    report.write_bytes(b"source URDF")
    args.urdf.symlink_to(report)
    before = _hashes(tmp_path)
    _forbid_fit(monkeypatch)

    with pytest.raises(ValueError, match="solve output conflicts with input"):
        cli._solve(args)

    assert args.urdf.is_symlink()
    assert _hashes(tmp_path) == before


@pytest.mark.parametrize("artifact", ["parameters", "report"])
@pytest.mark.parametrize("kind", ["directory", "symlink", "broken-symlink"])
def test_solve_rejects_nonregular_outputs_before_loading(
    tmp_path, monkeypatch, artifact, kind
):
    args = _arguments(tmp_path)
    destination = args.output
    if artifact == "report":
        destination = destination.with_suffix(".json")
    target = tmp_path / "preserved-source"
    if kind == "directory":
        destination.mkdir()
    else:
        if kind == "symlink":
            target.write_bytes(b"preserved source")
        destination.symlink_to(target)
    before = _hashes(tmp_path)
    _forbid_fit(monkeypatch)

    with pytest.raises(ValueError, match="solve output is not a regular file"):
        cli._solve(args)

    assert _hashes(tmp_path) == before
    assert destination.is_dir() if kind == "directory" else destination.is_symlink()


@pytest.mark.parametrize("artifact", ["parameters", "report"])
def test_solve_preserves_existing_output_refusal(tmp_path, monkeypatch, artifact):
    args = _arguments(tmp_path, force=False)
    destination = args.output
    if artifact == "report":
        destination = destination.with_suffix(".json")
    destination.write_bytes(b"previous artifact")
    _forbid_fit(monkeypatch)

    with pytest.raises(ValueError, match="solve output already exists; use --force"):
        cli._solve(args)

    assert destination.read_bytes() == b"previous artifact"


@pytest.fixture(scope="module")
def solved_snapshot(tmp_path_factory):
    directory = tmp_path_factory.mktemp("solve-publication")
    urdf = directory / "nominal.urdf"
    urdf.write_bytes(example_urdf().read_bytes())
    nominal = NumpyBackend.from_urdf(urdf, end_link="ee_link")
    joints = np.random.default_rng(902).uniform(-1.5, 1.5, (32, nominal.dof))
    for name, seed in (("first.npz", 905), ("second.npz", 906)):
        truth = nominal.perturbed(seed=seed)
        observations = np.array([truth.forward(row) for row in joints])
        KinematicDataset(
            joints,
            observations,
            "base_T_ee",
            np.arange(100, 132),
            nominal.joint_names,
        ).save(directory / name)
    assert cli._solve(_arguments(directory)) == 0
    load_fitted_model(directory / "fit.npz", urdf)
    return directory


@pytest.fixture
def replacement(solved_snapshot, tmp_path):
    directory = tmp_path / "replacement"
    shutil.copytree(solved_snapshot, directory)
    return directory, _arguments(directory, dataset="second.npz")


def _assert_preserved(directory, before):
    assert _hashes(directory) == before
    assert list(directory.glob(".calibration-pair-*")) == []
    load_fitted_model(directory / "fit.npz", directory / "nominal.urdf")


def test_solve_report_serialization_failure_preserves_previous_pair(
    replacement, monkeypatch
):
    directory, args = replacement
    before = _hashes(directory)

    def fail_serialization(*args, **kwargs):
        raise TypeError("injected report serialization failure")

    monkeypatch.setattr(cli.json, "dumps", fail_serialization)
    with pytest.raises(TypeError, match="injected report serialization"):
        cli._solve(args)

    _assert_preserved(directory, before)


def test_solve_report_staging_failure_preserves_previous_pair(replacement, monkeypatch):
    directory, args = replacement
    before = _hashes(directory)
    original_fsync = io_utils.os.fsync
    staged_writes = 0

    def fail_second_fsync(descriptor):
        nonlocal staged_writes
        staged_writes += 1
        if staged_writes == 2:
            raise OSError("injected report write failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(io_utils.os, "fsync", fail_second_fsync)
    with pytest.raises(OSError, match="injected report write failure"):
        cli._solve(args)

    assert staged_writes == 2
    _assert_preserved(directory, before)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_solve_second_install_failure_restores_or_removes_pair(
    replacement, monkeypatch, existing, failure
):
    directory, args = replacement
    if not existing:
        args.output.unlink()
        args.output.with_suffix(".json").unlink()
    before = _hashes(directory)
    original_replace = io_utils.os.replace

    def fail_report_install(source, destination):
        if Path(source).name == "1.new":
            raise failure("injected report installation failure")
        return original_replace(source, destination)

    monkeypatch.setattr(io_utils.os, "replace", fail_report_install)
    with pytest.raises(failure, match="injected report installation failure"):
        cli._solve(args)

    assert _hashes(directory) == before
    assert list(directory.glob(".calibration-pair-*")) == []
    if existing:
        load_fitted_model(args.output, args.urdf)


@pytest.mark.parametrize(
    ("existing", "operation"),
    [
        (True, "first-backup"),
        (False, "first-install"),
        (True, "first-install"),
        (False, "second-install"),
        (True, "second-install"),
    ],
)
def test_solve_interrupt_after_replace_restores_or_removes_pair(
    replacement, monkeypatch, existing, operation
):
    directory, args = replacement
    if not existing:
        args.output.unlink()
        args.output.with_suffix(".json").unlink()
    before = _hashes(directory)
    original_replace = io_utils.os.replace
    armed = True

    def interrupt_after_replace(source, destination):
        nonlocal armed
        result = original_replace(source, destination)
        selected = (
            Path(destination).name == "0.previous"
            if operation == "first-backup"
            else Path(source).name
            == ("0.new" if operation == "first-install" else "1.new")
        )
        if armed and selected:
            armed = False
            raise KeyboardInterrupt("injected interruption after replacement")
        return result

    monkeypatch.setattr(io_utils.os, "replace", interrupt_after_replace)
    with pytest.raises(
        KeyboardInterrupt, match="injected interruption after replacement"
    ):
        cli._solve(args)

    assert not armed
    assert _hashes(directory) == before
    assert list(directory.glob(".calibration-pair-*")) == []
    if existing:
        load_fitted_model(args.output, args.urdf)


def test_solve_failed_rollback_retains_recoverable_backups(replacement, monkeypatch):
    directory, args = replacement
    before = _hashes(directory)
    original_replace = io_utils.os.replace

    def fail_report_install_and_restore(source, destination):
        if Path(source).name == "1.new":
            raise OSError("injected report installation failure")
        if Path(source).name == "0.previous":
            raise OSError("injected previous model recovery failure")
        return original_replace(source, destination)

    monkeypatch.setattr(io_utils.os, "replace", fail_report_install_and_restore)
    with pytest.raises(
        OSError, match="rollback failed; previous files retained at"
    ) as caught:
        cli._solve(args)

    backups = list(directory.glob(".calibration-pair-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert str(backup) in str(caught.value)
    for index, name in enumerate(("fit.npz", "fit.json")):
        assert (
            hashlib.sha256((backup / f"{index}.previous").read_bytes()).hexdigest()
            == before[name]
        )
    assert not args.output.exists()
    assert not args.output.with_suffix(".json").exists()


def test_successful_force_replacement_publishes_a_bound_pair(replacement):
    directory, args = replacement
    before = _hashes(directory)

    assert cli._solve(args) == 0

    after = _hashes(directory)
    for name in ("first.npz", "second.npz", "nominal.urdf"):
        assert after[name] == before[name]
    assert after["fit.npz"] != before["fit.npz"]
    report = json.loads(args.output.with_suffix(".json").read_text())
    assert report["parameters_npz_sha256"] == after["fit.npz"]
    assert report["dataset_sha256"] == after["second.npz"]
    fitted = load_fitted_model(args.output, args.urdf)
    truth = NumpyBackend.from_urdf(args.urdf, end_link="ee_link").perturbed(seed=906)
    source = KinematicDataset.load(args.dataset)
    np.testing.assert_allclose(
        predict_poe(fitted.screw_axes, fitted.zero_pose, source.joint_positions),
        np.array([truth.forward(row) for row in source.joint_positions]),
        atol=2e-7,
    )
    assert list(directory.glob(".calibration-pair-*")) == []
