"""Actual Viser serving and CLI lifecycle for recorded fit artifacts."""

import json
import socket
import urllib.request

import numpy as np
import pytest

from kinematic_calib.cli import main as calibration_main
from kinematic_calib.dataset import KinematicDataset, ObservationFrame
from kinematic_calibration.backends import viser as viewer_backend
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.cli import main
from kinematic_calibration.experiment import example_urdf

viser = pytest.importorskip("viser")


@pytest.fixture(scope="module")
def fitted_artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("fitted-viewer")
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=83)
    configurations = np.random.default_rng(84).uniform(-1.5, 1.5, (40, nominal.dof))
    # Measurements come from independent geometry, never the nominal planning FK.
    observations = np.asarray([truth.forward(row) for row in configurations])
    for name, rows in (("train", slice(0, 32)), ("validation", slice(32, 40))):
        KinematicDataset(
            configurations[rows],
            observations[rows],
            ObservationFrame.BASE_T_EE,
            np.arange(100, 140)[rows],
            nominal.joint_names,
        ).save(path / f"{name}.npz")
    assert (
        calibration_main(
            [
                "solve",
                "--backend",
                "numpy",
                "--model-mode",
                "robot-body",
                "--dataset",
                str(path / "train.npz"),
                "--urdf",
                str(example_urdf()),
                "--ee-name",
                "ee_link",
                "--output",
                str(path / "fit.npz"),
                "--quiet",
            ]
        )
        == 0
    )
    return path


def _assert_serves_html(server):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{server.get_port()}", timeout=3
    ) as response:
        assert response.status == 200
        assert b"<html" in response.read().lower()


def _assert_port_closed(port):
    with pytest.raises(OSError):
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            pass


@pytest.mark.parametrize("image_metadata", [False, True])
def test_fitted_viewer_serves_measurements_without_truth_or_default_target(
    fitted_artifacts, tmp_path, image_metadata
):
    result = fitted_artifacts / "fit.json"
    if image_metadata:
        report = json.loads(result.read_text())
        report["experiment_type"] = "image-kinematic-calibration"
        result = tmp_path / "image-source-fit.json"
        result.write_text(json.dumps(report))
    server = viewer_backend.create_fitted_viewer(
        fitted_artifacts / "fit.npz",
        fitted_artifacts / "validation.npz",
        example_urdf(),
        result=result,
        port=0,
    )
    port = server.get_port()
    try:
        _assert_serves_html(server)
        for name in (
            "/base",
            "/poses/nominal",
            "/poses/calibrated",
            "/poses/measured",
            "/robot/nominal",
        ):
            assert server.scene.get_handle_by_name(name) is not None
        for name in ("/poses/truth", "/robot/truth", "/current/truth", "/target"):
            assert server.scene.get_handle_by_name(name) is None
        data = KinematicDataset.load(fitted_artifacts / "validation.npz")
        measured_frame = server.scene.get_handle_by_name("/current/measured")
        assert measured_frame.visible
        np.testing.assert_allclose(
            measured_frame.position, data.observed_poses[0, :3, 3]
        )
        predicted_frame = server.scene.get_handle_by_name("/current/calibrated")
        np.testing.assert_allclose(
            predicted_frame.position, measured_frame.position, atol=1e-6
        )
    finally:
        server.stop()
    _assert_port_closed(port)


def test_view_fit_cli_stops_the_actual_server_after_duration(
    fitted_artifacts, monkeypatch
):
    create_viewer = viewer_backend.create_fitted_viewer
    ports = []

    def record_server(*args, **kwargs):
        server = create_viewer(*args, **kwargs)
        try:
            _assert_serves_html(server)
        except BaseException:
            server.stop()
            raise
        ports.append(server.get_port())
        return server

    monkeypatch.setattr(viewer_backend, "create_fitted_viewer", record_server)
    assert (
        main(
            [
                "view-fit",
                "--parameters",
                str(fitted_artifacts / "fit.npz"),
                "--dataset",
                str(fitted_artifacts / "validation.npz"),
                "--urdf",
                str(example_urdf()),
                "--duration",
                "0.05",
                "--port",
                "0",
            ]
        )
        == 0
    )
    assert len(ports) == 1
    _assert_port_closed(ports[0])


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_duration_is_rejected_before_loading_artifacts_or_starting_server(
    duration, monkeypatch, capsys
):
    def unexpected_server(*args, **kwargs):
        raise AssertionError(
            "invalid duration must be rejected before viewer initialization"
        )

    monkeypatch.setattr(viewer_backend, "create_fitted_viewer", unexpected_server)
    assert (
        main(
            [
                "view-fit",
                "--parameters",
                "missing.npz",
                "--dataset",
                "missing.npz",
                "--urdf",
                "missing.urdf",
                f"--duration={duration}",
            ]
        )
        == 2
    )
    assert "duration must be finite and positive" in capsys.readouterr().err


@pytest.mark.parametrize("artifact", ["parameters", "urdf"])
def test_wrong_fit_artifact_is_rejected_before_server_initialization(
    fitted_artifacts, tmp_path, monkeypatch, artifact
):
    parameters = fitted_artifacts / "fit.npz"
    urdf = example_urdf()
    if artifact == "parameters":
        replacement = tmp_path / "changed.npz"
        replacement.write_bytes(parameters.read_bytes() + b"changed")
        parameters = replacement
        message = "parameter NPZ SHA-256"
    else:
        replacement = tmp_path / "changed.urdf"
        replacement.write_bytes(urdf.read_bytes() + b"\n")
        urdf = replacement
        message = "source URDF SHA-256"

    def unexpected_server(*args, **kwargs):
        raise AssertionError(
            "invalid artifacts must be rejected before server initialization"
        )

    monkeypatch.setattr(viser, "ViserServer", unexpected_server)
    with pytest.raises(ValueError, match=message):
        viewer_backend.create_fitted_viewer(
            parameters,
            fitted_artifacts / "validation.npz",
            urdf,
            result=fitted_artifacts / "fit.json",
            port=0,
        )
