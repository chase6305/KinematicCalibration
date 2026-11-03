import urllib.request

import pytest

pytest.importorskip("viser")

from kinematic_calibration.backends.viser import create_viewer
from kinematic_calibration.experiment import run_experiment


def test_viewer_serves_experiment_and_target_mesh(tmp_path):
    run_experiment(tmp_path, samples=12, validation_samples=8)
    server = create_viewer(tmp_path, port=0)
    try:
        url = f"http://127.0.0.1:{server.get_port()}"
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.status == 200
            assert b"<html" in response.read().lower()
    finally:
        server.stop()
