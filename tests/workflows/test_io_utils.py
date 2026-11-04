import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset
from kinematic_calib.io_utils import (
    atomic_save_npy,
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_text,
    open_npz,
)


def test_atomic_write_text_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("old")

    atomic_write_text(output, "new")

    assert output.read_text() == "new"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_savez_uses_exact_destination_name(tmp_path: Path) -> None:
    output = tmp_path / "result.npz"

    atomic_savez_compressed(output, values=np.array([1.0, 2.0]))

    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], [1.0, 2.0])
    assert not (tmp_path / "result.npz.npz").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_save_npy_uses_exact_destination_name(tmp_path: Path) -> None:
    output = tmp_path / "transform.npy"
    expected = np.eye(4)

    atomic_save_npy(output, expected)

    np.testing.assert_array_equal(np.load(output, allow_pickle=False), expected)
    assert not (tmp_path / "transform.npy.npy").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_json_rejects_non_finite_values(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="JSON compliant"):
        atomic_write_json(output, {"value": float("nan")})

    assert not output.exists()


@pytest.fixture
def deflate_corrupt_dataset(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(2),
        joint_positions=np.zeros((2, 1)),
        observed_poses=np.repeat(np.eye(4)[None], 2, axis=0),
        observation_frame=np.asarray("base_T_ee"),
    )
    assert KinematicDataset.load(path).joint_positions.shape == (2, 1)
    content = bytearray(path.read_bytes())
    with ZipFile(path) as archive:
        member = archive.getinfo("schema_version.npy")
        assert member.compress_type == ZIP_DEFLATED
        name_size, extra_size = struct.unpack_from(
            "<HH", content, member.header_offset + 26
        )
        offset = member.header_offset + 30 + name_size + extra_size
    # A reserved DEFLATE block type fails immediately without expanding data.
    content[offset] = (content[offset] & ~0x06) | 0x06
    assert len(content) < 4096
    path.write_bytes(content)
    return path, bytes(content)


@pytest.mark.parametrize("phase", ["opening", "member"])
def test_open_npz_deflate_errors_keep_path_cause_and_close_archive(
    deflate_corrupt_dataset, monkeypatch, phase
):
    path, original = deflate_corrupt_dataset
    real_load = np.load
    streams = []

    def tracked_load(*args, **kwargs):
        archive = real_load(*args, **kwargs)
        streams.append(archive.zip.fp)
        if phase == "opening":
            # Exercise the opening guard with an actual eager decode failure.
            with archive:
                archive["schema_version"]
        return archive

    monkeypatch.setattr("kinematic_calib.io_utils.np.load", tracked_load)
    with pytest.raises(ValueError) as error:
        with open_npz(path) as archive:
            archive["schema_version"]

    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, zlib.error)
    assert len(streams) == 1 and streams[0].closed
    assert path.read_bytes() == original


def test_dataset_deflate_error_is_an_input_error_for_api_and_cli(
    deflate_corrupt_dataset, tmp_path
):
    path, original = deflate_corrupt_dataset
    with pytest.raises(ValueError) as error:
        KinematicDataset.load(path)
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, zlib.error)

    output = tmp_path / "inspection.json"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kinematic_calib.cli",
            "inspect",
            "--dataset",
            str(path),
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 2, completed.stderr
    assert "Traceback" not in completed.stderr
    assert str(path) in completed.stderr
    assert completed.stdout == ""
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}
