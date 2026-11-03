from pathlib import Path

import numpy as np
import pytest

from kinematic_calib.io_utils import (
    atomic_save_npy,
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_text,
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
