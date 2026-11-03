"""Crash-safe output helpers for calibration artifacts."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_bytes(path: Path | str, content: bytes) -> None:
    """Atomically replace *path* after fully writing and syncing *content*."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path | str, content: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path | str, value) -> None:
    """Atomically write standards-compliant, indented UTF-8 JSON."""
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_savez_compressed(path: Path | str, **arrays) -> None:
    """Atomically write a compressed NumPy archive without suffix rewriting."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(path: Path | str, array) -> None:
    """Atomically write one NumPy array without suffix rewriting."""
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
