"""Crash-safe output helpers for calibration artifacts."""

import json
import os
import shutil
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


def atomic_write_pair(
    first_path: Path | str,
    first_content: bytes,
    second_path: Path | str,
    second_content: bytes,
) -> None:
    """Stage sibling files and restore the previous pair if publication fails.

    Rollback handles ordinary I/O failures and Python interruptions. The two
    replacements are not atomic across process crashes; bound hashes let readers
    reject an interrupted pair. Failed rollback retains its backup directory.
    """
    destinations = (Path(first_path), Path(second_path))
    if destinations[0].parent.resolve() != destinations[1].parent.resolve():
        raise ValueError("paired outputs must share a parent directory")
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("paired outputs must use different paths")
    for destination in destinations:
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ValueError(f"paired output is not a regular file: {destination}")
    parent = destinations[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".calibration-pair-", dir=parent))
    retain_stage = False
    try:
        for index, content in enumerate((first_content, second_content)):
            staged_path = stage / f"{index}.new"
            with staged_path.open("wb") as stream:
                staged_path.chmod(0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        backup_attempts, install_attempts = [], []
        try:
            for index, destination in enumerate(destinations):
                if destination.exists():
                    # Record intent before replacement: an interruption can
                    # arrive after the syscall has already moved the file.
                    backup_attempts.append(index)
                    os.replace(destination, stage / f"{index}.previous")
            for index, destination in enumerate(destinations):
                install_attempts.append(index)
                os.replace(stage / f"{index}.new", destination)
        except BaseException:
            try:
                for index in reversed(install_attempts):
                    destinations[index].unlink(missing_ok=True)
                for index in backup_attempts:
                    previous = stage / f"{index}.previous"
                    if previous.exists():
                        os.replace(previous, destinations[index])
            except BaseException as recovery_error:
                retain_stage = True
                raise OSError(
                    f"calibration output rollback failed; previous files retained at {stage}"
                ) from recovery_error
            raise
    finally:
        if not retain_stage:
            shutil.rmtree(stage)


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
