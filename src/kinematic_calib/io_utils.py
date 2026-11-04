"""NumPy archive reads and staged publication of calibration artifacts."""

import json
import os
import shutil
import tempfile
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from zipfile import BadZipFile

import numpy as np


@contextmanager
def open_npz(path: Path | str) -> Iterator[np.lib.npyio.NpzFile]:
    """Open a pickle-free NPZ and report decoding failures with its path."""
    try:
        archive = np.load(path, allow_pickle=False)
    except (EOFError, ValueError, BadZipFile, zlib.error) as exc:
        raise ValueError(f"invalid NPZ archive {path}: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"expected an NPZ archive: {path}")
    with archive:
        try:
            yield archive
        except (EOFError, BadZipFile, zlib.error) as exc:
            raise ValueError(f"invalid NPZ archive {path}: {exc}") from exc


def refuse_input_output_aliases(
    inputs: Sequence[Path | None], outputs: Sequence[Path], command: str
) -> None:
    """Reject output paths that resolve or link to an input file."""
    for source in inputs:
        if source is None:
            continue
        for output in outputs:
            if source.resolve() == output.resolve() or (
                source.exists() and output.exists() and source.samefile(output)
            ):
                raise ValueError(
                    f"{command} output conflicts with input: {output} aliases {source}"
                )


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
    """Stage two files and restore the previous pair if publication fails.

    Rollback handles ordinary I/O failures and Python interruptions. The two
    replacements are not atomic across process crashes. Each parent has its own
    staging directory so replacements stay on the destination filesystem. Failed
    rollback retains the staging directories and their recoverable backups.
    """
    destinations = (Path(first_path), Path(second_path))
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("paired outputs must use different paths")
    for destination in destinations:
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ValueError(f"paired output is not a regular file: {destination}")
    stages = {}
    output_stages = []
    retain_stages = False
    try:
        for destination in destinations:
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = parent.resolve()
            if resolved_parent not in stages:
                stages[resolved_parent] = Path(
                    tempfile.mkdtemp(prefix=".calibration-pair-", dir=parent)
                )
            output_stages.append(stages[resolved_parent])
        for index, (content, stage) in enumerate(
            zip((first_content, second_content), output_stages)
        ):
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
                    os.replace(destination, output_stages[index] / f"{index}.previous")
            for index, destination in enumerate(destinations):
                install_attempts.append(index)
                os.replace(output_stages[index] / f"{index}.new", destination)
        except BaseException:
            try:
                for index in reversed(install_attempts):
                    destinations[index].unlink(missing_ok=True)
                for index in backup_attempts:
                    previous = output_stages[index] / f"{index}.previous"
                    if previous.exists():
                        os.replace(previous, destinations[index])
            except BaseException as recovery_error:
                retain_stages = True
                retained = ", ".join(str(stage) for stage in stages.values())
                raise OSError(
                    f"calibration output rollback failed; previous files retained at {retained}"
                ) from recovery_error
            raise
    finally:
        if not retain_stages:
            for stage in stages.values():
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
