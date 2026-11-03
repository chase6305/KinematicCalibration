"""Offline camera captures to measured Camera_T_Target datasets.

This module never loads a robot model. Joint readings are paired with board
poses estimated from image evidence, using an explicit timestamp tolerance.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    _validate_transforms,
    validated_joint_names,
)
from kinematic_calib.io_utils import atomic_write_json

from .targets import BoardConfig, BoardType, get_board_config
from .targets.detection import (
    _camera_parameters,
    detect_opencv_circles,
    detect_toolkit_board,
    pose_from_correspondences,
)


class CaptureImportError(ValueError):
    """Rejected capture batch, with a complete per-frame diagnostic report."""

    def __init__(self, message: str, report: dict, report_path: Path):
        super().__init__(message)
        self.report = report
        self.report_path = report_path


def _number(value, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return value


def _sample_id(value) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError("sample_id must be a nonnegative int64 integer")
    return value


def _validate_manifest(manifest: dict, root: Path, board: BoardConfig | None):
    if not isinstance(manifest, dict):
        raise ValueError("capture manifest must be a JSON object")
    version = manifest.get("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("unsupported capture manifest schema")
    if manifest.get("observation_frame", "camera_T_target") != "camera_T_target":
        raise ValueError("capture observation_frame must be camera_T_target")
    if manifest.get("camera_convention", "opencv-optical") != "opencv-optical":
        raise ValueError(
            "camera_convention must be opencv-optical (+X right, +Y down, +Z forward)"
        )
    if (
        manifest.get("joint_unit", "rad") != "rad"
        or manifest.get("translation_unit", "m") != "m"
    ):
        raise ValueError("capture units must be radians and metres")
    if (
        "camera_matrix" not in manifest
        or "distortion" not in manifest
        or "board_type" not in manifest
    ):
        raise ValueError("manifest requires camera_matrix, distortion and board_type")
    if manifest["distortion"] is None:
        raise ValueError("distortion must contain explicit calibrated coefficients")
    intrinsic, distortion = _camera_parameters(
        manifest["camera_matrix"], manifest["distortion"]
    )
    selected = get_board_config(manifest["board_type"]) if board is None else board
    if (
        not isinstance(selected, BoardConfig)
        or selected.board_type.value != manifest["board_type"]
    ):
        raise ValueError("board configuration must match manifest board_type")
    if len(selected.points) < 4:
        raise ValueError("full camera board pose requires at least four control points")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a nonempty list")
    ids, image_paths, validated = set(), set(), []
    dof = None
    for row, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample row {row} must be an object")
        sample_id = _sample_id(sample.get("sample_id"))
        if sample_id in ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        if sample.get("observation_frame", "camera_T_target") != "camera_T_target":
            raise ValueError(
                f"sample {sample_id} has an inconsistent observation_frame"
            )
        if (
            sample.get("board_type", selected.board_type.value)
            != selected.board_type.value
        ):
            raise ValueError(f"sample {sample_id} has an inconsistent board identity")
        raw_joints = np.asarray(sample.get("joint_positions"))
        if (
            raw_joints.dtype.kind not in "iuf"
            or raw_joints.ndim != 1
            or not len(raw_joints)
            or not np.isfinite(raw_joints).all()
        ):
            raise ValueError(
                f"sample {sample_id} joint_positions must be a finite real vector"
            )
        if dof is None:
            dof = len(raw_joints)
        elif len(raw_joints) != dof:
            raise ValueError(
                f"sample {sample_id} joint column count differs from earlier samples"
            )
        joint_time = _number(
            sample.get("joint_timestamp_s"), f"sample {sample_id} joint_timestamp_s"
        )
        image_time = _number(
            sample.get("image_timestamp_s"), f"sample {sample_id} image_timestamp_s"
        )
        _number(joint_time - image_time, f"sample {sample_id} timestamp difference")
        has_image, has_points = "image" in sample, "image_points" in sample
        if has_image == has_points:
            raise ValueError(
                f"sample {sample_id} requires exactly one image or image_points source"
            )
        if "point_ids" in sample:
            ids_for_points = sample["point_ids"]
            if (
                not isinstance(ids_for_points, list)
                or any(type(value) is not int for value in ids_for_points)
                or ids_for_points != list(range(len(selected.points)))
            ):
                raise ValueError(
                    f"sample {sample_id} point_ids must match board.points row-major order"
                )
        image_path = None
        if has_image:
            image_name = sample["image"]
            if (
                not isinstance(image_name, str)
                or not image_name
                or Path(image_name).is_absolute()
            ):
                raise ValueError(f"sample {sample_id} image must be a relative path")
            image_path = (root / image_name).resolve()
            if image_path in image_paths:
                raise ValueError(
                    f"sample {sample_id} reuses an image from another sample"
                )
            image_paths.add(image_path)
        validated.append(
            (
                sample_id,
                raw_joints.astype(float),
                joint_time,
                image_time,
                sample,
                image_path,
            )
        )
    names = manifest.get("joint_names")
    if (
        not isinstance(names, list)
        or len(names) != dof
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != dof
    ):
        raise ValueError("joint_names must uniquely identify every joint column")
    validated_joint_names(names, dof)
    for _, _, _, _, sample, _ in validated:
        if "joint_names" in sample and sample["joint_names"] != names:
            raise ValueError(
                "sample joint_names do not match manifest joint column identity"
            )
    return intrinsic, distortion, selected, names, validated


def _read_image(path: Path):
    import cv2 as cv

    contents = path.read_bytes()
    try:
        image = cv.imdecode(
            np.frombuffer(contents, dtype=np.uint8), cv.IMREAD_UNCHANGED
        )
    except cv.error as exc:
        raise ValueError(f"image could not be decoded: {path.name}") from exc
    if image is None:
        raise ValueError(f"image could not be decoded: {path.name}")
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
    return image, hashlib.sha256(contents).hexdigest()


def _preflight(paths: tuple[Path, ...], force: bool):
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"capture output is not a regular file: {path}")
        if path.exists() and not force:
            raise FileExistsError(
                f"capture output exists: {path}; use force to replace"
            )


def _publish(stage: Path, destinations: dict[str, Path], force: bool):
    """Publish the data/report pair and restore existing files after failures."""
    _preflight(tuple(destinations.values()), force)
    backup = Path(tempfile.mkdtemp(prefix=".captures-previous-", dir=stage.parent))
    backup_attempts, install_attempts = [], []
    try:
        for name, destination in destinations.items():
            if destination.exists():
                # Track intent even if an interruption follows the actual move.
                backup_attempts.append(name)
                os.replace(destination, backup / name)
        for name, destination in destinations.items():
            install_attempts.append(name)
            os.replace(stage / name, destination)
    except BaseException:
        try:
            for name in reversed(install_attempts):
                destinations[name].unlink(missing_ok=True)
            for name in backup_attempts:
                previous = backup / name
                if previous.exists():
                    os.replace(previous, destinations[name])
        except BaseException as recovery_error:
            raise OSError(
                f"capture rollback failed; previous files retained at {backup}"
            ) from recovery_error
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)


def import_captures(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    max_sync_error_s: float = 0.03,
    max_reprojection_rms_px: float = 2.0,
    detector: str = "auto",
    board: BoardConfig | None = None,
    allow_partial: bool = False,
    allow_legacy_profile: bool = False,
    force: bool = False,
) -> tuple[KinematicDataset, dict]:
    """Import archived images or ordered detections, without moving hardware.

    ``image_points`` must follow ``board.points`` row-major identity. Optional
    ``point_ids`` must be ``[0, ..., N-1]``. Timestamps are seconds on a shared
    clock; joint positions are actual encoder readings in radians. Image paths
    resolve relative to the manifest. Unknown distortion must not be silently
    treated as calibrated zero distortion; provide explicit coefficients.

    Any rejected frame fails the batch by default. The sibling ``.rejected.json``
    report records every frame and the dataset is preserved. Set ``allow_partial``
    explicitly to write the accepted subset and its complete ``.report.json``.
    All-rejected batches raise :class:`CaptureImportError` even with partial mode.
    """
    max_sync_error_s = _number(max_sync_error_s, "max_sync_error_s")
    if max_sync_error_s < 0:
        raise ValueError("max_sync_error_s must be nonnegative")
    max_reprojection_rms_px = _number(
        max_reprojection_rms_px, "max_reprojection_rms_px", positive=True
    )
    if detector not in {"auto", "opencv", "toolkit"}:
        raise ValueError("detector must be auto, opencv or toolkit")
    if any(
        type(value) is not bool
        for value in (allow_partial, allow_legacy_profile, force)
    ):
        raise ValueError(
            "allow_partial, allow_legacy_profile and force must be booleans"
        )
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("capture dataset output must have an .npz extension")
    report_path = output_path.with_suffix(".report.json")
    rejected_path = output_path.with_suffix(".rejected.json")
    destinations = (output_path, report_path, rejected_path)
    if manifest_path.resolve() in {path.resolve() for path in destinations}:
        raise ValueError("capture output conflicts with the source manifest")
    _preflight((output_path, report_path), force)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    intrinsic, distortion, board, names, samples = _validate_manifest(
        manifest, manifest_path.parent, board
    )
    image_sources = {sample[-1] for sample in samples if sample[-1] is not None}
    if image_sources & {path.resolve() for path in destinations}:
        raise ValueError("capture output conflicts with a source image")
    frames, joints, poses, accepted_ids = [], [], [], []
    for sample_id, q, joint_time, image_time, sample, image_path in samples:
        sync_error = abs(joint_time - image_time)
        entry = {
            "sample_id": sample_id,
            "joint_timestamp_s": joint_time,
            "image_timestamp_s": image_time,
            "sync_error_s": sync_error,
            "source": "image" if image_path is not None else "image_points",
            "detection_attempted": False,
            "detected": False,
            "rejected": True,
            "reprojection_rms_px": None,
            "rejection_reasons": [],
            "dataset_row": None,
        }
        if image_path is not None:
            entry["image"] = sample["image"]
        if sync_error > max_sync_error_s:
            entry["rejection_reasons"].append(
                "joint/image timestamp difference exceeds max_sync_error_s"
            )
        else:
            try:
                if image_path is None:
                    entry["detection_attempted"] = True
                    detection = pose_from_correspondences(
                        board,
                        sample["image_points"],
                        intrinsic,
                        distortion,
                        max_reprojection_rms_px=np.finfo(float).max,
                    )
                else:
                    image, entry["image_sha256"] = _read_image(image_path)
                    entry["detection_attempted"] = True
                    selected_detector = (
                        "opencv"
                        if detector == "auto"
                        and board.board_type == BoardType.OPENCV_ACIRCLE
                        else detector
                    )
                    if selected_detector == "opencv":
                        detection = detect_opencv_circles(
                            image,
                            intrinsic,
                            distortion,
                            board=board,
                            max_reprojection_rms_px=np.finfo(float).max,
                        )
                    else:
                        detection = detect_toolkit_board(
                            image,
                            intrinsic,
                            distortion,
                            board=board,
                            allow_legacy_profile=allow_legacy_profile,
                        )
                if detection is None:
                    entry["rejection_reasons"].append("board was not detected")
                else:
                    entry["detected"] = True
                    entry["detector_backend"] = detection.backend
                    diagnostics = getattr(detection, "pose_diagnostics", None)
                    if diagnostics is not None:
                        entry["pose_diagnostics"] = diagnostics
                        if diagnostics.get("ambiguous") is True:
                            entry["rejection_reasons"].append(
                                "planar board pose is ambiguous; collect a closer or more tilted view"
                            )
                    pose = _validate_transforms(
                        "Camera_T_Target", np.asarray(detection.camera_T_board)[None]
                    )[0]
                    if np.any((board.points @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
                        raise ValueError("detected board has nonpositive optical depth")
                    rms = detection.reprojection_rms_px
                    if rms is None:
                        entry["rejection_reasons"].append(
                            "detector provides no reprojection error; use ordered image_points for a verifiable pose"
                        )
                    else:
                        rms = _number(rms, "detected reprojection_rms_px")
                        if rms < 0:
                            raise ValueError(
                                "detected reprojection_rms_px must be nonnegative"
                            )
                        entry["reprojection_rms_px"] = rms
                        if rms > max_reprojection_rms_px:
                            entry["rejection_reasons"].append(
                                "board reprojection RMS exceeds max_reprojection_rms_px"
                            )
                        elif not entry["rejection_reasons"]:
                            entry["rejected"] = False
                            entry["dataset_row"] = len(joints)
                            entry["camera_T_target"] = pose.tolist()
                            joints.append(q)
                            poses.append(pose)
                            accepted_ids.append(sample_id)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                entry["rejection_reasons"].append(str(exc))
        frames.append(entry)
    rejected_count = sum(entry["rejected"] for entry in frames)
    report = {
        "schema_version": 1,
        "acquisition_type": "offline-camera-board",
        "observation_frame": "camera_T_target",
        "camera_convention": "opencv-optical",
        "joint_unit": "rad",
        "translation_unit": "m",
        "board_type": board.board_type.value,
        "board_points_m": board.points.tolist(),
        "joint_names": names,
        "dof": len(samples[0][1]),
        "camera_matrix": intrinsic.tolist(),
        "distortion": distortion.tolist(),
        "manifest_file": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "max_sync_error_s": max_sync_error_s,
        "max_reprojection_rms_px": max_reprojection_rms_px,
        "total_samples": len(frames),
        "accepted_samples": len(joints),
        "rejected_samples": rejected_count,
        "all_samples_accepted": rejected_count == 0,
        "partial": rejected_count > 0,
        "allow_partial": allow_partial,
        "dataset_written": False,
        "frames": frames,
    }
    if not joints or (rejected_count and not allow_partial):
        _preflight((rejected_path,), force)
        atomic_write_json(rejected_path, report)
        raise CaptureImportError(
            f"capture import rejected {rejected_count}/{len(frames)} frames; diagnostics: {rejected_path}",
            report,
            rejected_path,
        )
    dataset = KinematicDataset(
        np.asarray(joints),
        np.asarray(poses),
        ObservationFrame.CAMERA_T_TARGET,
        np.asarray(accepted_ids),
        tuple(names),
    )
    report["dataset_written"] = True
    report["dataset_file"] = output_path.name
    report["coverage"] = dataset.get_summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".captures-", dir=output_path.parent
    ) as temporary:
        stage = Path(temporary)
        dataset.save(stage / "dataset.npz")
        report["dataset_sha256"] = hashlib.sha256(
            (stage / "dataset.npz").read_bytes()
        ).hexdigest()
        atomic_write_json(stage / "report.json", report)
        _publish(stage, {"dataset.npz": output_path, "report.json": report_path}, force)
    return dataset, report


def verify_capture_report(dataset_path: Path | str, report_path: Path | str) -> dict:
    """Check a saved dataset against its import receipt without loading images.

    This verifies the report's identities, poses and recorded quality decisions.
    It does not re-detect source PNGs or verify the original capture manifest.
    """
    report = json.loads(Path(report_path).read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("acquisition_type") != "offline-camera-board"
        or report.get("dataset_written") is not True
    ):
        raise ValueError("unsupported or unwritten offline capture report")
    dataset_path = Path(dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if report.get("dataset_sha256") != digest:
        raise ValueError("capture report dataset SHA-256 mismatch")
    dataset = KinematicDataset.load(dataset_path)
    if (
        dataset.observation_frame != ObservationFrame.CAMERA_T_TARGET
        or report.get("observation_frame") != "camera_T_target"
        or report.get("camera_convention") != "opencv-optical"
        or report.get("joint_unit") != "rad"
        or report.get("translation_unit") != "m"
        or type(report.get("dof")) is not int
        or report["dof"] != dataset.joint_positions.shape[1]
        or report.get("joint_names") != list(dataset.joint_names or ())
    ):
        raise ValueError("capture report frame, units or joint identity mismatch")
    if report.get("distortion") is None:
        raise ValueError("capture report requires explicit distortion coefficients")
    _camera_parameters(report.get("camera_matrix"), report["distortion"])
    BoardType(report.get("board_type"))
    points = np.asarray(report.get("board_points_m"))
    if (
        points.dtype.kind not in "iuf"
        or points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or not np.isfinite(points).all()
        or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
    ):
        raise ValueError(
            "capture report board points must be finite noncollinear [N,3]"
        )
    max_sync = _number(report.get("max_sync_error_s"), "max_sync_error_s")
    max_rms = _number(
        report.get("max_reprojection_rms_px"), "max_reprojection_rms_px", positive=True
    )
    if max_sync < 0:
        raise ValueError("max_sync_error_s must be nonnegative")
    frames = report.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("capture report frames must be a nonempty list")
    seen, accepted_ids = set(), []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("capture report frame must be an object")
        sample_id = _sample_id(frame.get("sample_id"))
        if sample_id in seen:
            raise ValueError("capture report sample IDs must be unique")
        seen.add(sample_id)
        joint_time = _number(frame.get("joint_timestamp_s"), "joint_timestamp_s")
        image_time = _number(frame.get("image_timestamp_s"), "image_timestamp_s")
        sync = _number(frame.get("sync_error_s"), "sync_error_s")
        if sync < 0 or not np.isclose(
            sync, abs(joint_time - image_time), rtol=1e-12, atol=1e-12
        ):
            raise ValueError("capture report sync error differs from its timestamps")
        reasons = frame.get("rejection_reasons")
        if (
            type(frame.get("rejected")) is not bool
            or frame.get("source") not in ("image", "image_points")
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError("capture report has invalid frame status")
        if frame["rejected"]:
            if frame.get("dataset_row") is not None or not reasons:
                raise ValueError(
                    "rejected capture frame must have reasons and no dataset row"
                )
            continue
        row = len(accepted_ids)
        if (
            type(frame.get("dataset_row")) is not int
            or frame["dataset_row"] != row
            or row >= len(dataset.joint_positions)
            or frame.get("detected") is not True
            or frame.get("detection_attempted") is not True
            or reasons
            or sync < 0
            or sync > max_sync
        ):
            raise ValueError(
                "accepted capture frame has inconsistent row or quality status"
            )
        rms = _number(frame.get("reprojection_rms_px"), "reprojection_rms_px")
        diagnostics = frame.get("pose_diagnostics", {})
        if (
            rms < 0
            or rms > max_rms
            or not isinstance(diagnostics, dict)
            or diagnostics.get("ambiguous") is True
        ):
            raise ValueError(
                "accepted capture frame violates its reprojection or ambiguity gate"
            )
        pose = _validate_transforms(
            "camera_T_target", np.asarray(frame.get("camera_T_target"))[None]
        )[0]
        if not np.array_equal(pose, dataset.observed_poses[row]):
            raise ValueError("capture report pose differs from the dataset observation")
        if np.any((points @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
            raise ValueError("accepted capture frame has nonpositive optical depth")
        accepted_ids.append(sample_id)
    accepted, rejected = len(accepted_ids), len(frames) - len(accepted_ids)
    for field, expected in (
        ("total_samples", len(frames)),
        ("accepted_samples", accepted),
        ("rejected_samples", rejected),
    ):
        if type(report.get(field)) is not int or report[field] != expected:
            raise ValueError(f"capture report {field} is inconsistent")
    if (
        report.get("all_samples_accepted") is not (rejected == 0)
        or report.get("partial") is not (rejected > 0)
        or type(report.get("allow_partial")) is not bool
        or (rejected and not report["allow_partial"])
        or not np.array_equal(dataset.sample_indices, accepted_ids)
        or report.get("coverage") != dataset.get_summary()
    ):
        raise ValueError(
            "capture report subset identity, acceptance or coverage mismatch"
        )
    return {
        "integrity_verified": True,
        "scope": "dataset_and_import_report",
        "acquisition_type": "offline-camera-board",
        "dataset_sha256": digest,
        "total_samples": len(frames),
        "accepted_samples": accepted,
        "rejected_samples": rejected,
    }


__all__ = ["CaptureImportError", "import_captures", "verify_capture_report"]
