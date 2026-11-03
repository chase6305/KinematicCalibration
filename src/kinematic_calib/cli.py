"""Command-line entry point for the robotkinecal workflow."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import platform
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from .arguments import build_parser
from .dataset import (
    DATASET_SCHEMA_VERSION,
    KinematicDataset,
    ObservationFrame,
    minimum_full_pose_samples,
    read_dataset_schema_version,
)
from .dataset import sample_sha256 as _dataset_sample_sha256
from .io_utils import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_pair,
    atomic_write_text,
    open_npz,
)
from .io_utils import refuse_input_output_aliases as _refuse_input_output_aliases
from .quality import evaluate_solution_quality as _evaluate_solution_quality
from .robot_model import (
    estimate_joint_zero_offsets,
    get_identifiable_urdf_joint_definitions,
    load_calibration_model,
)
from .service import FitOptions, fit_dataset

LOGGER = logging.getLogger(__name__)


def _jsonable(value):
    """Convert NumPy-rich solver output into JSON-native values."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_dumps_fixed(value, *, indent: int = 2) -> str:
    """Render strict JSON numbers without scientific notation."""

    def render(item, level: int) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (int, np.integer)):
            return str(int(item))
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not np.isfinite(number):
                raise ValueError("fixed-format JSON does not support non-finite values")
            rendered = format(number, ".17f").rstrip("0").rstrip(".")
            return "0" if rendered in {"", "-0"} else rendered
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False)
        if isinstance(item, dict):
            if not item:
                return "{}"
            padding = " " * (indent * (level + 1))
            closing = " " * (indent * level)
            entries = [
                f"{padding}{json.dumps(str(key), ensure_ascii=False)}: "
                f"{render(entry, level + 1)}"
                for key, entry in item.items()
            ]
            return "{\n" + ",\n".join(entries) + f"\n{closing}" + "}"
        if isinstance(item, (list, tuple, np.ndarray)):
            items = item.tolist() if isinstance(item, np.ndarray) else item
            if not items:
                return "[]"
            padding = " " * (indent * (level + 1))
            closing = " " * (indent * level)
            entries = [f"{padding}{render(entry, level + 1)}" for entry in items]
            return "[\n" + ",\n".join(entries) + f"\n{closing}" + "]"
        raise TypeError(f"unsupported fixed-format JSON value: {type(item).__name__}")

    return render(_jsonable(value), 0) + "\n"


def _solver_dataset_quality(
    dataset: KinematicDataset, args: argparse.Namespace, command: str
) -> tuple[dict, list[str]]:
    thresholds = (
        args.min_joint_span,
        args.min_translation_span,
        args.min_rotation_span,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError(
            f"{command} coverage thresholds must be finite and non-negative"
        )
    coverage = dataset.get_summary()
    warnings = dataset.get_quality_warnings(
        min_joint_span_rad=args.min_joint_span,
        min_translation_span_norm_m=args.min_translation_span,
        min_rotation_from_first_rad=args.min_rotation_span,
    )
    for warning in warnings:
        LOGGER.warning("%s dataset quality: %s", command, warning)
    if warnings and args.fail_on_low_diversity:
        raise ValueError(
            f"{command} dataset diversity requirements failed: " + "; ".join(warnings)
        )
    return coverage, warnings


def _merge_dataset_quality_gate(
    accepted: bool, quality_failures: list[str], dataset_warnings: list[str]
) -> tuple[bool, list[str]]:
    failures = list(quality_failures)
    failures.extend(f"dataset diversity: {warning}" for warning in dataset_warnings)
    return accepted and not dataset_warnings, failures


def _validate_quality_threshold_args(args: argparse.Namespace, command: str) -> None:
    thresholds = (
        args.max_position_error,
        args.max_orientation_error,
        args.max_condition_number,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError(f"{command} quality thresholds must be positive and finite")


def _load_transform(path: Path, name: str) -> np.ndarray:
    """Read one real 4x4 NPY transform without altering its stored values."""
    try:
        value = np.load(path, allow_pickle=False)
    except (EOFError, ValueError, BadZipFile) as exc:
        raise ValueError(f"invalid NPY transform {name} at {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        if isinstance(value, np.lib.npyio.NpzFile):
            value.close()
        raise ValueError(f"{name} must be stored as an NPY array: {path}")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values: {path}")
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 .npy matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant must be +1")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Fingerprint the workflow and its directly ported numerical solver."""
    digest = hashlib.sha256()
    package_root = Path(__file__).parent.parent
    for package in ("kinematic_calib", "robotkinecal"):
        for path in sorted((package_root / package).rglob("*.py")):
            digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    numpy_backend = package_root / "kinematic_calibration" / "backends" / "numpy.py"
    digest.update(numpy_backend.relative_to(package_root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(numpy_backend.read_bytes())
    return digest.hexdigest()


def _runtime_provenance() -> dict:
    packages = {}
    for distribution in (
        "kinematic-calibration",
        "numpy",
        "scipy",
        "spatialmath-python",
        "robotkinecal",
        "roboticstoolbox-python",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": packages,
        "implementation_sha256": _implementation_sha256(),
    }


def _urdf_joint_names_by_child(path: Path) -> dict[str, str]:
    """Map URDF child-link names to their joint names."""
    try:
        root = ET.parse(path).getroot()
        return {
            child.attrib["link"]: joint.attrib["name"]
            for joint in root.findall("joint")
            if (child := joint.find("child")) is not None
        }
    except (ET.ParseError, KeyError) as exc:
        raise ValueError(f"invalid URDF joint metadata in {path}: {exc}") from exc


def _refuse_existing_outputs(paths: Sequence[Path], force: bool, command: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise ValueError(
            f"{command} output already exists; use --force: " + ", ".join(existing)
        )


def _create_template(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.dof <= 0:
        raise ValueError("samples and dof must be positive")
    if args.output.suffix.lower() != ".npz":
        raise ValueError("template --output must use the .npz extension")
    poses = np.repeat(np.eye(4)[None], args.samples, axis=0)
    dataset = KinematicDataset(np.zeros((args.samples, args.dof)), poses, args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(args.output)
    LOGGER.info("Wrote dataset template: %s", args.output)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    if args.output is not None:
        _refuse_input_output_aliases((args.dataset,), (args.output,), "inspect")
    thresholds = (
        args.min_joint_span,
        args.min_translation_span,
        args.min_rotation_span,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("inspection thresholds must be finite and non-negative")
    dataset = KinematicDataset.load(args.dataset)
    summary = dataset.get_summary()
    dataset_schema_version = read_dataset_schema_version(args.dataset)
    summary["dataset_schema_version"] = dataset_schema_version
    summary["legacy_dataset_schema"] = dataset_schema_version != DATASET_SCHEMA_VERSION
    summary["dataset_sha256"] = _sha256(args.dataset)
    warnings = dataset.get_quality_warnings(
        min_joint_span_rad=args.min_joint_span,
        min_translation_span_norm_m=args.min_translation_span,
        min_rotation_from_first_rad=args.min_rotation_span,
    )
    summary["quality_thresholds"] = {
        "min_joint_span_rad": args.min_joint_span,
        "min_translation_span_norm_m": args.min_translation_span,
        "min_rotation_from_first_rad": args.min_rotation_span,
    }
    summary["quality_warnings"] = warnings
    rendered = json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, _jsonable(summary))
        LOGGER.info("Wrote dataset inspection: %s", args.output)
    else:
        print(rendered, end="")
    if warnings:
        for warning in warnings:
            LOGGER.warning("%s", warning)
    return 1 if warnings and args.fail_on_low_diversity else 0


def _split_dataset(args: argparse.Namespace) -> int:
    outputs = (args.train_output, args.validation_output, args.manifest)
    if args.train_output.suffix.lower() != ".npz":
        raise ValueError("split-dataset --train-output must use the .npz extension")
    if args.validation_output.suffix.lower() != ".npz":
        raise ValueError(
            "split-dataset --validation-output must use the .npz extension"
        )
    if args.manifest.suffix.lower() != ".json":
        raise ValueError("split-dataset --manifest must use the .json extension")
    resolved = [path.resolve() for path in (args.dataset, *outputs)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("dataset and split output paths must all be different")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise ValueError(
            "split output already exists; use --force: " + ", ".join(existing)
        )
    if args.validation_count is None:
        if (
            not np.isfinite(args.validation_fraction)
            or not 0.0 < args.validation_fraction < 1.0
        ):
            raise ValueError("validation fraction must be between 0 and 1")
    elif args.validation_count <= 0:
        raise ValueError("validation count must be positive")
    if args.search_attempts <= 0:
        raise ValueError("--search-attempts must be positive")
    split_coverage_thresholds = (
        args.min_joint_span,
        args.min_translation_span,
        args.min_rotation_span,
    )
    if any(
        not np.isfinite(value) or value < 0.0 for value in split_coverage_thresholds
    ):
        raise ValueError("split coverage thresholds must be finite and non-negative")

    dataset = KinematicDataset.load(args.dataset)
    duplicate_count = dataset.get_summary()["duplicate_sample_count"]
    if duplicate_count:
        raise ValueError(
            f"split source contains {duplicate_count} duplicate observations; run "
            "deduplicate-dataset first to prevent train/validation leakage"
        )
    sample_count = len(dataset.joint_positions)
    validation_count = (
        args.validation_count
        if args.validation_count is not None
        else max(1, int(round(sample_count * args.validation_fraction)))
    )
    if validation_count >= sample_count:
        raise ValueError("validation count must be smaller than the sample count")
    train_count = sample_count - validation_count
    minimum_train = minimum_full_pose_samples(dataset.joint_positions.shape[1])
    if train_count < minimum_train:
        raise ValueError(
            f"split leaves {train_count} training samples, but at least "
            f"{minimum_train} are required"
        )

    def subset(indices: np.ndarray) -> KinematicDataset:
        return KinematicDataset(
            dataset.joint_positions[indices],
            dataset.observed_poses[indices],
            dataset.observation_frame,
            dataset.sample_indices[indices],
            dataset.joint_names,
        )

    best_candidate = None
    attempts_evaluated = 0
    for attempt in range(args.search_attempts):
        candidate_seed = args.seed + attempt
        generator = np.random.default_rng(candidate_seed)
        candidate_validation_indices = np.sort(
            generator.choice(sample_count, size=validation_count, replace=False)
        )
        candidate_train_indices = np.setdiff1d(
            np.arange(sample_count),
            candidate_validation_indices,
            assume_unique=True,
        )
        candidate_train = subset(candidate_train_indices)
        candidate_validation = subset(candidate_validation_indices)
        candidate_train.validate_solver_sample_count()
        candidate_train_warnings = candidate_train.get_quality_warnings(
            min_joint_span_rad=args.min_joint_span,
            min_translation_span_norm_m=args.min_translation_span,
            min_rotation_from_first_rad=args.min_rotation_span,
        )
        candidate_validation_warnings = candidate_validation.get_quality_warnings(
            min_joint_span_rad=args.min_joint_span,
            min_translation_span_norm_m=args.min_translation_span,
            min_rotation_from_first_rad=args.min_rotation_span,
        )
        candidate = {
            "seed": candidate_seed,
            "train_indices": candidate_train_indices,
            "validation_indices": candidate_validation_indices,
            "train_dataset": candidate_train,
            "validation_dataset": candidate_validation,
            "train_warnings": candidate_train_warnings,
            "validation_warnings": candidate_validation_warnings,
            "warning_count": len(candidate_train_warnings)
            + len(candidate_validation_warnings),
        }
        attempts_evaluated += 1
        if (
            best_candidate is None
            or candidate["warning_count"] < best_candidate["warning_count"]
        ):
            best_candidate = candidate
        if candidate["warning_count"] == 0:
            break

    if best_candidate is None:
        raise RuntimeError("split candidate search produced no candidates")
    selected_seed = best_candidate["seed"]
    train_indices = best_candidate["train_indices"]
    validation_indices = best_candidate["validation_indices"]
    train_dataset = best_candidate["train_dataset"]
    validation_dataset = best_candidate["validation_dataset"]
    train_quality_warnings = best_candidate["train_warnings"]
    validation_quality_warnings = best_candidate["validation_warnings"]
    quality_warnings = []
    if validation_count < 5:
        quality_warnings.append(
            "fewer than 5 validation samples; accuracy statistics are weak"
        )
    quality_warnings.extend(
        f"training subset: {warning}" for warning in train_quality_warnings
    )
    quality_warnings.extend(
        f"validation subset: {warning}" for warning in validation_quality_warnings
    )
    if quality_warnings and args.fail_on_low_diversity:
        raise ValueError(
            "split diversity requirements failed: " + "; ".join(quality_warnings)
        )
    train_dataset.save(args.train_output)
    validation_dataset.save(args.validation_output)
    manifest = {
        "schema_version": 1,
        "report_type": "kinematic-dataset-split",
        "seed": selected_seed,
        "seed_requested": args.seed,
        "search_attempts_requested": args.search_attempts,
        "search_attempts_evaluated": attempts_evaluated,
        "validation_fraction_requested": args.validation_fraction,
        "validation_count_requested": args.validation_count,
        "samples": sample_count,
        "train_samples": train_count,
        "validation_samples": validation_count,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "train_source_sample_indices": train_dataset.sample_indices.tolist(),
        "validation_source_sample_indices": (
            validation_dataset.sample_indices.tolist()
        ),
        "train_coverage": train_dataset.get_summary(),
        "validation_coverage": validation_dataset.get_summary(),
        "coverage_thresholds": {
            "min_joint_span_rad": args.min_joint_span,
            "min_translation_span_norm_m": args.min_translation_span,
            "min_rotation_from_first_rad": args.min_rotation_span,
        },
        "train_quality_warnings": train_quality_warnings,
        "validation_quality_warnings": validation_quality_warnings,
        "quality_warnings": quality_warnings,
        "source_dataset_sha256": _sha256(args.dataset),
        "train_dataset_sha256": _sha256(args.train_output),
        "validation_dataset_sha256": _sha256(args.validation_output),
    }
    atomic_write_json(args.manifest, _jsonable(manifest))
    LOGGER.info(
        "Wrote dataset split: train=%s (%d), validation=%s (%d), manifest=%s",
        args.train_output,
        train_count,
        args.validation_output,
        validation_count,
        args.manifest,
    )
    return 0


def _deduplicate_dataset(args: argparse.Namespace) -> int:
    if args.dataset.suffix.lower() != ".npz" or args.output.suffix.lower() != ".npz":
        raise ValueError("deduplicate-dataset input and output must use .npz")
    if args.manifest.suffix.lower() != ".json":
        raise ValueError("deduplicate-dataset --manifest must use the .json extension")
    resolved = [path.resolve() for path in (args.dataset, args.output, args.manifest)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("deduplicate input and output paths must all be different")
    _refuse_existing_outputs(
        (args.output, args.manifest), args.force, "deduplicate-dataset"
    )

    dataset = KinematicDataset.load(args.dataset)
    source_summary = dataset.get_summary()
    removed_positions = source_summary["duplicate_sample_positions"]
    removed = set(removed_positions)
    kept_positions = np.asarray(
        [
            index
            for index in range(len(dataset.joint_positions))
            if index not in removed
        ],
        dtype=np.int64,
    )
    cleaned = KinematicDataset(
        dataset.joint_positions[kept_positions],
        dataset.observed_poses[kept_positions],
        dataset.observation_frame,
        dataset.sample_indices[kept_positions],
        dataset.joint_names,
    )
    cleaned.save(args.output)
    manifest = {
        "schema_version": 1,
        "report_type": "kinematic-dataset-deduplication",
        "source_dataset": str(args.dataset),
        "output_dataset": str(args.output),
        "source_dataset_sha256": _sha256(args.dataset),
        "output_dataset_sha256": _sha256(args.output),
        "source_samples": len(dataset.joint_positions),
        "output_samples": len(cleaned.joint_positions),
        "removed_sample_count": len(removed_positions),
        "kept_positions": kept_positions.tolist(),
        "kept_source_sample_indices": cleaned.sample_indices.tolist(),
        "removed_positions": removed_positions,
        "removed_source_sample_indices": source_summary[
            "duplicate_source_sample_indices"
        ],
        "duplicate_of_positions": source_summary["duplicate_of_positions"],
        "output_coverage": cleaned.get_summary(),
    }
    atomic_write_json(args.manifest, manifest)
    LOGGER.info(
        "Wrote deduplicated dataset: %s (%d -> %d samples)",
        args.output,
        len(dataset.joint_positions),
        len(cleaned.joint_positions),
    )
    LOGGER.info("Wrote deduplication manifest: %s", args.manifest)
    return 0


def _upgrade_dataset(args: argparse.Namespace) -> int:
    if args.dataset.suffix.lower() != ".npz" or args.output.suffix.lower() != ".npz":
        raise ValueError("upgrade-dataset input and output must use .npz")
    if args.manifest.suffix.lower() != ".json":
        raise ValueError("upgrade-dataset --manifest must use the .json extension")
    if args.dataset.resolve() == args.output.resolve():
        raise ValueError("upgrade-dataset refuses in-place replacement")
    _refuse_input_output_aliases(
        (args.dataset,), (args.output, args.manifest), "upgrade-dataset"
    )
    existing = [str(path) for path in (args.output, args.manifest) if path.exists()]
    if existing and not args.force:
        raise ValueError(
            "upgrade output already exists; use --force: " + ", ".join(existing)
        )

    source_schema_version = read_dataset_schema_version(args.dataset)
    dataset = KinematicDataset.load(args.dataset)
    source_hash = _sha256(args.dataset)
    dataset.save(args.output)
    manifest = {
        "schema_version": 1,
        "report_type": "kinematic-dataset-upgrade",
        "source_dataset": str(args.dataset),
        "output_dataset": str(args.output),
        "source_schema_version": source_schema_version,
        "output_schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset_sha256": source_hash,
        "output_dataset_sha256": _sha256(args.output),
        "samples": len(dataset.joint_positions),
        "dof": dataset.joint_positions.shape[1],
        "sample_indices": dataset.sample_indices.tolist(),
        "hash_change_note": (
            "upgrading changes the dataset SHA-256; calibration reports bound "
            "to the source dataset must not be paired with the upgraded file"
        ),
    }
    atomic_write_json(args.manifest, manifest)
    LOGGER.info(
        "Upgraded dataset schema %s -> %d: %s",
        "legacy" if source_schema_version is None else source_schema_version,
        DATASET_SCHEMA_VERSION,
        args.output,
    )
    LOGGER.info("Wrote dataset upgrade manifest: %s", args.manifest)
    return 0


def _solve(args: argparse.Namespace) -> int:
    if args.output.suffix.lower() != ".npz":
        raise ValueError("solve --output must use the .npz extension")
    _validate_quality_threshold_args(args, "solve")
    if args.max_iterations <= 0:
        raise ValueError("solve --max-iterations must be positive")
    if not np.isfinite(args.damping) or args.damping < 0.0:
        raise ValueError("solve --damping must be finite and non-negative")
    output_paths = (args.output, args.output.with_suffix(".json"))
    for path in output_paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"solve output is not a regular file: {path}")
    _refuse_input_output_aliases(
        (args.dataset, args.urdf, args.base_t_camera, args.ee_t_target),
        output_paths,
        "solve",
    )
    _refuse_existing_outputs(output_paths, args.force, "solve")
    from spatialmath import SE3

    from .validation import evaluate_saved_poe

    backend = getattr(args, "backend", "rtb")
    if backend == "numpy" and any(
        (
            getattr(args, "experimental_nominal_allocation", False),
            getattr(args, "experimental_svd_gauge_allocation", False),
            getattr(args, "geometry_anchor_joint", None),
        )
    ):
        raise ValueError("URDF geometry allocation options require --backend rtb")

    dataset = KinematicDataset.load(args.dataset)
    source_observation_frame = dataset.observation_frame.value
    dataset_coverage, dataset_quality_warnings = _solver_dataset_quality(
        dataset, args, "solve"
    )
    source_sample_sha256 = _dataset_sample_sha256(dataset)
    source_dataset_schema_version = read_dataset_schema_version(args.dataset)
    requested_mode = args.model_mode
    self_calibrating = requested_mode in {"self-calibrating", "composite"}
    if getattr(args, "experimental_nominal_allocation", False) and getattr(
        args, "experimental_svd_gauge_allocation", False
    ):
        raise ValueError(
            "--experimental-nominal-allocation and "
            "--experimental-svd-gauge-allocation are mutually exclusive"
        )
    if getattr(args, "geometry_anchor_joint", None) and not getattr(
        args, "experimental_nominal_allocation", False
    ):
        raise ValueError(
            "--geometry-anchor-joint requires --experimental-nominal-allocation"
        )
    if self_calibrating and getattr(args, "geometry_anchor_joint", None):
        raise ValueError(
            "--geometry-anchor-joint requires --model-mode robot-body because "
            "self-calibrating geometry has no complete URDF parameter allocation"
        )
    if self_calibrating and getattr(args, "experimental_svd_gauge_allocation", False):
        raise ValueError(
            "--experimental-svd-gauge-allocation requires --model-mode "
            "robot-body and complete URDF geometry"
        )
    external_transform_hashes = None
    if requested_mode == "composite":
        LOGGER.warning("--model-mode composite is deprecated; use self-calibrating")
    if args.model_mode == "robot-body":
        if dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET:
            if args.base_t_camera is None or args.ee_t_target is None:
                raise ValueError(
                    "robot-body mode requires --base-t-camera and --ee-t-target "
                    "for camera_T_target observations"
                )
            base_T_camera = _load_transform(args.base_t_camera, "base_T_camera")
            ee_T_target = _load_transform(args.ee_t_target, "ee_T_target")
            external_transform_hashes = {
                "base_T_camera_sha256": _sha256(args.base_t_camera),
                "ee_T_target_sha256": _sha256(args.ee_t_target),
            }
            dataset = dataset.as_robot_body(base_T_camera, ee_T_target)
        elif args.base_t_camera is not None or args.ee_t_target is not None:
            raise ValueError(
                "robot-body mode does not accept external transforms when the "
                "dataset already contains base_T_ee observations"
            )
    else:
        if args.base_t_camera is not None or args.ee_t_target is not None:
            raise ValueError(
                "self-calibrating mode does not accept --base-t-camera or --ee-t-target"
            )
        if dataset.observation_frame != ObservationFrame.CAMERA_T_TARGET:
            raise ValueError(
                "self-calibrating mode requires camera_T_target observations"
            )

    model = load_calibration_model(
        args.urdf,
        args.ee_name,
        backend=backend,
        base_link=getattr(args, "base_link", None),
        verbose=not args.quiet,
    )
    calibrator = model.calibrator
    fit = fit_dataset(
        dataset,
        calibrator,
        options=FitOptions(
            model_mode="self-calibrating" if self_calibrating else "robot-body",
            max_iterations=args.max_iterations,
            damping=args.damping,
        ),
    )
    chain_dof = int(calibrator.N_JOINTS)
    estimated_camera_t_base = fit.estimated_camera_t_base
    estimated_ee_t_target = fit.estimated_ee_t_target
    extrinsic_initialization = fit.extrinsic_initialization
    result = fit.result
    screw_axes = fit.screw_axes
    zero_pose = fit.zero_pose
    final_iteration = result.iteration_results[-1]
    # Iteration statistics describe the pre-update linearization. Assess the
    # final parameters that will be saved against their actual training poses.
    training_metrics = evaluate_saved_poe(
        screw_axes,
        zero_pose.A,
        fit.dataset.joint_positions,
        fit.dataset.observed_poses,
        sample_indices=fit.dataset.sample_indices,
    )
    final_twist_error = final_iteration.post_update_twist_errors_norm
    if final_twist_error is None:
        final_twist_error = final_iteration.pre_update_twist_errors_norm
    full_rank = int(final_iteration.matrix_rank) == int(final_iteration.N_PARAMS)
    accepted, quality_failures = _evaluate_solution_quality(
        converged=bool(result.has_converged),
        full_rank=full_rank,
        position_error_max_m=training_metrics["position_error_max_m"],
        orientation_error_max_rad=training_metrics["orientation_error_max_rad"],
        condition_number=float(final_iteration.condition_number),
        max_position_error=args.max_position_error,
        max_orientation_error=args.max_orientation_error,
        max_condition_number=args.max_condition_number,
    )
    accepted, quality_failures = _merge_dataset_quality_gate(
        accepted, quality_failures, dataset_quality_warnings
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "backend": np.asarray(model.backend),
        "base_link": np.asarray(model.base_link),
        "ee_name": np.asarray(args.ee_name),
        "joint_names": np.asarray(model.joint_names),
        "joint_identity_verified": np.asarray(fit.joint_identity["verified"]),
        "parameter_type": np.asarray("kinematic-poe"),
        "screw_axes": np.asarray(screw_axes),
        "zero_configuration_pose": np.asarray(zero_pose.A),
        "model_mode": np.asarray(
            "self-calibrating" if self_calibrating else "robot-body"
        ),
        "observation_frame": np.asarray(dataset.observation_frame.value),
        "source_observation_frame": np.asarray(source_observation_frame),
        "termination_reason": np.asarray(result.termination_reason),
        "source_dataset_sha256": np.asarray(_sha256(args.dataset)),
        "source_sample_sha256": source_sample_sha256,
        "source_sample_indices": np.asarray(dataset.sample_indices, dtype=np.int64),
        "source_dataset_schema_version": np.asarray(
            "legacy"
            if source_dataset_schema_version is None
            else str(source_dataset_schema_version)
        ),
        "source_urdf_sha256": np.asarray(_sha256(args.urdf)),
        "implementation_sha256": np.asarray(_implementation_sha256()),
    }
    if self_calibrating:
        camera_t_base = SE3(estimated_camera_t_base)
        base_gauge_axes = np.asarray(
            [camera_t_base.inv().Ad() @ axis for axis in screw_axes]
        )
        output_fields.update(
            estimated_camera_T_base=np.asarray(estimated_camera_t_base),
            estimated_EE_T_target=np.asarray(estimated_ee_t_target),
            intermediate_screw_axes_base_gauge=base_gauge_axes[1:],
            identifiable_joint_indices=np.arange(1, chain_dof),
        )
    elif external_transform_hashes is not None:
        output_fields.update(
            source_base_T_camera_sha256=np.asarray(
                external_transform_hashes["base_T_camera_sha256"]
            ),
            source_EE_T_target_sha256=np.asarray(
                external_transform_hashes["ee_T_target_sha256"]
            ),
        )
    parameter_buffer = io.BytesIO()
    np.savez_compressed(parameter_buffer, **output_fields)
    parameter_bytes = parameter_buffer.getvalue()
    summary_path = args.output.with_suffix(".json")
    summary = {
        "schema_version": 1,
        "calibration_type": "kinematic-geometry",
        "provenance": _runtime_provenance(),
        "backend": model.backend,
        "base_link": model.base_link,
        "joint_names": list(model.joint_names),
        "joint_identity_verified": fit.joint_identity["verified"],
        "joint_identity": fit.joint_identity,
        "model_mode": "self-calibrating" if self_calibrating else "robot-body",
        "observation_frame": dataset.observation_frame.value,
        "source_observation_frame": source_observation_frame,
        "samples": len(dataset.joint_positions),
        "dof": chain_dof,
        "robot_total_joints": model.total_joints,
        "ee_name": args.ee_name,
        "termination_reason": result.termination_reason,
        "converged": bool(result.has_converged),
        "iterations": int(result.nb_iterations_executed),
        "position_error_mean_m": training_metrics["position_error_mean_m"],
        "position_error_max_m": training_metrics["position_error_max_m"],
        "orientation_error_mean_rad": training_metrics["orientation_error_mean_rad"],
        "orientation_error_max_rad": training_metrics["orientation_error_max_rad"],
        "final_twist_error_norm": float(final_twist_error),
        "regressor_rank": int(final_iteration.matrix_rank),
        "parameter_count": int(final_iteration.N_PARAMS),
        "full_rank": full_rank,
        "condition_number": float(final_iteration.condition_number),
        "accepted": accepted,
        "quality_thresholds": {
            "max_position_error_m": args.max_position_error,
            "max_orientation_error_rad": args.max_orientation_error,
            "max_condition_number": args.max_condition_number,
            "min_joint_span_rad": args.min_joint_span,
            "min_translation_span_norm_m": args.min_translation_span,
            "min_rotation_from_first_rad": args.min_rotation_span,
        },
        "quality_failures": quality_failures,
        "dataset_quality_warnings": dataset_quality_warnings,
        "dataset_sha256": _sha256(args.dataset),
        "dataset_schema_version": source_dataset_schema_version,
        "urdf_sha256": _sha256(args.urdf),
        "parameters_npz_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "dataset_coverage": dataset_coverage,
    }
    if external_transform_hashes is not None:
        summary.update(external_transform_hashes)
    if not self_calibrating and model.link_frames_available:
        if external_transform_hashes is not None:
            summary.update(external_transform_hashes)
        from .geometry_postprocess import postprocess_urdf_definitions

        raw_urdf_definitions = calibrator.get_urdf_xyzrpy(result)
        definitions, geometry_postprocessing = postprocess_urdf_definitions(
            raw_urdf_definitions,
            args.urdf,
            dataset.joint_positions,
            anchor_joint=getattr(args, "geometry_anchor_joint", None),
            enable_nominal_allocation=getattr(
                args, "experimental_nominal_allocation", False
            ),
            enable_svd_gauge_allocation=getattr(
                args, "experimental_svd_gauge_allocation", False
            ),
        )
        summary["raw_solver_urdf_joint_definitions"] = _jsonable(raw_urdf_definitions)
        summary["urdf_joint_definitions"] = _jsonable(definitions)
        summary["geometry_postprocessing"] = _jsonable(geometry_postprocessing)
        summary["full_urdf_export_supported"] = True
        summary["partial_urdf_export_supported"] = False
    elif self_calibrating:
        identifiable_urdf_definitions = (
            get_identifiable_urdf_joint_definitions(
                calibrator,
                estimated_camera_t_base,
                _urdf_joint_names_by_child(args.urdf),
            )
            if model.link_frames_available
            else None
        )
        summary["external_extrinsics_required"] = False
        summary["extrinsic_initialization"] = extrinsic_initialization
        summary["estimated_camera_T_base"] = _jsonable(estimated_camera_t_base)
        summary["estimated_EE_T_target"] = _jsonable(estimated_ee_t_target)
        summary["identifiable_joint_indices"] = list(range(1, chain_dof))
        summary["identifiable_urdf_joint_definitions"] = identifiable_urdf_definitions
        summary["identifiability_note"] = (
            "zero-based joints after the first revolute joint are reported in "
            "the nominal Base gauge; the first joint relative to Base/Camera "
            "and the fixed transform after the last joint relative to EE/Target "
            "are not uniquely separable without external constraints"
        )
        summary["urdf_joint_definitions"] = None
        summary["full_urdf_export_supported"] = False
        summary["partial_urdf_export_supported"] = True
        summary["full_urdf_export_disabled_reason"] = (
            "self-calibrating Camera/Target parameters contain unobservable "
            "base and terminal gauge freedom"
        )
    if not model.link_frames_available:
        summary.update(
            urdf_joint_definitions=None,
            identifiable_urdf_joint_definitions=None,
            full_urdf_export_supported=False,
            partial_urdf_export_supported=False,
            full_urdf_export_disabled_reason=(
                "numpy backend calibrates space POE axes and home pose; calibrated "
                "intermediate URDF link frames are unavailable. Use the POE model "
                "for prediction/validation or select --backend rtb for URDF export."
            ),
        )
    summary_bytes = (
        json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _refuse_existing_outputs(output_paths, args.force, "solve")
    atomic_write_pair(args.output, parameter_bytes, summary_path, summary_bytes)
    LOGGER.info("Wrote calibration parameters: %s", args.output)
    LOGGER.info("Wrote calibration summary: %s", summary_path)
    if self_calibrating and not args.quiet:
        LOGGER.info(
            "Calibrated identifiable intermediate joints: "
            "zero_based_indices=%s, "
            "intermediate_screw_axes_base_gauge=\n%s",
            summary["identifiable_joint_indices"],
            np.array2string(
                output_fields["intermediate_screw_axes_base_gauge"],
                precision=8,
                suppress_small=False,
            ),
        )
        LOGGER.info(
            "Calibrated full composite screw axes (Camera/Target gauge)=\n%s",
            np.array2string(np.asarray(screw_axes), precision=8, suppress_small=False),
        )
        LOGGER.info(
            "Calibrated identifiable URDF-style joint definitions=\n%s",
            json.dumps(summary["identifiable_urdf_joint_definitions"], indent=2),
        )
    elif not args.quiet:
        LOGGER.info(
            "Calibrated robot-body screw axes (Base/EE gauge)=\n%s",
            np.array2string(np.asarray(screw_axes), precision=8, suppress_small=False),
        )
        LOGGER.info(
            "Calibrated URDF joint definitions=\n%s",
            json.dumps(summary["urdf_joint_definitions"], indent=2),
        )
    for failure in quality_failures:
        LOGGER.error("Calibration rejected: %s", failure)
    return 0 if accepted else 1


def _zero_offset(args: argparse.Namespace) -> int:
    if args.output.suffix.lower() != ".json":
        raise ValueError("zero-offset --output must use the .json extension")
    _refuse_input_output_aliases(
        (args.dataset, args.urdf, args.base_t_camera, args.ee_t_target),
        (args.output,),
        "zero-offset",
    )
    _validate_quality_threshold_args(args, "zero-offset")
    if not np.isfinite(args.max_offset_rad) or args.max_offset_rad <= 0.0:
        raise ValueError("--max-offset-rad must be a positive finite value")
    _refuse_existing_outputs((args.output,), args.force, "zero-offset")
    if not np.isfinite(args.max_offset_std_rad) or args.max_offset_std_rad <= 0.0:
        raise ValueError("--max-offset-std-rad must be a positive finite value")
    try:
        import roboticstoolbox as rtb
    except ImportError as exc:
        raise RuntimeError(
            "roboticstoolbox is unavailable in the active Conda environment"
        ) from exc
    dataset = KinematicDataset.load(args.dataset)
    dataset_coverage, dataset_quality_warnings = _solver_dataset_quality(
        dataset, args, "zero-offset"
    )
    dataset_schema_version = read_dataset_schema_version(args.dataset)
    if dataset.observation_frame != ObservationFrame.CAMERA_T_TARGET:
        raise ValueError("zero-offset currently requires camera_T_target observations")
    robot = rtb.ERobot.URDF(str(args.urdf.resolve()))
    joint_names_by_child = _urdf_joint_names_by_child(args.urdf)
    selected_links = []
    link = robot.link_dict.get(args.ee_name)
    if link is None:
        raise ValueError(f"end-effector link {args.ee_name!r} does not exist")
    while link is not None:
        if link.isjoint:
            selected_links.append(link)
        link = link.parent
    selected_links.reverse()
    if len(selected_links) != dataset.joint_positions.shape[1]:
        raise ValueError(
            f"selected chain has {len(selected_links)} joints but dataset has "
            f"{dataset.joint_positions.shape[1]} columns"
        )
    joint_identity = dataset.check_joint_identity(
        [joint_names_by_child.get(link.name, link.name) for link in selected_links]
    )
    if not joint_identity["verified"]:
        LOGGER.warning("%s", joint_identity["reason"])
    dataset.validate_solver_sample_count()

    camera_t_base = None
    ee_t_target = None
    external_transform_hashes = None
    if args.model_mode == "robot-body":
        if args.base_t_camera is None or args.ee_t_target is None:
            raise ValueError(
                "robot-body zero-offset mode requires --base-t-camera and --ee-t-target"
            )
        camera_t_base = np.linalg.inv(
            _load_transform(args.base_t_camera, "base_T_camera")
        )
        ee_t_target = _load_transform(args.ee_t_target, "ee_T_target")
        external_transform_hashes = {
            "base_T_camera_sha256": _sha256(args.base_t_camera),
            "ee_T_target_sha256": _sha256(args.ee_t_target),
        }
    elif args.base_t_camera is not None or args.ee_t_target is not None:
        raise ValueError(
            "self-calibrating zero-offset mode does not accept external transforms"
        )

    result = estimate_joint_zero_offsets(
        robot=robot,
        ee_name=args.ee_name,
        joint_positions=dataset.joint_positions,
        observations=dataset.observed_poses,
        camera_t_base=camera_t_base,
        ee_t_target=ee_t_target,
        max_offset_rad=args.max_offset_rad,
    )
    accepted, quality_failures = _evaluate_solution_quality(
        converged=result["success"],
        full_rank=result["full_rank"],
        position_error_max_m=result["position_error_max_m"],
        orientation_error_max_rad=result["orientation_error_max_rad"],
        condition_number=result["condition_number"],
        max_position_error=args.max_position_error,
        max_orientation_error=args.max_orientation_error,
        max_condition_number=args.max_condition_number,
    )
    accepted, quality_failures = _merge_dataset_quality_gate(
        accepted, quality_failures, dataset_quality_warnings
    )
    estimated_indices = set(result["estimated_indices"].tolist())
    bound_tolerance_rad = max(1e-9, args.max_offset_rad * 1e-6)
    offsets_at_bound = [
        int(index)
        for index in result["estimated_indices"]
        if abs(abs(float(result["offsets"][index])) - args.max_offset_rad)
        <= bound_tolerance_rad
    ]
    if offsets_at_bound:
        accepted = False
        quality_failures.append(
            "joint zero offsets reached --max-offset-rad for zero-based joints: "
            + ", ".join(str(index) for index in offsets_at_bound)
        )
    uncertain_indices = [
        int(index)
        for index in result["estimated_indices"]
        if not np.isfinite(result["offset_std_rad"][index])
        or result["offset_std_rad"][index] > args.max_offset_std_rad
    ]
    if uncertain_indices:
        accepted = False
        quality_failures.append(
            "joint zero-offset uncertainty exceeds --max-offset-std-rad for "
            "zero-based joints: " + ", ".join(str(index) for index in uncertain_indices)
        )
    joints = []
    for index, (link, offset) in enumerate(
        zip(selected_links, result["offsets"], strict=True)
    ):
        estimated = index in estimated_indices
        offset_std = result["offset_std_rad"][index]
        finite_uncertainty = estimated and np.isfinite(offset_std) and offset_std >= 0.0
        z_score = (
            abs(float(offset)) / float(offset_std)
            if finite_uncertainty and offset_std > 0.0
            else None
        )
        joints.append(
            {
                "zero_based_index": index,
                "joint_name": joint_names_by_child.get(link.name, link.name),
                "zero_offset_rad": float(offset) if estimated else None,
                "zero_offset_deg": float(np.rad2deg(offset)) if estimated else None,
                "zero_offset_std_rad": (
                    float(offset_std) if finite_uncertainty else None
                ),
                "zero_offset_std_deg": (
                    float(np.rad2deg(offset_std)) if finite_uncertainty else None
                ),
                "zero_offset_abs_z_score": z_score,
                "statistically_distinguishable_from_zero_3sigma": (
                    bool(z_score >= 3.0) if z_score is not None else None
                ),
                "estimated": estimated,
                "note": (
                    "estimated with URDF geometry fixed"
                    if estimated
                    else "fixed to zero gauge because it is coupled to an unknown external transform"
                ),
            }
        )

    position_worst_index = int(np.argmax(result["position_errors_m"]))
    orientation_worst_index = int(np.argmax(result["orientation_errors_rad"]))
    sample_errors = [
        {
            "sample_index": int(index),
            "source_sample_index": int(dataset.sample_indices[index]),
            "joint_positions_rad": dataset.joint_positions[index].tolist(),
            "position_error_m": float(position_error),
            "orientation_error_rad": float(orientation_error),
            "position_exceeds_threshold": bool(
                position_error > args.max_position_error
            ),
            "orientation_exceeds_threshold": bool(
                orientation_error > args.max_orientation_error
            ),
        }
        for index, (position_error, orientation_error) in enumerate(
            zip(
                result["position_errors_m"],
                result["orientation_errors_rad"],
                strict=True,
            )
        )
    ]

    report = {
        "schema_version": 1,
        "calibration_type": "joint-zero-offset-only",
        "provenance": _runtime_provenance(),
        "model_mode": args.model_mode,
        "offset_convention": "q_effective = q_measured + zero_offset_rad",
        "fixed_parameters": ["joint xyz", "joint rpy", "joint axis", "link geometry"],
        "samples": len(dataset.joint_positions),
        "dof": dataset.joint_positions.shape[1],
        "ee_name": args.ee_name,
        "success": result["success"],
        "message": result["message"],
        "evaluations": result["evaluations"],
        "cost": result["cost"],
        "position_error_mean_m": result["position_error_mean_m"],
        "position_error_max_m": result["position_error_max_m"],
        "position_error_rms_m": result["position_error_rms_m"],
        "position_error_p95_m": result["position_error_p95_m"],
        "position_error_worst_sample_index": position_worst_index,
        "position_error_worst_source_sample_index": int(
            dataset.sample_indices[position_worst_index]
        ),
        "orientation_error_mean_rad": result["orientation_error_mean_rad"],
        "orientation_error_max_rad": result["orientation_error_max_rad"],
        "orientation_error_rms_rad": result["orientation_error_rms_rad"],
        "orientation_error_p95_rad": result["orientation_error_p95_rad"],
        "orientation_error_worst_sample_index": orientation_worst_index,
        "orientation_error_worst_source_sample_index": int(
            dataset.sample_indices[orientation_worst_index]
        ),
        "jacobian_rank": result["jacobian_rank"],
        "parameter_count": result["parameter_count"],
        "full_rank": result["full_rank"],
        "condition_number": result["condition_number"],
        "accepted": accepted,
        "quality_thresholds": {
            "max_position_error_m": args.max_position_error,
            "max_orientation_error_rad": args.max_orientation_error,
            "max_condition_number": args.max_condition_number,
            "min_joint_span_rad": args.min_joint_span,
            "min_translation_span_norm_m": args.min_translation_span,
            "min_rotation_from_first_rad": args.min_rotation_span,
        },
        "quality_failures": quality_failures,
        "dataset_quality_warnings": dataset_quality_warnings,
        "dataset_coverage": dataset_coverage,
        "sample_errors": sample_errors,
        "offset_bound": {
            "max_offset_rad": args.max_offset_rad,
            "tolerance_rad": bound_tolerance_rad,
            "joints_at_bound": offsets_at_bound,
        },
        "offset_uncertainty": {
            "method": "linearized covariance from residual variance and Jacobian",
            "max_offset_std_rad": args.max_offset_std_rad,
            "joints_exceeding_threshold": uncertain_indices,
            "residual_degrees_of_freedom": result["residual_degrees_of_freedom"],
            "residual_variance": (
                result["residual_variance"]
                if np.isfinite(result["residual_variance"])
                else None
            ),
            "note": (
                "approximate 1-sigma values; SE(3) translation and rotation "
                "residuals are equally weighted"
            ),
        },
        "joints": joints,
        "estimated_joint_indices": sorted(estimated_indices),
        "gauge_fixed_joint_indices": [
            index
            for index in range(len(selected_links))
            if index not in estimated_indices
        ],
        "identifiability_note": (
            "In self-calibrating mode, the first and last encoder offsets are "
            "exactly coupled to Camera_T_Base and EE_T_Target and are fixed to "
            "zero gauge. Robot-body mode estimates every offset."
        ),
        "estimated_camera_T_base": _jsonable(result["camera_t_base"]),
        "estimated_EE_T_target": _jsonable(result["ee_t_target"]),
        "dataset_sha256": _sha256(args.dataset),
        "joint_identity_verified": joint_identity["verified"],
        "joint_identity": joint_identity,
        "dataset_schema_version": dataset_schema_version,
        "urdf_sha256": _sha256(args.urdf),
    }
    if external_transform_hashes is not None:
        report.update(external_transform_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, _jsonable(report))
    LOGGER.info("Wrote joint zero-offset calibration: %s", args.output)
    LOGGER.info("Calibrated joint zero offsets=\n%s", json.dumps(joints, indent=2))
    LOGGER.info(
        "Zero-offset residuals: position_m(mean/p95/max)=%.8g/%.8g/%.8g, "
        "orientation_rad(mean/p95/max)=%.8g/%.8g/%.8g, "
        "jacobian_rank=%d/%d, condition_number=%.6g",
        report["position_error_mean_m"],
        report["position_error_p95_m"],
        report["position_error_max_m"],
        report["orientation_error_mean_rad"],
        report["orientation_error_p95_rad"],
        report["orientation_error_max_rad"],
        report["jacobian_rank"],
        report["parameter_count"],
        report["condition_number"],
    )
    LOGGER.info(
        "Worst zero-offset source samples: position=%d, orientation=%d",
        report["position_error_worst_source_sample_index"],
        report["orientation_error_worst_source_sample_index"],
    )
    for failure in quality_failures:
        LOGGER.error("Zero-offset calibration rejected: %s", failure)
    return 0 if accepted else 1


def _calibrate(args: argparse.Namespace) -> int:
    """Run selected post-collection stages and verify their artifacts."""
    backend = getattr(args, "backend", "rtb")
    base_link = getattr(args, "base_link", None)
    if backend == "numpy":
        if args.calibration_type != "geometry":
            raise ValueError(
                "--backend numpy requires --calibration-type geometry; "
                "zero-offset calibration requires the RTB adapter"
            )
        if any(
            (
                args.geometry_anchor_joint,
                args.experimental_nominal_allocation,
                args.experimental_svd_gauge_allocation,
            )
        ):
            raise ValueError("URDF geometry allocation options require --backend rtb")
    elif base_link is not None:
        raise ValueError("--base-link requires --backend numpy")
    concise_output = args.output_detail == "concise" or args.quiet
    if args.urdf is None:
        from importlib.resources import files

        args.urdf = Path(
            str(files("kinematic_calibration") / "assets" / "robots" / "ur10.urdf")
        )
    if args.output_dir is None:
        args.output_dir = args.dataset.parent / "calibration"
    if args.skip_handeye and args.handeye_samples is not None:
        raise ValueError("--skip-handeye cannot be combined with --handeye-samples")
    if args.handeye_samples is None and not args.skip_handeye:
        automatic_handeye_samples = args.dataset.parent / "handeye_observations.npz"
        if automatic_handeye_samples.is_file():
            args.handeye_samples = automatic_handeye_samples
            if not concise_output:
                LOGGER.info(
                    "Auto-discovered hand-eye observations: %s",
                    automatic_handeye_samples,
                )
    if args.dataset.suffix.lower() != ".npz":
        raise ValueError("calibrate --dataset must use the .npz extension")
    if (
        args.handeye_samples is not None
        and args.handeye_samples.suffix.lower() != ".npz"
    ):
        raise ValueError("calibrate --handeye-samples must use the .npz extension")
    if (
        args.handeye_result is not None
        and args.handeye_result.suffix.lower() != ".json"
    ):
        raise ValueError("calibrate --handeye-result must use the .json extension")
    calibration_dataset = KinematicDataset.load(args.dataset)
    source_is_robot_body = (
        calibration_dataset.observation_frame == ObservationFrame.BASE_T_EE
    )
    _validate_quality_threshold_args(args, "calibrate")
    if args.max_iterations <= 0:
        raise ValueError("calibrate --max-iterations must be positive")
    if not np.isfinite(args.damping) or args.damping < 0.0:
        raise ValueError("calibrate --damping must be finite and non-negative")
    if not np.isfinite(args.max_offset_rad) or args.max_offset_rad <= 0.0:
        raise ValueError("calibrate --max-offset-rad must be positive and finite")
    if not np.isfinite(args.max_offset_std_rad) or args.max_offset_std_rad <= 0.0:
        raise ValueError("calibrate --max-offset-std-rad must be positive and finite")

    if not args.urdf.is_file():
        raise ValueError(f"source URDF does not exist: {args.urdf}")
    manifest_path = args.output_dir / "calibration_manifest.json"
    final_result_path = args.output_dir / "final_result.json"
    handeye_result = args.handeye_result or args.output_dir / "handeye_result.json"
    base_t_camera_path = args.output_dir / "base_T_camera.npy"
    ee_t_target_path = args.output_dir / "ee_T_target.npy"
    geometry_parameters = args.output_dir / "geometry_parameters.npz"
    geometry_result = geometry_parameters.with_suffix(".json")
    zero_offset_result = args.output_dir / "zero_offsets.json"

    possible_extrinsics = bool(
        args.handeye_samples is not None
        and not args.no_use_handeye_extrinsics
        and not source_is_robot_body
    )
    output_paths = [manifest_path, final_result_path]
    if args.calibration_type in {"geometry", "both"}:
        output_paths.extend((geometry_parameters, geometry_result))
        if backend == "rtb":
            if possible_extrinsics:
                export_suffixes = ("calibrated", "self_calibrated")
            elif source_is_robot_body:
                export_suffixes = ("calibrated",)
            else:
                export_suffixes = ("self_calibrated",)
            output_paths.extend(
                args.output_dir / f"{args.urdf.stem}_{suffix}.urdf"
                for suffix in export_suffixes
            )
    if args.calibration_type in {"zero-offset", "both"}:
        output_paths.append(zero_offset_result)
    if possible_extrinsics:
        output_paths.extend((base_t_camera_path, ee_t_target_path))
    input_paths = (args.dataset, args.urdf, args.handeye_samples)
    _refuse_input_output_aliases(input_paths, output_paths, "calibrate")
    if args.handeye_samples is not None:
        # This path can be reused or recomputed, but other stages must preserve it.
        _refuse_input_output_aliases((handeye_result,), output_paths, "calibrate")
        if args.force or not handeye_result.exists():
            _refuse_input_output_aliases(
                input_paths, (handeye_result,), "calibrate"
            )
            output_paths.append(handeye_result)
    for index, output_path in enumerate(output_paths):
        _refuse_input_output_aliases(
            (output_path,), output_paths[index + 1 :], "calibrate"
        )
    _refuse_existing_outputs(
        (manifest_path, final_result_path), args.force, "calibrate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def invoke(function, namespace):
        """Run one noisy legacy stage under the selected presentation policy."""
        if not concise_output:
            return function(namespace)
        previous_disable = logging.root.manager.disable
        try:
            logging.disable(logging.INFO)
            with contextlib.redirect_stdout(io.StringIO()):
                return function(namespace)
        finally:
            logging.disable(previous_disable)

    handeye = {
        "attempted": args.handeye_samples is not None,
        "result": str(handeye_result) if args.handeye_samples is not None else None,
        "accepted": False,
        "extrinsics_used": False,
        "fallback_reason": None,
    }
    verified_handeye_report = None
    if args.handeye_samples is not None:
        try:
            from kinematic_calibration.handeye_cli import (
                _load_verified_result as _load_verified_handeye_result,
            )
            from kinematic_calibration.handeye_cli import (
                _solve as _solve_opencv_handeye,
            )

            if handeye_result.exists() and not args.force:
                if not concise_output:
                    LOGGER.info("Reusing hand-eye result: %s", handeye_result)
            else:
                handeye_code = invoke(
                    _solve_opencv_handeye,
                    argparse.Namespace(
                        samples=args.handeye_samples,
                        output=handeye_result,
                        method=args.opencv_method,
                        max_position_error=args.max_position_error,
                        max_orientation_error=args.max_orientation_error,
                        force=args.force,
                    ),
                )
                if handeye_code != 0:
                    LOGGER.warning(
                        "Hand-eye result was rejected; continuing without external "
                        "transforms"
                    )
            verified_handeye_report, _, handeye_accepted = (
                _load_verified_handeye_result(args.handeye_samples, handeye_result)
            )
            handeye["accepted"] = bool(handeye_accepted)
            if not handeye_accepted:
                handeye["fallback_reason"] = "hand-eye quality gate rejected the result"
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            handeye["fallback_reason"] = f"hand-eye solve or verification failed: {exc}"
            LOGGER.warning("%s; continuing without external transforms", exc)
    else:
        handeye["fallback_reason"] = "no --handeye-samples supplied"

    use_extrinsics = bool(
        handeye["accepted"]
        and verified_handeye_report is not None
        and not args.no_use_handeye_extrinsics
        and not source_is_robot_body
    )
    if use_extrinsics:
        base_t_camera = np.asarray(
            verified_handeye_report.get("Base_T_Camera"), dtype=np.float64
        )
        ee_t_target = np.asarray(
            verified_handeye_report.get("EE_T_Board"), dtype=np.float64
        )
        # Validate directions and rigid-transform contents before propagating them.
        for name, value in (
            ("Base_T_Camera", base_t_camera),
            ("EE_T_Target", ee_t_target),
        ):
            if value.shape != (4, 4):
                raise ValueError(f"accepted hand-eye result has invalid {name}")
        if (
            any(path.exists() for path in (base_t_camera_path, ee_t_target_path))
            and not args.force
        ):
            raise ValueError(
                "calibrate external-transform output exists; use --force to replace it"
            )
        atomic_save_npy(base_t_camera_path, base_t_camera)
        atomic_save_npy(ee_t_target_path, ee_t_target)
        handeye["extrinsics_used"] = True
        handeye["fallback_reason"] = None
        model_mode = "robot-body"
        if not concise_output:
            LOGGER.info("Using accepted hand-eye transforms for downstream calibration")
    else:
        model_mode = "robot-body" if source_is_robot_body else "self-calibrating"
        if handeye["accepted"] and source_is_robot_body:
            handeye["fallback_reason"] = (
                "dataset already contains base_T_ee observations"
            )
        elif handeye["accepted"] and args.no_use_handeye_extrinsics:
            handeye["fallback_reason"] = "disabled by --no-use-handeye-extrinsics"
        if not concise_output:
            LOGGER.info("Using %s mode without external transforms", model_mode)

    shared = {
        "dataset": args.dataset,
        "urdf": args.urdf,
        "ee_name": args.ee_name,
        "backend": backend,
        "base_link": base_link,
        "model_mode": model_mode,
        "base_t_camera": base_t_camera_path if use_extrinsics else None,
        "ee_t_target": ee_t_target_path if use_extrinsics else None,
        "max_position_error": args.max_position_error,
        "max_orientation_error": args.max_orientation_error,
        "max_condition_number": args.max_condition_number,
        "min_joint_span": args.min_joint_span,
        "min_translation_span": args.min_translation_span,
        "min_rotation_span": args.min_rotation_span,
        "fail_on_low_diversity": args.fail_on_low_diversity,
        "force": args.force,
    }
    stages = {}
    geometry_code = None
    exported_urdf = None
    export_code = None
    export_skipped_reason = None
    if args.calibration_type in {"geometry", "both"}:
        geometry_code = invoke(
            _solve,
            argparse.Namespace(
                **shared,
                output=geometry_parameters,
                max_iterations=args.max_iterations,
                damping=args.damping,
                quiet=args.quiet,
                geometry_anchor_joint=args.geometry_anchor_joint,
                experimental_nominal_allocation=(args.experimental_nominal_allocation),
                experimental_svd_gauge_allocation=(
                    args.experimental_svd_gauge_allocation
                ),
            ),
        )
        verification_code = invoke(
            _verify_result,
            argparse.Namespace(
                result=geometry_result,
                parameters=geometry_parameters,
                dataset=args.dataset,
                urdf=args.urdf,
                base_t_camera=base_t_camera_path if use_extrinsics else None,
                ee_t_target=ee_t_target_path if use_extrinsics else None,
            ),
        )
        export_suffix = (
            "calibrated" if model_mode == "robot-body" else "self_calibrated"
        )
        exported_urdf = args.output_dir / f"{args.urdf.stem}_{export_suffix}.urdf"
        if backend == "numpy":
            # A fitted space-POE model does not identify physical link frames.
            export_skipped_reason = (
                "NumPy POE fitting does not provide calibrated URDF link frames"
            )
        elif geometry_code == 0 and verification_code == 0:
            export_code = invoke(
                _export_urdf,
                argparse.Namespace(
                    result=geometry_result,
                    urdf=args.urdf,
                    output=exported_urdf,
                    force=args.force,
                    allow_partial=model_mode == "self-calibrating",
                ),
            )
        stages["geometry"] = {
            "accepted": geometry_code == 0 and verification_code == 0,
            "parameters": str(geometry_parameters),
            "result": str(geometry_result),
            "verification_exit_code": verification_code,
            "exported_urdf": str(exported_urdf) if export_code == 0 else None,
            "export_skipped_reason": export_skipped_reason,
        }

    zero_code = None
    if args.calibration_type in {"zero-offset", "both"}:
        geometry_ready = exported_urdf is not None and export_code == 0
        if args.calibration_type == "both" and not geometry_ready:
            stages["zero_offset"] = {
                "accepted": False,
                "result": None,
                "verification_exit_code": None,
                "source_urdf": None,
                "skipped_reason": (
                    "geometry calibration did not produce a verified exported URDF"
                ),
            }
        else:
            zero_source_urdf = (
                exported_urdf if args.calibration_type == "both" else args.urdf
            )
            zero_shared = {**shared, "urdf": zero_source_urdf}
            zero_code = invoke(
                _zero_offset,
                argparse.Namespace(
                    **zero_shared,
                    output=zero_offset_result,
                    max_offset_rad=args.max_offset_rad,
                    max_offset_std_rad=args.max_offset_std_rad,
                ),
            )
            zero_verification_code = invoke(
                _verify_result,
                argparse.Namespace(
                    result=zero_offset_result,
                    parameters=None,
                    dataset=args.dataset,
                    urdf=zero_source_urdf,
                    base_t_camera=base_t_camera_path if use_extrinsics else None,
                    ee_t_target=ee_t_target_path if use_extrinsics else None,
                ),
            )
            stages["zero_offset"] = {
                "accepted": zero_code == 0 and zero_verification_code == 0,
                "result": str(zero_offset_result),
                "verification_exit_code": zero_verification_code,
                "source_urdf": str(zero_source_urdf),
                "skipped_reason": None,
            }

    accepted = all(stage["accepted"] for stage in stages.values())
    manifest = {
        "schema_version": 1,
        "workflow": "post-collection-calibration",
        "accepted": accepted,
        "dataset": str(args.dataset),
        "urdf": str(args.urdf),
        "ee_name": args.ee_name,
        "calibration_type": args.calibration_type,
        "backend": backend,
        "base_link": base_link,
        "selected_model_mode": model_mode,
        "handeye": handeye,
        "stages": stages,
    }
    atomic_write_json(manifest_path, manifest)
    geometry_summary = (
        json.loads(geometry_result.read_text())
        if "geometry" in stages and geometry_result.is_file()
        else None
    )
    zero_summary = (
        json.loads(zero_offset_result.read_text())
        if "zero_offset" in stages and zero_offset_result.is_file()
        else None
    )
    detailed_final_result = {
        "accepted": accepted,
        "model_mode": model_mode,
        "handeye": {
            "accepted": handeye["accepted"],
            "extrinsics_used": handeye["extrinsics_used"],
            "fallback_reason": handeye["fallback_reason"],
        },
        "geometry": (
            {
                "accepted": stages["geometry"]["accepted"],
                "position_error_mean_m": geometry_summary.get("position_error_mean_m"),
                "position_error_max_m": geometry_summary.get("position_error_max_m"),
                "orientation_error_mean_rad": geometry_summary.get(
                    "orientation_error_mean_rad"
                ),
                "orientation_error_max_rad": geometry_summary.get(
                    "orientation_error_max_rad"
                ),
                "exported_urdf": stages["geometry"]["exported_urdf"],
                "parameters": stages["geometry"]["parameters"],
                "export_skipped_reason": stages["geometry"]["export_skipped_reason"],
            }
            if geometry_summary is not None
            else None
        ),
        "zero_offsets": (
            {
                "accepted": stages["zero_offset"]["accepted"],
                "joints": zero_summary.get("joints", []),
            }
            if zero_summary is not None
            else None
        ),
        "artifacts": {
            "directory": str(args.output_dir),
            "manifest": str(manifest_path),
        },
    }
    if concise_output:
        handeye_result_summary = {
            "accepted": handeye["accepted"],
            "used_by_robot_calibration": handeye["extrinsics_used"],
            "Base_T_Camera": (
                verified_handeye_report.get("Base_T_Camera")
                if verified_handeye_report is not None
                else None
            ),
            "EE_T_Target": (
                verified_handeye_report.get("EE_T_Board")
                if verified_handeye_report is not None
                else None
            ),
        }
        if not handeye["extrinsics_used"]:
            handeye_result_summary["fallback_reason"] = handeye["fallback_reason"]
        final_result = {
            "accepted": accepted,
            "mode": model_mode,
            "handeye_calibration": handeye_result_summary,
        }
        if geometry_summary is not None:
            from .reporting import summarize_geometry

            final_result["kinematic_calibration"] = summarize_geometry(
                geometry_summary, args.urdf, stages["geometry"]
            )
        if zero_summary is not None:
            zero_joints = zero_summary.get("joints", [])
            final_result["zero_offset_calibration"] = {
                "accepted": stages["zero_offset"]["accepted"],
                "offset_convention": zero_summary.get("offset_convention"),
                "joint_names": [joint.get("joint_name") for joint in zero_joints],
                "zero_offset_rad": [
                    joint.get("zero_offset_rad") for joint in zero_joints
                ],
                "zero_offset_deg": [
                    joint.get("zero_offset_deg") for joint in zero_joints
                ],
                "estimated": [joint.get("estimated") for joint in zero_joints],
            }
    else:
        final_result = detailed_final_result
    rendered_final_result = _json_dumps_fixed(final_result)
    atomic_write_text(final_result_path, rendered_final_result)
    if not concise_output:
        LOGGER.info("Wrote unified calibration manifest: %s", manifest_path)
    print(rendered_final_result, end="")
    return 0 if accepted else 1


def _export_urdf(args: argparse.Namespace) -> int:
    _refuse_input_output_aliases(
        (args.urdf, args.result), (args.output,), "export-urdf"
    )
    from .urdf_export import export_calibrated_urdf

    report = export_calibrated_urdf(
        result_path=args.result,
        source_urdf=args.urdf,
        output_urdf=args.output,
        force=args.force,
        allow_partial=args.allow_partial,
    )
    LOGGER.info("Exported calibrated URDF: %s", report["output_urdf"])
    LOGGER.info("URDF export report=\n%s", json.dumps(report, indent=2))
    return 0


def _verify_result(args: argparse.Namespace) -> int:
    if args.result.suffix.lower() != ".json":
        raise ValueError("verify-result --result must use the .json extension")
    report = json.loads(args.result.read_text())
    if not isinstance(report, dict):
        raise ValueError("calibration result JSON must contain an object")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported calibration result schema version")

    checks = {
        "dataset_sha256": (args.dataset, report.get("dataset_sha256")),
        "urdf_sha256": (args.urdf, report.get("urdf_sha256")),
    }
    external_arguments = {
        "base_T_camera_sha256": args.base_t_camera,
        "ee_T_target_sha256": args.ee_t_target,
    }
    for field, path in external_arguments.items():
        expected_hash = report.get(field)
        if expected_hash is not None:
            if path is None:
                option = field.removesuffix("_sha256").replace("_T_", "-t-")
                raise ValueError(
                    f"result references {field}; provide --{option.lower()}"
                )
            checks[field] = (path, expected_hash)
        elif path is not None:
            raise ValueError(f"result does not reference {field}")
    expected_parameters_hash = report.get("parameters_npz_sha256")
    if expected_parameters_hash is not None:
        if args.parameters is None:
            raise ValueError("result references a parameter NPZ; provide --parameters")
        checks["parameters_npz_sha256"] = (
            args.parameters,
            expected_parameters_hash,
        )
    elif args.parameters is not None:
        raise ValueError("result does not reference a parameter NPZ")

    verified = {}
    for field, (path, expected_hash) in checks.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"result has no valid {field}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{field} mismatch for {path}")
        verified[field] = actual_hash

    if args.parameters is not None:
        from .validation import validate_saved_poe_parameters

        with open_npz(args.parameters) as archive:
            required = {
                "schema_version",
                "parameter_type",
                "model_mode",
                "source_dataset_sha256",
                "source_urdf_sha256",
                "implementation_sha256",
                "screw_axes",
                "zero_configuration_pose",
                "observation_frame",
                "source_observation_frame",
                "source_sample_sha256",
                "source_sample_indices",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"parameter archive is missing metadata: {sorted(missing)}"
                )
            if (
                archive["schema_version"].shape != ()
                or archive["schema_version"].item() != 1
            ):
                raise ValueError("unsupported parameter archive schema version")
            if str(archive["parameter_type"].item()) != "kinematic-poe":
                raise ValueError("unsupported parameter archive type")
            parameter_metadata = {
                "model_mode": str(archive["model_mode"].item()),
                "dataset_sha256": str(archive["source_dataset_sha256"].item()),
                "urdf_sha256": str(archive["source_urdf_sha256"].item()),
                "implementation_sha256": str(archive["implementation_sha256"].item()),
                "observation_frame": str(archive["observation_frame"].item()),
                "source_observation_frame": str(
                    archive["source_observation_frame"].item()
                ),
            }
            # Legacy schema-1 parameter archives did not bind the terminal link.
            for field in ("backend", "base_link", "ee_name"):
                if field in archive.files:
                    if (
                        archive[field].shape != ()
                        or not isinstance(archive[field].item(), str)
                        or not archive[field].item().strip()
                    ):
                        raise ValueError(
                            f"parameter archive {field} must be a nonempty string scalar"
                        )
                    parameter_metadata[field] = archive[field].item()
                    if parameter_metadata[field] != report.get(field):
                        raise ValueError(
                            f"parameter archive {field} does not match result"
                        )
                elif field in report and field != "ee_name":
                    raise ValueError(f"parameter archive is missing result {field}")
            if "backend" in parameter_metadata and parameter_metadata[
                "backend"
            ] not in {"rtb", "numpy"}:
                raise ValueError("parameter archive has an unsupported backend")
            if "joint_names" in archive.files:
                names = archive["joint_names"]
                if (
                    names.ndim != 1
                    or names.dtype.kind != "U"
                    or names.tolist() != report.get("joint_names")
                ):
                    raise ValueError("parameter joint identities do not match result")
            parameter_identity_verified = None
            if "joint_identity_verified" in archive.files:
                value = archive["joint_identity_verified"]
                if value.shape != () or value.dtype.kind != "b":
                    raise ValueError(
                        "parameter joint_identity_verified must be a boolean scalar"
                    )
                parameter_identity_verified = value.item()
                if (
                    report.get("joint_identity_verified")
                    is not parameter_identity_verified
                ):
                    raise ValueError(
                        "parameter joint identity verification does not match result"
                    )
            for archive_field, report_field in (
                ("source_base_T_camera_sha256", "base_T_camera_sha256"),
                ("source_EE_T_target_sha256", "ee_T_target_sha256"),
            ):
                if archive_field in archive.files:
                    parameter_metadata[report_field] = str(
                        archive[archive_field].item()
                    )
            screw_axes, _ = validate_saved_poe_parameters(
                archive["screw_axes"], archive["zero_configuration_pose"]
            )
            if "joint_names" in archive.files and len(archive["joint_names"]) != len(
                screw_axes
            ):
                raise ValueError("parameter joint identities do not match model DOF")
            observation_frame = str(archive["observation_frame"].item())
            if observation_frame not in {item.value for item in ObservationFrame}:
                raise ValueError("parameter archive has an invalid observation frame")
            source_observation_frame = str(archive["source_observation_frame"].item())
            parameter_sample_sha256 = np.asarray(
                archive["source_sample_sha256"]
            ).astype(str)
            parameter_sample_indices = np.asarray(
                archive["source_sample_indices"], dtype=np.int64
            )
        for field in (
            "model_mode",
            "dataset_sha256",
            "urdf_sha256",
            "observation_frame",
            "source_observation_frame",
        ):
            if report.get(field) != parameter_metadata[field]:
                raise ValueError(f"parameter archive {field} does not match result")
        for field in ("base_T_camera_sha256", "ee_T_target_sha256"):
            if (
                field in parameter_metadata
                and report.get(field) != parameter_metadata[field]
            ):
                raise ValueError(f"parameter archive {field} does not match result")
        if report.get("dof") != len(screw_axes):
            raise ValueError("parameter screw-axis count does not match result DOF")
        source_dataset = KinematicDataset.load(args.dataset)
        identity = source_dataset.check_joint_identity(report.get("joint_names"))
        if (
            parameter_identity_verified is not None
            and identity["verified"] is not parameter_identity_verified
        ):
            raise ValueError(
                "parameter joint identity verification does not match source dataset"
            )
        if "joint_identity" in report and report["joint_identity"] != identity:
            raise ValueError(
                "result joint identity details do not match source dataset"
            )
        if source_observation_frame != source_dataset.observation_frame.value:
            raise ValueError(
                "parameter source observation frame does not match dataset"
            )
        expected_sample_sha256 = _dataset_sample_sha256(source_dataset)
        if not np.array_equal(parameter_sample_sha256, expected_sample_sha256):
            raise ValueError(
                "parameter training sample fingerprints do not match dataset"
            )
        if not np.array_equal(parameter_sample_indices, source_dataset.sample_indices):
            raise ValueError(
                "parameter training source sample indices do not match dataset"
            )
        provenance = report.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("calibration result provenance must be an object")
        if (
            provenance.get("implementation_sha256")
            != parameter_metadata["implementation_sha256"]
        ):
            raise ValueError(
                "parameter implementation fingerprint does not match result"
            )

    accepted = report.get("accepted") is True
    rendered = {
        "integrity_verified": True,
        "accepted": accepted,
        "calibration_type": report.get("calibration_type"),
        "model_mode": report.get("model_mode"),
        "verified_sha256": verified,
        "quality_failures": report.get("quality_failures", []),
    }
    print(json.dumps(rendered, indent=2))
    if not accepted:
        LOGGER.error("Calibration artifacts match, but the result was not accepted")
    return 0 if accepted else 1


def _verify_collection(args: argparse.Namespace) -> int:
    if args.dataset.suffix.lower() != ".npz":
        raise ValueError("verify-collection --dataset must use the .npz extension")
    if args.report.suffix.lower() != ".json":
        raise ValueError("verify-collection --report must use the .json extension")
    report = json.loads(args.report.read_text())
    if not isinstance(report, dict):
        raise ValueError("collection report JSON must contain an object")
    if report.get("acquisition_type") == "offline-camera-board":
        from kinematic_calibration.acquisition import verify_capture_report

        print(json.dumps(verify_capture_report(args.dataset, args.report), indent=2))
        return 0
    if report.get("schema_version") != 1:
        raise ValueError("unsupported collection report schema version")
    if report.get("report_type") != "ptpr-kinematic-collection":
        raise ValueError("JSON is not a PTPR kinematic collection report")
    if report.get("dataset_sha256") != _sha256(args.dataset):
        raise ValueError("collection report dataset SHA-256 mismatch")

    integer_fields = (
        "candidate_count",
        "translation_candidate_count",
        "rotation_candidate_count",
        "successful_detection_count",
        "failed_detection_count",
    )
    counts = {}
    for field in integer_fields:
        value = report.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"collection report has invalid {field}")
        counts[field] = value
    if counts["candidate_count"] != (
        counts["translation_candidate_count"] + counts["rotation_candidate_count"]
    ):
        raise ValueError("collection candidate counts are inconsistent")
    if counts["candidate_count"] != (
        counts["successful_detection_count"] + counts["failed_detection_count"]
    ):
        raise ValueError("collection detection counts are inconsistent")

    successful = report.get("successful_candidate_indices")
    failed = report.get("failed_candidate_indices")
    if not isinstance(successful, list) or not isinstance(failed, list):
        raise ValueError("collection candidate index fields must be lists")
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        for index in successful + failed
    ):
        raise ValueError("collection candidate indices must be integers")
    if successful != sorted(set(successful)) or failed != sorted(set(failed)):
        raise ValueError("collection candidate indices must be sorted and unique")
    if (
        len(successful) != counts["successful_detection_count"]
        or len(failed) != counts["failed_detection_count"]
    ):
        raise ValueError("collection candidate index counts are inconsistent")
    expected_indices = list(range(counts["candidate_count"]))
    if sorted(successful + failed) != expected_indices:
        raise ValueError(
            "successful and failed candidate indices do not partition all candidates"
        )

    dataset = KinematicDataset.load(args.dataset)
    if dataset.sample_indices.tolist() != successful:
        raise ValueError("dataset sample_indices do not match successful candidates")
    expected_rate = len(successful) / counts["candidate_count"]
    rate = report.get("detection_success_rate")
    if not isinstance(rate, (int, float)) or not np.isclose(rate, expected_rate):
        raise ValueError("collection detection success rate is inconsistent")
    if report.get("dataset_coverage") != dataset.get_summary():
        raise ValueError("collection coverage summary does not match the dataset")

    rendered = {
        "integrity_verified": True,
        "dataset_sha256": report["dataset_sha256"],
        "candidate_count": counts["candidate_count"],
        "successful_detection_count": len(successful),
        "failed_detection_count": len(failed),
        "detection_success_rate": expected_rate,
    }
    print(json.dumps(rendered, indent=2))
    return 0


def _verify_split(args: argparse.Namespace) -> int:
    if args.manifest.suffix.lower() != ".json":
        raise ValueError("verify-split --manifest must use the .json extension")
    for name in ("source", "train", "validation"):
        if getattr(args, name).suffix.lower() != ".npz":
            raise ValueError(f"verify-split --{name} must use the .npz extension")
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("split manifest JSON must contain an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema version")
    if manifest.get("report_type") != "kinematic-dataset-split":
        raise ValueError("JSON is not a kinematic dataset split manifest")

    checks = {
        "source_dataset_sha256": args.source,
        "train_dataset_sha256": args.train,
        "validation_dataset_sha256": args.validation,
    }
    verified = {}
    for field, path in checks.items():
        expected = manifest.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"split manifest has no valid {field}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{field} mismatch for {path}")
        verified[field] = actual

    source = KinematicDataset.load(args.source)
    train = KinematicDataset.load(args.train)
    validation = KinematicDataset.load(args.validation)
    if (
        train.joint_names != source.joint_names
        or validation.joint_names != source.joint_names
    ):
        raise ValueError("split datasets did not preserve source joint_names")
    train_indices = manifest.get("train_indices")
    validation_indices = manifest.get("validation_indices")
    for name, indices in (
        ("train_indices", train_indices),
        ("validation_indices", validation_indices),
    ):
        if not isinstance(indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indices
        ):
            raise ValueError(f"split manifest has invalid {name}")
        if indices != sorted(set(indices)):
            raise ValueError(f"split manifest {name} must be sorted and unique")
    source_positions = list(range(len(source.joint_positions)))
    if sorted(train_indices + validation_indices) != source_positions:
        raise ValueError("train and validation indices do not partition the source")
    if manifest.get("samples") != len(source.joint_positions):
        raise ValueError("split source sample count is inconsistent")
    if manifest.get("train_samples") != len(train.joint_positions):
        raise ValueError("split train sample count is inconsistent")
    if manifest.get("validation_samples") != len(validation.joint_positions):
        raise ValueError("split validation sample count is inconsistent")
    selected_seed = manifest.get("seed")
    requested_seed = manifest.get("seed_requested")
    attempts_requested = manifest.get("search_attempts_requested")
    attempts_evaluated = manifest.get("search_attempts_evaluated")
    for field, value in (
        ("seed", selected_seed),
        ("seed_requested", requested_seed),
        ("search_attempts_requested", attempts_requested),
        ("search_attempts_evaluated", attempts_evaluated),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"split manifest has invalid {field}")
    if attempts_requested <= 0 or not 1 <= attempts_evaluated <= attempts_requested:
        raise ValueError("split search attempt counts are inconsistent")
    if not requested_seed <= selected_seed < requested_seed + attempts_evaluated:
        raise ValueError("selected split seed is outside the evaluated seed range")
    replayed_validation_indices = np.sort(
        np.random.default_rng(selected_seed).choice(
            len(source.joint_positions),
            size=len(validation.joint_positions),
            replace=False,
        )
    ).tolist()
    if replayed_validation_indices != validation_indices:
        raise ValueError("selected split seed does not reproduce validation indices")
    coverage_thresholds = manifest.get("coverage_thresholds")
    if not isinstance(coverage_thresholds, dict):
        raise ValueError("split manifest coverage_thresholds must be an object")
    threshold_values = (
        coverage_thresholds.get("min_joint_span_rad"),
        coverage_thresholds.get("min_translation_span_norm_m"),
        coverage_thresholds.get("min_rotation_from_first_rad"),
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value < 0.0
        for value in threshold_values
    ):
        raise ValueError("split manifest has invalid coverage thresholds")
    min_joint_span, min_translation_span, min_rotation_span = threshold_values

    replayed_candidates = []
    for attempt in range(attempts_evaluated):
        candidate_seed = requested_seed + attempt
        candidate_validation_indices = np.sort(
            np.random.default_rng(candidate_seed).choice(
                len(source.joint_positions),
                size=len(validation.joint_positions),
                replace=False,
            )
        )
        candidate_train_indices = np.setdiff1d(
            np.arange(len(source.joint_positions)),
            candidate_validation_indices,
            assume_unique=True,
        )

        def candidate_subset(indices: np.ndarray) -> KinematicDataset:
            return KinematicDataset(
                source.joint_positions[indices],
                source.observed_poses[indices],
                source.observation_frame,
                source.sample_indices[indices],
                source.joint_names,
            )

        candidate_train = candidate_subset(candidate_train_indices)
        candidate_validation = candidate_subset(candidate_validation_indices)
        candidate_train_warnings = candidate_train.get_quality_warnings(
            min_joint_span_rad=min_joint_span,
            min_translation_span_norm_m=min_translation_span,
            min_rotation_from_first_rad=min_rotation_span,
        )
        candidate_validation_warnings = candidate_validation.get_quality_warnings(
            min_joint_span_rad=min_joint_span,
            min_translation_span_norm_m=min_translation_span,
            min_rotation_from_first_rad=min_rotation_span,
        )
        replayed_candidates.append(
            {
                "seed": candidate_seed,
                "train_warnings": candidate_train_warnings,
                "validation_warnings": candidate_validation_warnings,
                "warning_count": len(candidate_train_warnings)
                + len(candidate_validation_warnings),
            }
        )
    best_replayed = min(replayed_candidates, key=lambda item: item["warning_count"])
    if best_replayed["seed"] != selected_seed:
        raise ValueError("selected split seed is not the best evaluated candidate")
    if (
        attempts_evaluated < attempts_requested
        and replayed_candidates[-1]["warning_count"] != 0
    ):
        raise ValueError("split search stopped early without a warning-free candidate")
    if manifest.get("train_quality_warnings") != best_replayed["train_warnings"]:
        raise ValueError("manifest training coverage warnings are inconsistent")
    if (
        manifest.get("validation_quality_warnings")
        != best_replayed["validation_warnings"]
    ):
        raise ValueError("manifest validation coverage warnings are inconsistent")

    def verify_subset(
        name: str, subset: KinematicDataset, positions: list[int]
    ) -> None:
        if subset.observation_frame != source.observation_frame:
            raise ValueError(f"{name} observation frame differs from source")
        if not np.array_equal(
            subset.joint_positions, source.joint_positions[positions]
        ):
            raise ValueError(f"{name} joint samples do not match source positions")
        if not np.array_equal(subset.observed_poses, source.observed_poses[positions]):
            raise ValueError(f"{name} pose samples do not match source positions")
        if not np.array_equal(subset.sample_indices, source.sample_indices[positions]):
            raise ValueError(f"{name} source sample indices do not match")

    verify_subset("train", train, train_indices)
    verify_subset("validation", validation, validation_indices)
    if manifest.get("train_source_sample_indices") != train.sample_indices.tolist():
        raise ValueError("manifest train source sample indices do not match")
    if (
        manifest.get("validation_source_sample_indices")
        != validation.sample_indices.tolist()
    ):
        raise ValueError("manifest validation source sample indices do not match")
    if (
        train.get_summary()["duplicate_sample_count"]
        or validation.get_summary()["duplicate_sample_count"]
    ):
        raise ValueError("verified split contains duplicate observations")

    print(
        json.dumps(
            {
                "integrity_verified": True,
                "source_samples": len(source.joint_positions),
                "train_samples": len(train.joint_positions),
                "validation_samples": len(validation.joint_positions),
                "verified_sha256": verified,
            },
            indent=2,
        )
    )
    return 0


def _verify_deduplication(args: argparse.Namespace) -> int:
    if args.manifest.suffix.lower() != ".json":
        raise ValueError("verify-deduplication --manifest must use the .json extension")
    if args.source.suffix.lower() != ".npz" or args.output.suffix.lower() != ".npz":
        raise ValueError("verify-deduplication source and output must use .npz")
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("deduplication manifest JSON must contain an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported deduplication manifest schema version")
    if manifest.get("report_type") != "kinematic-dataset-deduplication":
        raise ValueError("JSON is not a kinematic dataset deduplication manifest")
    verified = {}
    for field, path in (
        ("source_dataset_sha256", args.source),
        ("output_dataset_sha256", args.output),
    ):
        expected = manifest.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"deduplication manifest has no valid {field}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{field} mismatch for {path}")
        verified[field] = actual

    source = KinematicDataset.load(args.source)
    output = KinematicDataset.load(args.output)
    summary = source.get_summary()
    expected_removed = summary["duplicate_sample_positions"]
    expected_kept = [
        index
        for index in range(len(source.joint_positions))
        if index not in set(expected_removed)
    ]
    expected_fields = {
        "source_samples": len(source.joint_positions),
        "output_samples": len(output.joint_positions),
        "removed_sample_count": len(expected_removed),
        "kept_positions": expected_kept,
        "kept_source_sample_indices": source.sample_indices[expected_kept].tolist(),
        "removed_positions": expected_removed,
        "removed_source_sample_indices": summary["duplicate_source_sample_indices"],
        "duplicate_of_positions": summary["duplicate_of_positions"],
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(f"deduplication manifest has inconsistent {field}")
    if output.joint_names != source.joint_names:
        raise ValueError("deduplicated joint_names differ from source")
    if output.observation_frame != source.observation_frame:
        raise ValueError("deduplicated observation frame differs from source")
    if not np.array_equal(
        output.joint_positions, source.joint_positions[expected_kept]
    ):
        raise ValueError("deduplicated joint samples differ from kept source rows")
    if not np.array_equal(output.observed_poses, source.observed_poses[expected_kept]):
        raise ValueError("deduplicated poses differ from kept source rows")
    if not np.array_equal(output.sample_indices, source.sample_indices[expected_kept]):
        raise ValueError("deduplicated source sample indices were not preserved")
    output_summary = output.get_summary()
    if output_summary["duplicate_sample_count"]:
        raise ValueError("deduplicated output still contains duplicate observations")
    if manifest.get("output_coverage") != output_summary:
        raise ValueError("deduplication output coverage is inconsistent")

    print(
        json.dumps(
            {
                "integrity_verified": True,
                "source_samples": len(source.joint_positions),
                "output_samples": len(output.joint_positions),
                "removed_sample_count": len(expected_removed),
                "verified_sha256": verified,
            },
            indent=2,
        )
    )
    return 0


def _verify_validation(args: argparse.Namespace) -> int:
    expected_suffixes = {
        "report": ".json",
        "parameters": ".npz",
        "dataset": ".npz",
        "result": ".json",
    }
    for name, suffix in expected_suffixes.items():
        path = getattr(args, name)
        if path.suffix.lower() != suffix:
            raise ValueError(
                f"verify-validation --{name} must use the {suffix} extension"
            )
    report = json.loads(args.report.read_text())
    if not isinstance(report, dict):
        raise ValueError("validation report JSON must contain an object")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported validation report schema version")
    if report.get("report_type") != "kinematic-independent-validation":
        raise ValueError("report is not an independent kinematic validation")

    checks = {
        "parameters_npz_sha256": (args.parameters, report.get("parameters_npz_sha256")),
        "validation_dataset_sha256": (
            args.dataset,
            report.get("validation_dataset_sha256"),
        ),
        "calibration_result_sha256": (
            args.result,
            report.get("calibration_result_sha256"),
        ),
    }
    verified = {}
    for field, (path, expected_hash) in checks.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"validation report has no valid {field}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{field} mismatch for {path}")
        verified[field] = actual_hash

    solve_report = json.loads(args.result.read_text())
    if not isinstance(solve_report, dict) or solve_report.get("accepted") is not True:
        raise ValueError("referenced calibration result is not accepted")
    if solve_report.get("parameters_npz_sha256") != verified["parameters_npz_sha256"]:
        raise ValueError(
            "calibration result does not reference the supplied parameters"
        )
    if report.get("parameter_acceptance_verified") is not True:
        raise ValueError("validation did not verify parameter acceptance")
    if report.get("independent_validation") is not True:
        raise ValueError("validation dataset independence was not verified")
    current_implementation_sha256 = _implementation_sha256()
    if report.get("current_implementation_sha256") != current_implementation_sha256:
        raise ValueError("validation report was produced by a different implementation")
    if report.get("implementation_matches_parameters") is not True:
        raise ValueError("validation implementation did not match parameter provenance")

    accepted = report.get("accepted") is True
    rendered = {
        "integrity_verified": True,
        "accepted": accepted,
        "report_type": report["report_type"],
        "implementation_sha256": current_implementation_sha256,
        "verified_sha256": verified,
        "quality_failures": report.get("quality_failures", []),
    }
    print(json.dumps(rendered, indent=2))
    if not accepted:
        LOGGER.error("Validation artifacts match, but the report was not accepted")
    return 0 if accepted else 1


def _validate_result(args: argparse.Namespace) -> int:
    from .validation import evaluate_saved_poe

    if args.parameters.suffix.lower() != ".npz":
        raise ValueError("validate-result --parameters must use the .npz extension")
    calibration_result_path = args.result or args.parameters.with_suffix(".json")
    if args.output is not None:
        _refuse_input_output_aliases(
            (
                args.parameters,
                args.dataset,
                calibration_result_path,
                args.base_t_camera,
                args.ee_t_target,
            ),
            (args.output,),
            "validate-result",
        )
        if args.output.suffix.lower() != ".json":
            raise ValueError("validate-result --output must use the .json extension")
        _refuse_existing_outputs((args.output,), args.force, "validate-result")
    calibration_report = None
    if calibration_result_path.exists():
        if calibration_result_path.suffix.lower() != ".json":
            raise ValueError("validate-result --result must use the .json extension")
        calibration_report = json.loads(calibration_result_path.read_text())
        if not isinstance(calibration_report, dict):
            raise ValueError("calibration result JSON must contain an object")
        if calibration_report.get("schema_version") not in (None, 1):
            raise ValueError("unsupported calibration result schema version")
        if calibration_report.get("calibration_type") not in (
            None,
            "kinematic-geometry",
        ):
            raise ValueError("calibration result is not a kinematic geometry report")
        if calibration_report.get("accepted") is not True:
            raise ValueError("calibration result was not accepted")
        expected_parameters_hash = calibration_report.get("parameters_npz_sha256")
        if expected_parameters_hash != _sha256(args.parameters):
            raise ValueError("parameter NPZ SHA-256 does not match calibration result")
    elif not args.allow_unverified_parameters:
        raise ValueError(
            f"calibration result is required: {calibration_result_path}; provide "
            "--result or use --allow-unverified-parameters for diagnostics"
        )
    thresholds = (args.max_position_error, args.max_orientation_error)
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("validation thresholds must be positive finite values")
    if args.min_validation_samples <= 0:
        raise ValueError("--min-validation-samples must be positive")
    coverage_thresholds = (
        args.min_joint_span,
        args.min_translation_span,
        args.min_rotation_span,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in coverage_thresholds):
        raise ValueError(
            "validation coverage thresholds must be finite and non-negative"
        )
    with open_npz(args.parameters) as archive:
        required = {"screw_axes", "zero_configuration_pose", "model_mode"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"parameter archive is missing fields: {sorted(missing)}")
        if "schema_version" in archive.files:
            schema_value = archive["schema_version"]
            raw_schema_version = (
                schema_value.item() if schema_value.shape == () else None
            )
            if isinstance(raw_schema_version, bool) or raw_schema_version != 1:
                raise ValueError("unsupported parameter archive schema version")
        if (
            "parameter_type" in archive.files
            and str(archive["parameter_type"].item()) != "kinematic-poe"
        ):
            raise ValueError("unsupported parameter archive type")
        screw_axes = archive["screw_axes"]
        zero_pose = archive["zero_configuration_pose"]
        model_mode = str(archive["model_mode"].item())
        source_dataset_sha256 = (
            str(archive["source_dataset_sha256"].item())
            if "source_dataset_sha256" in archive.files
            else None
        )
        source_dataset_schema_version = (
            str(archive["source_dataset_schema_version"].item())
            if "source_dataset_schema_version" in archive.files
            else None
        )
        source_urdf_sha256 = (
            str(archive["source_urdf_sha256"].item())
            if "source_urdf_sha256" in archive.files
            else None
        )
        parameter_implementation_sha256 = (
            str(archive["implementation_sha256"].item())
            if "implementation_sha256" in archive.files
            else None
        )
        source_sample_sha256 = (
            np.asarray(archive["source_sample_sha256"]).astype(str)
            if "source_sample_sha256" in archive.files
            else None
        )
        source_sample_indices = (
            np.asarray(archive["source_sample_indices"], dtype=np.int64)
            if "source_sample_indices" in archive.files
            else None
        )
        model_context = {}
        for field in ("backend", "base_link", "ee_name", "joint_names"):
            if field in archive.files:
                value = archive[field]
                if field == "ee_name" and (
                    value.shape != ()
                    or value.dtype.kind != "U"
                    or not value.item().strip()
                ):
                    raise ValueError("parameter ee_name must be a nonempty string scalar")
                model_context[field] = (
                    value.tolist() if field == "joint_names" else value.item()
                )
        training_identity_verified = None
        if "joint_identity_verified" in archive.files:
            value = archive["joint_identity_verified"]
            if value.shape != () or value.dtype.kind != "b":
                raise ValueError(
                    "parameter joint_identity_verified must be a boolean scalar"
                )
            training_identity_verified = value.item()

    if source_sample_sha256 is not None:
        if source_sample_sha256.ndim != 1:
            raise ValueError("source_sample_sha256 must be a one-dimensional array")
        if any(len(value) != 64 for value in source_sample_sha256):
            raise ValueError("parameter archive contains an invalid sample SHA-256")
    if source_sample_indices is not None:
        if source_sample_sha256 is None or source_sample_indices.shape != (
            len(source_sample_sha256),
        ):
            raise ValueError(
                "source_sample_indices do not match source sample fingerprints"
            )
        if np.any(source_sample_indices < 0) or len(
            set(source_sample_indices.tolist())
        ) != len(source_sample_indices):
            raise ValueError("source_sample_indices must be unique and non-negative")

    if calibration_report is not None:
        for field, value in model_context.items():
            if calibration_report.get(field) != value:
                raise ValueError(
                    f"parameter NPZ {field} does not match calibration result"
                )
        if (
            training_identity_verified is not None
            and calibration_report.get("joint_identity_verified")
            is not training_identity_verified
        ):
            raise ValueError(
                "parameter joint identity verification does not match calibration result"
            )
        if calibration_report.get("model_mode") != model_mode:
            raise ValueError(
                "parameter NPZ model mode does not match calibration result"
            )
        if (
            source_dataset_sha256 is not None
            and calibration_report.get("dataset_sha256") != source_dataset_sha256
        ):
            raise ValueError(
                "parameter NPZ source dataset hash does not match calibration result"
            )
        if (
            source_urdf_sha256 is not None
            and calibration_report.get("urdf_sha256") != source_urdf_sha256
        ):
            raise ValueError("parameter NPZ source URDF hash does not match result")
        report_provenance = calibration_report.get("provenance") or {}
        if not isinstance(report_provenance, dict):
            raise ValueError("calibration result provenance must be an object")
        report_implementation_sha256 = report_provenance.get("implementation_sha256")
        if (
            parameter_implementation_sha256 is not None
            and report_implementation_sha256 != parameter_implementation_sha256
        ):
            raise ValueError(
                "parameter NPZ implementation hash does not match calibration result"
            )
    current_implementation_sha256 = _implementation_sha256()
    if (
        parameter_implementation_sha256 is not None
        and parameter_implementation_sha256 != current_implementation_sha256
        and not args.allow_implementation_mismatch
    ):
        raise ValueError(
            "parameters were produced by a different implementation; rerun with "
            "matching code or use --allow-implementation-mismatch for diagnostics"
        )

    dataset = KinematicDataset.load(args.dataset)
    validation_identity = dataset.check_joint_identity(model_context.get("joint_names"))
    joint_identity_verified = (
        validation_identity["verified"] and training_identity_verified is True
    )
    if not validation_identity["verified"]:
        LOGGER.warning("validation %s", validation_identity["reason"])
    if training_identity_verified is not True:
        LOGGER.warning(
            "training joint-column identity is unverified in saved parameters"
        )
    validation_sample_sha256 = _dataset_sample_sha256(dataset)
    validation_dataset_coverage = dataset.get_summary()
    raw_validation_coverage_failures = dataset.get_quality_warnings(
        min_joint_span_rad=args.min_joint_span,
        min_translation_span_norm_m=args.min_translation_span,
        min_rotation_from_first_rad=args.min_rotation_span,
    )
    validation_dataset_schema_version = read_dataset_schema_version(args.dataset)
    validation_dataset_sha256 = _sha256(args.dataset)
    training_hash_to_positions = {}
    if source_sample_sha256 is not None:
        for index, value in enumerate(source_sample_sha256.tolist()):
            training_hash_to_positions.setdefault(value, []).append(index)
    overlapping_samples = []
    for validation_position, value in enumerate(validation_sample_sha256.tolist()):
        training_positions = training_hash_to_positions.get(value)
        if training_positions is None:
            continue
        overlapping_samples.append(
            {
                "sha256": value,
                "training_positions": training_positions,
                "training_source_sample_indices": (
                    [int(source_sample_indices[index]) for index in training_positions]
                    if source_sample_indices is not None
                    else None
                ),
                "validation_position": validation_position,
                "validation_source_sample_index": int(
                    dataset.sample_indices[validation_position]
                ),
            }
        )
    if source_dataset_sha256 == validation_dataset_sha256:
        independent = False
    elif source_sample_sha256 is None:
        independent = None
    else:
        independent = not overlapping_samples
    if independent is not True and not args.allow_training_data:
        raise ValueError(
            "validation dataset independence could not be established; it is "
            "identical to, overlaps, or lacks sample provenance from the optimization "
            "dataset; use independent data or --allow-training-data for diagnostics"
        )
    if model_mode == "self-calibrating":
        if args.base_t_camera is not None or args.ee_t_target is not None:
            raise ValueError("self-calibrating validation does not accept extrinsics")
        if dataset.observation_frame != ObservationFrame.CAMERA_T_TARGET:
            raise ValueError(
                "self-calibrating parameters require camera_T_target observations"
            )
    elif model_mode == "robot-body":
        if dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET:
            if args.base_t_camera is None or args.ee_t_target is None:
                raise ValueError(
                    "robot-body validation requires both external transforms for "
                    "camera_T_target observations"
                )
            dataset = dataset.as_robot_body(
                _load_transform(args.base_t_camera, "base_T_camera"),
                _load_transform(args.ee_t_target, "ee_T_target"),
            )
        elif args.base_t_camera is not None or args.ee_t_target is not None:
            raise ValueError(
                "robot-body validation does not accept extrinsics for base_T_ee data"
            )
    else:
        raise ValueError(f"unsupported saved model mode: {model_mode!r}")

    metrics = evaluate_saved_poe(
        screw_axes,
        zero_pose,
        dataset.joint_positions,
        dataset.observed_poses,
        sample_indices=dataset.sample_indices,
    )
    for sample_error in metrics["sample_errors"]:
        sample_error["position_exceeds_threshold"] = (
            sample_error["position_error_m"] > args.max_position_error
        )
        sample_error["orientation_exceeds_threshold"] = (
            sample_error["orientation_error_rad"] > args.max_orientation_error
        )
    numerical_failures = []
    if metrics["position_error_max_m"] > args.max_position_error:
        numerical_failures.append("maximum validation position error exceeds threshold")
    if metrics["orientation_error_max_rad"] > args.max_orientation_error:
        numerical_failures.append(
            "maximum validation orientation error exceeds threshold"
        )
    validation_coverage_failures = []
    if independent is True:
        validation_coverage_failures = raw_validation_coverage_failures
        numerical_failures.extend(
            f"independent validation coverage: {failure}"
            for failure in validation_coverage_failures
        )
    unique_validation_samples = validation_dataset_coverage["unique_sample_count"]
    if independent is True and unique_validation_samples < args.min_validation_samples:
        numerical_failures.append(
            f"independent validation has {unique_validation_samples} unique samples; "
            f"at least {args.min_validation_samples} are required"
        )
    implementation_matches = (
        None
        if parameter_implementation_sha256 is None
        else parameter_implementation_sha256 == current_implementation_sha256
    )
    deployment_failures = []
    if calibration_report is None:
        deployment_failures.append("parameter acceptance could not be verified")
    if independent is not True:
        deployment_failures.append(
            "validation dataset is not verified as independent from training data"
        )
    if implementation_matches is not True:
        deployment_failures.append(
            "parameter implementation does not match the current implementation"
            if implementation_matches is False
            else "parameter implementation provenance is unavailable"
        )
    failures = numerical_failures + deployment_failures
    numerically_accepted = not numerical_failures
    accepted = numerically_accepted and not deployment_failures
    report = {
        "schema_version": 1,
        "report_type": (
            "kinematic-independent-validation"
            if independent is True
            else "kinematic-training-data-consistency"
            if independent is False
            else "kinematic-validation-provenance-unavailable"
        ),
        "provenance": _runtime_provenance(),
        "parameter_implementation_sha256": parameter_implementation_sha256,
        "current_implementation_sha256": current_implementation_sha256,
        "implementation_matches_parameters": implementation_matches,
        "accepted": accepted,
        "numerically_accepted": numerically_accepted,
        "parameter_acceptance_verified": calibration_report is not None,
        "joint_identity_verified": joint_identity_verified,
        "validation_joint_identity": validation_identity,
        "training_joint_identity_verified": training_identity_verified,
        "calibration_result": (
            str(calibration_result_path) if calibration_report is not None else None
        ),
        "calibration_result_sha256": (
            _sha256(calibration_result_path) if calibration_report is not None else None
        ),
        "independent_validation": independent,
        "overlapping_training_sample_count": len(overlapping_samples),
        "overlapping_training_samples": overlapping_samples,
        "source_sample_provenance_available": source_sample_sha256 is not None,
        "source_sample_indices_available": source_sample_indices is not None,
        "validation_purpose": (
            "independent-accuracy"
            if independent is True
            else "training-data-consistency"
            if independent is False
            else "provenance-unavailable"
        ),
        "model_mode": model_mode,
        **metrics,
        "quality_thresholds": {
            "max_position_error_m": args.max_position_error,
            "max_orientation_error_rad": args.max_orientation_error,
            "min_validation_samples": args.min_validation_samples,
            "min_joint_span_rad": args.min_joint_span,
            "min_translation_span_norm_m": args.min_translation_span,
            "min_rotation_from_first_rad": args.min_rotation_span,
        },
        "quality_failures": failures,
        "numerical_quality_failures": numerical_failures,
        "validation_coverage_failures": validation_coverage_failures,
        "deployment_failures": deployment_failures,
        "validation_dataset_sha256": validation_dataset_sha256,
        "validation_dataset_schema_version": validation_dataset_schema_version,
        "validation_dataset_coverage": validation_dataset_coverage,
        "source_dataset_sha256": source_dataset_sha256,
        "source_dataset_schema_version": source_dataset_schema_version,
        "source_urdf_sha256": source_urdf_sha256,
        "parameters_npz_sha256": _sha256(args.parameters),
    }
    rendered = json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        atomic_write_json(args.output, _jsonable(report))
        LOGGER.info("Wrote kinematic validation report: %s", args.output)
    LOGGER.info(
        "Validation residuals: position_m(mean/p95/max)=%.8g/%.8g/%.8g, "
        "orientation_rad(mean/p95/max)=%.8g/%.8g/%.8g",
        metrics["position_error_mean_m"],
        metrics["position_error_p95_m"],
        metrics["position_error_max_m"],
        metrics["orientation_error_mean_rad"],
        metrics["orientation_error_p95_rad"],
        metrics["orientation_error_max_rad"],
    )
    LOGGER.info(
        "Worst validation source samples: position=%d, orientation=%d",
        metrics["position_error_worst_source_sample_index"],
        metrics["orientation_error_worst_source_sample_index"],
    )
    for failure in failures:
        LOGGER.error("Validation rejected: %s", failure)
    return 0 if accepted else 1


def _compare_zero_offsets(args: argparse.Namespace) -> int:
    if len(args.results) < 2:
        raise ValueError("compare-zero-offsets requires at least two --result files")
    if args.output.suffix.lower() != ".json":
        raise ValueError("compare-zero-offsets --output must use the .json extension")
    _refuse_input_output_aliases(args.results, (args.output,), "compare-zero-offsets")
    _refuse_existing_outputs((args.output,), args.force, "compare-zero-offsets")
    if (
        not np.isfinite(args.max_between_run_std_rad)
        or args.max_between_run_std_rad <= 0.0
    ):
        raise ValueError("--max-between-run-std-rad must be a positive finite value")

    reports = []
    for path in args.results:
        report = json.loads(path.read_text())
        if not isinstance(report, dict):
            raise ValueError(f"zero-offset report must contain an object: {path}")
        if report.get("calibration_type") != "joint-zero-offset-only":
            raise ValueError(f"not a zero-offset report: {path}")
        if report.get("accepted") is not True:
            raise ValueError(f"zero-offset report was not accepted: {path}")
        if report.get("provenance") is not None and not isinstance(
            report["provenance"], dict
        ):
            raise ValueError(f"zero-offset report has invalid provenance: {path}")
        joints = report.get("joints")
        if not isinstance(joints, list) or not joints:
            raise ValueError(f"zero-offset report has no joint list: {path}")
        if any(
            not isinstance(joint, dict)
            or not isinstance(joint.get("zero_based_index"), int)
            or not isinstance(joint.get("joint_name"), str)
            or "zero_offset_rad" not in joint
            for joint in joints
        ):
            raise ValueError(f"zero-offset report has invalid joint entries: {path}")
        reports.append(report)

    reference = reports[0]
    compatibility_fields = (
        "model_mode",
        "ee_name",
        "dof",
        "urdf_sha256",
        "offset_convention",
        "estimated_joint_indices",
        "gauge_fixed_joint_indices",
    )
    for field in compatibility_fields:
        if any(report.get(field) != reference.get(field) for report in reports[1:]):
            raise ValueError(f"zero-offset reports disagree on {field}")
    dataset_hashes = [report.get("dataset_sha256") for report in reports]
    implementation_hashes = [
        (report.get("provenance") or {}).get("implementation_sha256")
        for report in reports
    ]
    known_implementation_hashes = [value for value in implementation_hashes if value]
    if len(set(known_implementation_hashes)) > 1:
        raise ValueError(
            "zero-offset reports were produced by different implementations"
        )
    if any(not isinstance(value, str) or len(value) != 64 for value in dataset_hashes):
        raise ValueError(
            "zero-offset reports must contain valid dataset SHA-256 values"
        )
    if (
        not isinstance(reference.get("urdf_sha256"), str)
        or len(reference["urdf_sha256"]) != 64
    ):
        raise ValueError("zero-offset reports must contain a valid URDF SHA-256")
    if (
        len(set(dataset_hashes)) != len(dataset_hashes)
        and not args.allow_duplicate_datasets
    ):
        raise ValueError(
            "zero-offset reports reuse a dataset; use independent acquisitions or "
            "--allow-duplicate-datasets for a pipeline consistency check"
        )

    joint_sets = [
        [
            (joint.get("zero_based_index"), joint.get("joint_name"))
            for joint in report["joints"]
        ]
        for report in reports
    ]
    if any(joints != joint_sets[0] for joints in joint_sets[1:]):
        raise ValueError("zero-offset reports disagree on joint order or names")

    joint_statistics = []
    failures = []
    for joint_position, (joint_index, joint_name) in enumerate(joint_sets[0]):
        values = [
            report["joints"][joint_position]["zero_offset_rad"] for report in reports
        ]
        if all(value is None for value in values):
            joint_statistics.append(
                {
                    "zero_based_index": joint_index,
                    "joint_name": joint_name,
                    "estimated": False,
                    "offsets_rad": values,
                }
            )
            continue
        if any(value is None or not np.isfinite(value) for value in values):
            raise ValueError(f"inconsistent offset values for joint {joint_name}")
        values_array = np.asarray(values, dtype=float)
        standard_deviation = float(np.std(values_array, ddof=1))
        if standard_deviation > args.max_between_run_std_rad:
            failures.append(
                f"{joint_name} between-run std {standard_deviation:.6g} rad exceeds "
                f"{args.max_between_run_std_rad:.6g} rad"
            )
        joint_statistics.append(
            {
                "zero_based_index": joint_index,
                "joint_name": joint_name,
                "estimated": True,
                "offsets_rad": values,
                "mean_offset_rad": float(values_array.mean()),
                "between_run_std_rad": standard_deviation,
                "min_offset_rad": float(values_array.min()),
                "max_offset_rad": float(values_array.max()),
                "range_rad": float(np.ptp(values_array)),
                "sign_consistent": bool(
                    np.all(values_array > 0.0) or np.all(values_array < 0.0)
                ),
            }
        )

    output = {
        "schema_version": 1,
        "report_type": "joint-zero-offset-repeatability",
        "provenance": _runtime_provenance(),
        "source_implementation_sha256": (
            known_implementation_hashes[0] if known_implementation_hashes else None
        ),
        "accepted": not failures,
        "runs": len(reports),
        "result_files": [str(path) for path in args.results],
        "dataset_sha256": dataset_hashes,
        "urdf_sha256": reference["urdf_sha256"],
        "model_mode": reference["model_mode"],
        "quality_thresholds": {"max_between_run_std_rad": args.max_between_run_std_rad},
        "quality_failures": failures,
        "joints": joint_statistics,
    }
    atomic_write_json(args.output, _jsonable(output))
    LOGGER.info("Wrote zero-offset repeatability report: %s", args.output)
    for failure in failures:
        LOGGER.error("Zero-offset repeatability rejected: %s", failure)
    return 0 if not failures else 1


def _benchmark(args: argparse.Namespace) -> int:
    _refuse_input_output_aliases((args.urdf,), (args.output,), "benchmark")
    try:
        import roboticstoolbox as rtb

        from .benchmark import run_perturbation_benchmark
    except ImportError as exc:
        raise RuntimeError(
            "robotkinecal model dependencies are unavailable; install "
            "kinematic-calibration[robotics] in the active Python environment"
        ) from exc
    if args.output.suffix.lower() != ".json":
        raise ValueError("benchmark --output must use the .json extension")
    thresholds = (
        args.min_position_improvement,
        args.min_orientation_improvement,
    )
    _refuse_existing_outputs((args.output,), args.force, "benchmark")
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("improvement thresholds must be in [0, 1]")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    robot = rtb.ERobot.URDF(str(args.urdf.resolve()))
    report = run_perturbation_benchmark(
        robot=robot,
        ee_name=args.ee_name,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        perturbation=args.perturbation,
        seed=args.seed,
        max_iterations=args.max_iterations,
        position_noise_std_m=args.position_noise_std_m,
        orientation_noise_std_rad=args.orientation_noise_std_rad,
        joint_noise_std_rad=args.joint_noise_std_rad,
        outlier_ratio=args.outlier_ratio,
        outlier_position_std_m=args.outlier_position_std_m,
        outlier_orientation_std_rad=args.outlier_orientation_std_rad,
    )
    report["urdf_sha256"] = _sha256(args.urdf)
    report["provenance"] = _runtime_provenance()
    quality_failures = []
    if not report["converged"]:
        quality_failures.append("benchmark optimizer did not converge")
    if report["regressor_rank"] != report["parameter_count"]:
        quality_failures.append("benchmark regressor is rank deficient")
    if report["validation_position_mean_improvement"] < args.min_position_improvement:
        quality_failures.append("validation position improvement is below threshold")
    if (
        report["validation_orientation_mean_improvement"]
        < args.min_orientation_improvement
    ):
        quality_failures.append("validation orientation improvement is below threshold")
    report["quality_thresholds"] = {
        "min_position_improvement": args.min_position_improvement,
        "min_orientation_improvement": args.min_orientation_improvement,
    }
    report["quality_failures"] = quality_failures
    report["passed"] = not quality_failures
    report["accepted"] = report["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, _jsonable(report))
    LOGGER.info("Wrote perturbation benchmark: %s", args.output)
    for failure in quality_failures:
        LOGGER.error("Perturbation benchmark rejected: %s", failure)
    return 0 if report["passed"] else 1


def _validate_urdf_truth(args: argparse.Namespace) -> int:
    _refuse_input_output_aliases(
        (
            args.dataset,
            args.nominal_urdf,
            args.calibrated_urdf,
            args.ground_truth_urdf,
            args.zero_offset_result,
        ),
        (args.output,),
        "validate-urdf-truth",
    )
    from .urdf_truth_validation import compare_urdf_fk_truth

    if args.output.suffix.lower() != ".json":
        raise ValueError("validate-urdf-truth --output must use .json")
    _refuse_existing_outputs((args.output,), args.force, "validate-urdf-truth")
    thresholds = (
        args.min_position_improvement,
        args.min_orientation_improvement,
    )
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds
    ):
        raise ValueError("FK improvement thresholds must be finite in [0, 1]")
    dataset = KinematicDataset.load(args.dataset)
    coverage = dataset.get_summary()
    injected_bias = (
        None
        if args.injected_joint_bias_rad is None
        else np.asarray(args.injected_joint_bias_rad, dtype=float)
    )
    estimated_offset = None
    if args.zero_offset_result is not None:
        result = json.loads(args.zero_offset_result.read_text())
        values = result.get("zero_offset_rad")
        if values is None and isinstance(result.get("joints"), list):
            values = [item.get("zero_offset_rad") for item in result["joints"]]
        if values is None or any(value is None for value in values):
            raise ValueError("zero-offset result has incomplete zero_offset_rad values")
        estimated_offset = np.asarray(values, dtype=float)
    report = compare_urdf_fk_truth(
        args.nominal_urdf,
        args.calibrated_urdf,
        args.ground_truth_urdf,
        np.asarray(coverage["joint_min_rad"]),
        np.asarray(coverage["joint_max_rad"]),
        ee_name=args.ee_name,
        samples=args.samples,
        seed=args.seed,
        injected_bias_rad=injected_bias,
        estimated_offset_rad=estimated_offset,
        sampling_domain=args.sampling_domain,
    )
    failures = []
    if report["rms_improvement"]["position"] < args.min_position_improvement:
        failures.append("position RMS improvement is below threshold")
    if report["rms_improvement"]["orientation"] < args.min_orientation_improvement:
        failures.append("orientation RMS improvement is below threshold")
    report["dataset"] = str(args.dataset.resolve())
    report["dataset_sha256"] = _sha256(args.dataset)
    report["quality_thresholds"] = {
        "min_position_improvement": args.min_position_improvement,
        "min_orientation_improvement": args.min_orientation_improvement,
    }
    report["quality_failures"] = failures
    report["accepted"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, _jsonable(report))
    print(json.dumps(_jsonable(report), indent=2))
    return 0 if report["accepted"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "template":
            return _create_template(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "split-dataset":
            return _split_dataset(args)
        if args.command == "deduplicate-dataset":
            return _deduplicate_dataset(args)
        if args.command == "upgrade-dataset":
            return _upgrade_dataset(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "calibrate":
            return _calibrate(args)
        if args.command == "zero-offset":
            return _zero_offset(args)
        if args.command == "export-urdf":
            return _export_urdf(args)
        if args.command == "verify-result":
            return _verify_result(args)
        if args.command == "verify-collection":
            return _verify_collection(args)
        if args.command == "verify-split":
            return _verify_split(args)
        if args.command == "verify-deduplication":
            return _verify_deduplication(args)
        if args.command == "verify-validation":
            return _verify_validation(args)
        if args.command == "validate-result":
            return _validate_result(args)
        if args.command == "validate-urdf-truth":
            return _validate_urdf_truth(args)
        if args.command == "compare-zero-offsets":
            return _compare_zero_offsets(args)
        return _solve(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
