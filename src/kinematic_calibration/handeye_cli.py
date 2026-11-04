"""Offline CLI for archived OpenCV eye-to-hand observations."""

import argparse
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from kinematic_calib.io_utils import (
    atomic_write_json,
    atomic_write_pair,
    refuse_input_output_aliases,
)

from .handeye.opencv import (
    MIN_ROTATION_DIRECTION_RATIO,
    MIN_ROTATION_SPAN_RAD,
    MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD,
    OPENCV_ROBOT_WORLD_METHODS,
    _require_opencv,
    evaluate_eye_to_hand_solution,
    handeye_observations_sha256,
    load_opencv_observations,
    measure_handeye_motion,
    opencv_calibration_implementation_sha256,
    solve_opencv_eye_to_hand,
)

LOGGER = logging.getLogger(__name__)


def _preflight_outputs(
    inputs: Sequence[Path], outputs: Sequence[Path], force: bool, command: str
) -> None:
    refuse_input_output_aliases(inputs, outputs, command)
    for index, output in enumerate(outputs):
        refuse_input_output_aliases(outputs[:index], (output,), command)
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise ValueError(f"{command} output is not a regular file: {output}")
    existing = [str(output) for output in outputs if output.exists()]
    if existing and not force:
        raise ValueError(
            f"{command} output exists; use --force to replace it: " + ", ".join(existing)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve OpenCV eye-to-hand calibration from an archived NPZ."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="solve an archived observation set")
    solve.add_argument("--samples", type=Path, required=True, help="observation NPZ")
    solve.add_argument("--output", type=Path, required=True, help="result JSON")
    solve.add_argument(
        "--method",
        choices=sorted(OPENCV_ROBOT_WORLD_METHODS),
        default="shah",
        help="OpenCV robot-world/hand-eye method (default: shah)",
    )
    solve.add_argument(
        "--max-position-error",
        type=float,
        default=0.01,
        help="maximum accepted closure position error in metres (default: 0.01)",
    )
    solve.add_argument(
        "--max-orientation-error",
        type=float,
        default=0.05,
        help="maximum accepted closure rotation error in radians (default: 0.05)",
    )
    solve.add_argument("--force", action="store_true", help="replace output")
    verify = commands.add_parser("verify", help="verify a result against its samples")
    verify.add_argument("--samples", type=Path, required=True, help="observation NPZ")
    verify.add_argument("--result", type=Path, required=True, help="result JSON")
    export = commands.add_parser(
        "export", help="verify and export robot-body external transforms"
    )
    export.add_argument("--samples", type=Path, required=True, help="observation NPZ")
    export.add_argument(
        "--result", type=Path, required=True, help="accepted result JSON"
    )
    export.add_argument(
        "--base-t-camera", type=Path, required=True, help="Base_T_Camera NPY output"
    )
    export.add_argument(
        "--ee-t-target",
        type=Path,
        required=True,
        help="EE_T_Target NPY output; target is the calibration board",
    )
    export.add_argument("--force", action="store_true", help="replace both outputs")
    compare = commands.add_parser("compare", help="compare SHAH and LI solutions")
    compare.add_argument("--samples", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--max-position-error", type=float, default=0.01)
    compare.add_argument("--max-orientation-error", type=float, default=0.05)
    compare.add_argument(
        "--max-method-translation-difference", type=float, default=0.01
    )
    compare.add_argument("--max-method-rotation-difference", type=float, default=0.05)
    compare.add_argument("--force", action="store_true")
    inspect = commands.add_parser("inspect", help="inspect archived motion excitation")
    inspect.add_argument("--samples", type=Path, required=True)
    inspect.add_argument("--min-rotation-rank", type=int, default=2)
    inspect.add_argument(
        "--min-rotation-span", type=float, default=MIN_ROTATION_SPAN_RAD
    )
    inspect.add_argument(
        "--min-second-rotation-singular-value",
        type=float,
        default=MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD,
    )
    inspect.add_argument(
        "--min-rotation-direction-ratio",
        type=float,
        default=MIN_ROTATION_DIRECTION_RATIO,
    )
    influence = commands.add_parser(
        "influence", help="run leave-one-out candidate influence diagnostics"
    )
    influence.add_argument("--samples", type=Path, required=True)
    influence.add_argument("--output", type=Path, required=True)
    influence.add_argument(
        "--method",
        choices=sorted(OPENCV_ROBOT_WORLD_METHODS),
        default="shah",
    )
    influence.add_argument("--position-scale", type=float, default=0.01)
    influence.add_argument("--orientation-scale", type=float, default=0.05)
    influence.add_argument("--force", action="store_true")
    bootstrap = commands.add_parser(
        "bootstrap", help="estimate OpenCV hand-eye sampling stability"
    )
    bootstrap.add_argument("--samples", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument(
        "--method",
        choices=sorted(OPENCV_ROBOT_WORLD_METHODS),
        default="shah",
    )
    bootstrap.add_argument("--replicates", type=int, default=200)
    bootstrap.add_argument("--seed", type=int, default=1234)
    bootstrap.add_argument("--min-success-rate", type=float, default=0.8)
    bootstrap.add_argument("--max-translation-p95", type=float, default=0.01)
    bootstrap.add_argument("--max-rotation-p95", type=float, default=0.05)
    bootstrap.add_argument("--force", action="store_true")
    return parser


def _solve(args: argparse.Namespace) -> int:
    if args.samples.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must use the .json extension")
    _preflight_outputs((args.samples,), (args.output,), args.force, "solve")
    thresholds = (args.max_position_error, args.max_orientation_error)
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("error thresholds must be positive finite values")

    observations = load_opencv_observations(args.samples)
    result = solve_opencv_eye_to_hand(
        observations.base_T_ee,
        observations.camera_T_board,
        method=args.method,
    )
    excitation = measure_handeye_motion(observations.base_T_ee)
    failures = []
    if result.position_error_max_m > args.max_position_error:
        failures.append("maximum hand-eye position error exceeds threshold")
    if result.orientation_error_max_rad > args.max_orientation_error:
        failures.append("maximum hand-eye orientation error exceeds threshold")
    report = {
        "schema_version": 1,
        "calibration_type": "eye-to-hand",
        "backend": "opencv",
        "method": result.method,
        "accepted": not failures,
        "samples": len(observations.base_T_ee),
        "observation_equation": (
            "Base_T_EE * EE_T_Board = Base_T_Camera * Camera_T_Board"
        ),
        "observations_npz": str(args.samples),
        "observations_npz_sha256": _sha256(args.samples),
        "observations_sha256": handeye_observations_sha256(
            observations.base_T_ee,
            observations.camera_T_board,
            observations.candidate_indices,
        ),
        "candidate_indices": observations.candidate_indices.tolist(),
        "translation_candidate_count": observations.translation_candidate_count,
        "rotation_candidate_count": observations.rotation_candidate_count,
        "Base_T_Camera": result.base_T_camera.tolist(),
        "EE_T_Board": result.ee_T_board.tolist(),
        "position_error_mean_m": result.position_error_mean_m,
        "position_error_max_m": result.position_error_max_m,
        "orientation_error_mean_rad": result.orientation_error_mean_rad,
        "orientation_error_max_rad": result.orientation_error_max_rad,
        "motion_rotation_rank": result.motion_rotation_rank,
        "max_rotation_from_first_rad": result.max_rotation_from_first_rad,
        "rotation_singular_value_1_rad": excitation["rotation_singular_value_1_rad"],
        "rotation_singular_value_2_rad": excitation["rotation_singular_value_2_rad"],
        "rotation_direction_ratio": excitation["rotation_direction_ratio"],
        "position_error_worst_candidate_index": int(
            observations.candidate_indices[np.argmax(result.position_errors_m)]
        ),
        "orientation_error_worst_candidate_index": int(
            observations.candidate_indices[np.argmax(result.orientation_errors_rad)]
        ),
        "sample_errors": [
            {
                "sample_index": index,
                "candidate_index": int(observations.candidate_indices[index]),
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
                    result.position_errors_m,
                    result.orientation_errors_rad,
                    strict=True,
                )
            )
        ],
        "quality_thresholds": {
            "max_position_error_m": args.max_position_error,
            "max_orientation_error_rad": args.max_orientation_error,
            "min_rotation_span_rad": MIN_ROTATION_SPAN_RAD,
            "min_second_rotation_singular_value_rad": (
                MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD
            ),
            "min_rotation_direction_ratio": MIN_ROTATION_DIRECTION_RATIO,
        },
        "quality_failures": failures,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": _require_opencv().__version__,
            "numpy_version": np.__version__,
            "implementation_sha256": opencv_calibration_implementation_sha256(),
        },
    }
    atomic_write_json(args.output, report)
    LOGGER.info("Wrote offline OpenCV hand-eye result: %s", args.output)
    LOGGER.info("Base_T_Camera=\n%s", result.base_T_camera)
    LOGGER.info("EE_T_Board=\n%s", result.ee_T_board)
    LOGGER.info(
        "Closure residuals: position_m(mean/max)=%.8g/%.8g, "
        "orientation_rad(mean/max)=%.8g/%.8g",
        result.position_error_mean_m,
        result.position_error_max_m,
        result.orientation_error_mean_rad,
        result.orientation_error_max_rad,
    )
    for failure in failures:
        LOGGER.error("OpenCV hand-eye calibration rejected: %s", failure)
    return 0 if not failures else 1


def _load_verified_result(
    samples_path: Path, result_path: Path
) -> tuple[dict, dict, bool]:
    if samples_path.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if result_path.suffix.lower() != ".json":
        raise ValueError("--result must use the .json extension")
    observations = load_opencv_observations(samples_path)
    report = json.loads(result_path.read_text())
    if not isinstance(report, dict):
        raise ValueError("hand-eye result JSON must contain an object")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported hand-eye result schema version")
    if report.get("calibration_type") != "eye-to-hand":
        raise ValueError("result is not an eye-to-hand calibration")
    if report.get("backend") != "opencv":
        raise ValueError("result was not produced by the OpenCV backend")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("hand-eye result provenance must be an object")
    if (
        provenance.get("implementation_sha256")
        != opencv_calibration_implementation_sha256()
    ):
        raise ValueError("hand-eye result implementation SHA-256 mismatch")
    if report.get("observations_npz_sha256") != _sha256(samples_path):
        raise ValueError("hand-eye observation NPZ SHA-256 mismatch")
    content_hash = handeye_observations_sha256(
        observations.base_T_ee,
        observations.camera_T_board,
        observations.candidate_indices,
    )
    if report.get("observations_sha256") != content_hash:
        raise ValueError("hand-eye observation content SHA-256 mismatch")
    if report.get("samples") != len(observations.base_T_ee):
        raise ValueError("hand-eye result sample count is inconsistent")
    if report.get("candidate_indices") != observations.candidate_indices.tolist():
        raise ValueError("hand-eye result candidate indices are inconsistent")
    if (
        report.get("translation_candidate_count")
        != observations.translation_candidate_count
        or report.get("rotation_candidate_count")
        != observations.rotation_candidate_count
    ):
        raise ValueError("hand-eye result candidate counts are inconsistent")
    if report.get("method") not in OPENCV_ROBOT_WORLD_METHODS:
        raise ValueError("hand-eye result has an unsupported OpenCV method")
    excitation = measure_handeye_motion(observations.base_T_ee)
    if report.get("motion_rotation_rank") != excitation["motion_rotation_rank"]:
        raise ValueError("hand-eye result motion rotation rank is inconsistent")
    excitation_fields = (
        "max_rotation_from_first_rad",
        "rotation_singular_value_1_rad",
        "rotation_singular_value_2_rad",
        "rotation_direction_ratio",
    )
    for field in excitation_fields:
        expected = report.get(field)
        if not isinstance(expected, (int, float)) or not np.isclose(
            expected, excitation[field], rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"hand-eye result {field} is inconsistent")
    metrics = evaluate_eye_to_hand_solution(
        observations.base_T_ee,
        observations.camera_T_board,
        np.asarray(report.get("Base_T_Camera"), dtype=np.float64),
        np.asarray(report.get("EE_T_Board"), dtype=np.float64),
    )
    aggregate_fields = (
        "position_error_mean_m",
        "position_error_max_m",
        "orientation_error_mean_rad",
        "orientation_error_max_rad",
    )
    for field in aggregate_fields:
        actual = metrics[field]
        expected = report.get(field)
        if not isinstance(expected, (int, float)) or not np.isclose(
            expected, actual, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"hand-eye result {field} is inconsistent")
    thresholds = report.get("quality_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("hand-eye result quality_thresholds must be an object")
    position_threshold = thresholds.get("max_position_error_m")
    orientation_threshold = thresholds.get("max_orientation_error_rad")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value <= 0.0
        for value in (position_threshold, orientation_threshold)
    ):
        raise ValueError("hand-eye result quality thresholds are invalid")
    expected_excitation_thresholds = {
        "min_rotation_span_rad": MIN_ROTATION_SPAN_RAD,
        "min_second_rotation_singular_value_rad": (
            MIN_SECOND_ROTATION_SINGULAR_VALUE_RAD
        ),
        "min_rotation_direction_ratio": MIN_ROTATION_DIRECTION_RATIO,
    }
    for field, expected in expected_excitation_thresholds.items():
        if thresholds.get(field) != expected:
            raise ValueError(f"hand-eye result {field} threshold is inconsistent")
    failures = []
    if metrics["position_error_max_m"] > position_threshold:
        failures.append("maximum hand-eye position error exceeds threshold")
    if metrics["orientation_error_max_rad"] > orientation_threshold:
        failures.append("maximum hand-eye orientation error exceeds threshold")
    if report.get("quality_failures") != failures:
        raise ValueError("hand-eye result quality failures are inconsistent")
    expected_sample_errors = [
        {
            "sample_index": index,
            "candidate_index": int(observations.candidate_indices[index]),
            "position_error_m": float(position_error),
            "orientation_error_rad": float(orientation_error),
            "position_exceeds_threshold": bool(position_error > position_threshold),
            "orientation_exceeds_threshold": bool(
                orientation_error > orientation_threshold
            ),
        }
        for index, (position_error, orientation_error) in enumerate(
            zip(
                metrics["position_errors_m"],
                metrics["orientation_errors_rad"],
                strict=True,
            )
        )
    ]
    if report.get("sample_errors") != expected_sample_errors:
        raise ValueError("hand-eye result per-sample errors are inconsistent")
    if report.get("position_error_worst_candidate_index") != int(
        observations.candidate_indices[np.argmax(metrics["position_errors_m"])]
    ):
        raise ValueError("hand-eye worst position candidate is inconsistent")
    if report.get("orientation_error_worst_candidate_index") != int(
        observations.candidate_indices[np.argmax(metrics["orientation_errors_rad"])]
    ):
        raise ValueError("hand-eye worst orientation candidate is inconsistent")
    accepted = report.get("accepted") is True
    if accepted != (not failures):
        raise ValueError("hand-eye result acceptance is inconsistent")
    summary = {
        "integrity_verified": True,
        "accepted": accepted,
        "method": report.get("method"),
        "samples": len(observations.base_T_ee),
        "observations_npz_sha256": _sha256(samples_path),
        **{field: metrics[field] for field in aggregate_fields},
    }
    return report, summary, accepted


def _verify(args: argparse.Namespace) -> int:
    _, summary, accepted = _load_verified_result(args.samples, args.result)
    print(json.dumps(summary, indent=2))
    return 0 if accepted else 1


def _export(args: argparse.Namespace) -> int:
    outputs = (args.base_t_camera, args.ee_t_target)
    if any(path.suffix.lower() != ".npy" for path in outputs):
        raise ValueError("transform outputs must use the .npy extension")
    resolved = [path.resolve() for path in (args.samples, args.result, *outputs)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("sample, result, and transform output paths must be different")
    _preflight_outputs((args.samples, args.result), outputs, args.force, "export")
    report, summary, accepted = _load_verified_result(args.samples, args.result)
    if not accepted:
        raise ValueError("refusing to export a rejected hand-eye calibration")
    base_T_camera = np.asarray(report["Base_T_Camera"], dtype=np.float64)
    ee_T_target = np.asarray(report["EE_T_Board"], dtype=np.float64)
    base_buffer, target_buffer = io.BytesIO(), io.BytesIO()
    np.save(base_buffer, base_T_camera, allow_pickle=False)
    np.save(target_buffer, ee_T_target, allow_pickle=False)
    atomic_write_pair(
        args.base_t_camera,
        base_buffer.getvalue(),
        args.ee_t_target,
        target_buffer.getvalue(),
    )
    LOGGER.info("Wrote Base_T_Camera: %s", args.base_t_camera)
    LOGGER.info("Wrote EE_T_Target (calibration board): %s", args.ee_t_target)
    print(
        json.dumps(
            {
                **summary,
                "Base_T_Camera_path": str(args.base_t_camera),
                "Base_T_Camera_sha256": _sha256(args.base_t_camera),
                "EE_T_Target_path": str(args.ee_t_target),
                "EE_T_Target_sha256": _sha256(args.ee_t_target),
            },
            indent=2,
        )
    )
    return 0


def _transform_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    delta = np.linalg.inv(first) @ second
    cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "translation_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_rad": float(np.arccos(cosine)),
    }


def _compare(args: argparse.Namespace) -> int:
    if args.samples.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must use the .json extension")
    _preflight_outputs((args.samples,), (args.output,), args.force, "compare")
    thresholds = (
        args.max_position_error,
        args.max_orientation_error,
        args.max_method_translation_difference,
        args.max_method_rotation_difference,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("comparison thresholds must be positive finite values")
    observations = load_opencv_observations(args.samples)
    solutions = {
        method: solve_opencv_eye_to_hand(
            observations.base_T_ee,
            observations.camera_T_board,
            method=method,
        )
        for method in ("shah", "li")
    }
    base_difference = _transform_difference(
        solutions["shah"].base_T_camera, solutions["li"].base_T_camera
    )
    board_difference = _transform_difference(
        solutions["shah"].ee_T_board, solutions["li"].ee_T_board
    )
    failures = []
    for method, solution in solutions.items():
        if solution.position_error_max_m > args.max_position_error:
            failures.append(f"{method} maximum position error exceeds threshold")
        if solution.orientation_error_max_rad > args.max_orientation_error:
            failures.append(f"{method} maximum orientation error exceeds threshold")
    for name, difference in (
        ("Base_T_Camera", base_difference),
        ("EE_T_Board", board_difference),
    ):
        if difference["translation_m"] > args.max_method_translation_difference:
            failures.append(f"{name} method translation difference exceeds threshold")
        if difference["rotation_rad"] > args.max_method_rotation_difference:
            failures.append(f"{name} method rotation difference exceeds threshold")
    scores = {
        method: solution.position_error_max_m / args.max_position_error
        + solution.orientation_error_max_rad / args.max_orientation_error
        for method, solution in solutions.items()
    }
    recommended_method = min(scores, key=scores.get)

    def render_solution(solution) -> dict:
        return {
            "Base_T_Camera": solution.base_T_camera.tolist(),
            "EE_T_Board": solution.ee_T_board.tolist(),
            "position_error_mean_m": solution.position_error_mean_m,
            "position_error_max_m": solution.position_error_max_m,
            "orientation_error_mean_rad": solution.orientation_error_mean_rad,
            "orientation_error_max_rad": solution.orientation_error_max_rad,
            "normalized_quality_score": scores[solution.method],
        }

    report = {
        "schema_version": 1,
        "report_type": "opencv-eye-to-hand-method-comparison",
        "accepted": not failures,
        "samples": len(observations.base_T_ee),
        "observations_npz_sha256": _sha256(args.samples),
        "observations_sha256": handeye_observations_sha256(
            observations.base_T_ee,
            observations.camera_T_board,
            observations.candidate_indices,
        ),
        "candidate_indices": observations.candidate_indices.tolist(),
        "solutions": {
            method: render_solution(solution) for method, solution in solutions.items()
        },
        "method_differences": {
            "Base_T_Camera": base_difference,
            "EE_T_Board": board_difference,
        },
        "recommended_method": recommended_method,
        "quality_thresholds": {
            "max_position_error_m": args.max_position_error,
            "max_orientation_error_rad": args.max_orientation_error,
            "max_method_translation_difference_m": (
                args.max_method_translation_difference
            ),
            "max_method_rotation_difference_rad": args.max_method_rotation_difference,
        },
        "quality_failures": failures,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": _require_opencv().__version__,
            "numpy_version": np.__version__,
            "implementation_sha256": opencv_calibration_implementation_sha256(),
        },
    }
    atomic_write_json(args.output, report)
    LOGGER.info("Wrote OpenCV method comparison: %s", args.output)
    LOGGER.info("Recommended OpenCV method: %s", recommended_method)
    for failure in failures:
        LOGGER.error("OpenCV method comparison rejected: %s", failure)
    return 0 if not failures else 1


def _inspect(args: argparse.Namespace) -> int:
    if args.samples.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if args.min_rotation_rank <= 0:
        raise ValueError("--min-rotation-rank must be positive")
    if not np.isfinite(args.min_rotation_span) or args.min_rotation_span <= 0.0:
        raise ValueError("--min-rotation-span must be positive and finite")
    extra_thresholds = (
        args.min_second_rotation_singular_value,
        args.min_rotation_direction_ratio,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in extra_thresholds):
        raise ValueError("rotation direction thresholds must be positive and finite")
    observations = load_opencv_observations(args.samples)
    excitation = measure_handeye_motion(observations.base_T_ee)
    failures = []
    if excitation["motion_rotation_rank"] < args.min_rotation_rank:
        failures.append("motion rotation rank is below threshold")
    if excitation["max_rotation_from_first_rad"] < args.min_rotation_span:
        failures.append("motion rotation span is below threshold")
    if (
        excitation["rotation_singular_value_2_rad"]
        < args.min_second_rotation_singular_value
    ):
        failures.append("second rotation singular value is below threshold")
    if excitation["rotation_direction_ratio"] < args.min_rotation_direction_ratio:
        failures.append("rotation direction ratio is below threshold")
    candidate_count = (
        observations.translation_candidate_count + observations.rotation_candidate_count
    )
    report = {
        "schema_version": 1,
        "report_type": "opencv-eye-to-hand-observation-inspection",
        "accepted": not failures,
        "samples": len(observations.base_T_ee),
        "candidate_count": candidate_count,
        "detection_success_rate": len(observations.base_T_ee) / candidate_count,
        "candidate_indices": observations.candidate_indices.tolist(),
        **excitation,
        "quality_thresholds": {
            "min_rotation_rank": args.min_rotation_rank,
            "min_rotation_span_rad": args.min_rotation_span,
            "min_second_rotation_singular_value_rad": (
                args.min_second_rotation_singular_value
            ),
            "min_rotation_direction_ratio": args.min_rotation_direction_ratio,
        },
        "quality_failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def _influence(args: argparse.Namespace) -> int:
    if args.samples.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must use the .json extension")
    _preflight_outputs((args.samples,), (args.output,), args.force, "influence")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (args.position_scale, args.orientation_scale)
    ):
        raise ValueError("influence scales must be positive finite values")
    observations = load_opencv_observations(args.samples)
    if len(observations.base_T_ee) < 4:
        raise ValueError("leave-one-out influence analysis requires at least 4 samples")
    full = solve_opencv_eye_to_hand(
        observations.base_T_ee,
        observations.camera_T_board,
        method=args.method,
    )
    entries = []
    for sample_index in range(len(observations.base_T_ee)):
        keep = np.arange(len(observations.base_T_ee)) != sample_index
        try:
            reduced = solve_opencv_eye_to_hand(
                observations.base_T_ee[keep],
                observations.camera_T_board[keep],
                method=args.method,
            )
        except ValueError as exc:
            entries.append(
                {
                    "sample_index": sample_index,
                    "candidate_index": int(
                        observations.candidate_indices[sample_index]
                    ),
                    "reduced_solve_succeeded": False,
                    "failure": str(exc),
                }
            )
            continue
        base_shift = _transform_difference(full.base_T_camera, reduced.base_T_camera)
        position_improvement = full.position_error_max_m - reduced.position_error_max_m
        orientation_improvement = (
            full.orientation_error_max_rad - reduced.orientation_error_max_rad
        )
        influence_score = (
            position_improvement / args.position_scale
            + orientation_improvement / args.orientation_scale
        )
        entries.append(
            {
                "sample_index": sample_index,
                "candidate_index": int(observations.candidate_indices[sample_index]),
                "reduced_solve_succeeded": True,
                "reduced_position_error_max_m": reduced.position_error_max_m,
                "reduced_orientation_error_max_rad": (
                    reduced.orientation_error_max_rad
                ),
                "position_error_max_improvement_m": position_improvement,
                "orientation_error_max_improvement_rad": orientation_improvement,
                "Base_T_Camera_shift": base_shift,
                "normalized_influence_score": influence_score,
            }
        )
    successful_entries = [
        entry for entry in entries if entry["reduced_solve_succeeded"]
    ]
    most_influential = max(
        successful_entries,
        key=lambda entry: entry["normalized_influence_score"],
        default=None,
    )
    report = {
        "schema_version": 1,
        "report_type": "opencv-eye-to-hand-leave-one-out-influence",
        "method": args.method,
        "samples": len(observations.base_T_ee),
        "observations_npz_sha256": _sha256(args.samples),
        "observations_sha256": handeye_observations_sha256(
            observations.base_T_ee,
            observations.camera_T_board,
            observations.candidate_indices,
        ),
        "full_solution": {
            "position_error_max_m": full.position_error_max_m,
            "orientation_error_max_rad": full.orientation_error_max_rad,
            "Base_T_Camera": full.base_T_camera.tolist(),
        },
        "normalization_scales": {
            "position_scale_m": args.position_scale,
            "orientation_scale_rad": args.orientation_scale,
        },
        "most_influential_candidate_index": (
            most_influential["candidate_index"]
            if most_influential is not None
            else None
        ),
        "sample_influence": entries,
        "note": (
            "diagnostic only; do not remove a sample solely because it has the "
            "largest influence score"
        ),
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": _require_opencv().__version__,
            "numpy_version": np.__version__,
            "implementation_sha256": opencv_calibration_implementation_sha256(),
        },
    }
    atomic_write_json(args.output, report)
    LOGGER.info("Wrote OpenCV leave-one-out influence report: %s", args.output)
    return 0


def _bootstrap(args: argparse.Namespace) -> int:
    if args.samples.suffix.lower() != ".npz":
        raise ValueError("--samples must use the .npz extension")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must use the .json extension")
    _preflight_outputs((args.samples,), (args.output,), args.force, "bootstrap")
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")
    if not np.isfinite(args.min_success_rate) or not 0.0 < args.min_success_rate <= 1.0:
        raise ValueError("--min-success-rate must be in (0, 1]")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (args.max_translation_p95, args.max_rotation_p95)
    ):
        raise ValueError(
            "bootstrap stability thresholds must be positive finite values"
        )
    observations = load_opencv_observations(args.samples)
    baseline = solve_opencv_eye_to_hand(
        observations.base_T_ee,
        observations.camera_T_board,
        method=args.method,
    )
    generator = np.random.default_rng(args.seed)
    translation_shifts = []
    rotation_shifts = []
    failed_replicates = []
    sample_count = len(observations.base_T_ee)
    for replicate in range(args.replicates):
        indices = generator.choice(sample_count, size=sample_count, replace=True)
        try:
            solution = solve_opencv_eye_to_hand(
                observations.base_T_ee[indices],
                observations.camera_T_board[indices],
                method=args.method,
            )
        except ValueError as exc:
            failed_replicates.append({"replicate": replicate, "failure": str(exc)})
            continue
        difference = _transform_difference(
            baseline.base_T_camera, solution.base_T_camera
        )
        translation_shifts.append(difference["translation_m"])
        rotation_shifts.append(difference["rotation_rad"])
    success_count = len(translation_shifts)
    success_rate = success_count / args.replicates
    failures = []
    if success_rate < args.min_success_rate:
        failures.append("bootstrap solve success rate is below threshold")
    if success_count:
        translation_p95 = float(np.percentile(translation_shifts, 95))
        rotation_p95 = float(np.percentile(rotation_shifts, 95))
        if translation_p95 > args.max_translation_p95:
            failures.append("bootstrap translation P95 exceeds threshold")
        if rotation_p95 > args.max_rotation_p95:
            failures.append("bootstrap rotation P95 exceeds threshold")
        statistics = {
            "translation_mean_m": float(np.mean(translation_shifts)),
            "translation_std_m": float(np.std(translation_shifts, ddof=0)),
            "translation_p95_m": translation_p95,
            "translation_max_m": float(np.max(translation_shifts)),
            "rotation_mean_rad": float(np.mean(rotation_shifts)),
            "rotation_std_rad": float(np.std(rotation_shifts, ddof=0)),
            "rotation_p95_rad": rotation_p95,
            "rotation_max_rad": float(np.max(rotation_shifts)),
        }
    else:
        failures.append("all bootstrap solves failed")
        statistics = None
    report = {
        "schema_version": 1,
        "report_type": "opencv-eye-to-hand-bootstrap-stability",
        "accepted": not failures,
        "method": args.method,
        "samples": sample_count,
        "replicates": args.replicates,
        "seed": args.seed,
        "successful_replicates": success_count,
        "failed_replicates_count": len(failed_replicates),
        "success_rate": success_rate,
        "observations_npz_sha256": _sha256(args.samples),
        "observations_sha256": handeye_observations_sha256(
            observations.base_T_ee,
            observations.camera_T_board,
            observations.candidate_indices,
        ),
        "baseline_Base_T_Camera": baseline.base_T_camera.tolist(),
        "stability_statistics": statistics,
        "quality_thresholds": {
            "min_success_rate": args.min_success_rate,
            "max_translation_p95_m": args.max_translation_p95,
            "max_rotation_p95_rad": args.max_rotation_p95,
        },
        "quality_failures": failures,
        "failed_replicates": failed_replicates,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": _require_opencv().__version__,
            "numpy_version": np.__version__,
            "implementation_sha256": opencv_calibration_implementation_sha256(),
        },
    }
    atomic_write_json(args.output, report)
    LOGGER.info("Wrote OpenCV bootstrap stability report: %s", args.output)
    for failure in failures:
        LOGGER.error("OpenCV bootstrap stability rejected: %s", failure)
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = build_parser().parse_args(argv)
        if args.command == "verify":
            return _verify(args)
        if args.command == "export":
            return _export(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "influence":
            return _influence(args)
        if args.command == "bootstrap":
            return _bootstrap(args)
        return _solve(args)
    except (EOFError, BadZipFile) as exc:
        LOGGER.error("invalid hand-eye observation NPZ %s: %s", args.samples, exc)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
