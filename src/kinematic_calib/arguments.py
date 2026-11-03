"""Argument definitions, independent of solver and artifact execution."""

import argparse
from pathlib import Path

from .dataset import ObservationFrame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a serial robot model from synchronized poses."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="create an NPZ data template")
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--samples", type=int, default=20)
    template.add_argument("--dof", type=int, default=6)
    template.add_argument(
        "--frame",
        choices=[item.value for item in ObservationFrame],
        default=ObservationFrame.CAMERA_T_TARGET.value,
    )

    inspect = subparsers.add_parser(
        "inspect", help="validate a dataset and print coverage metrics"
    )
    inspect.add_argument("--dataset", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    inspect.add_argument("--min-joint-span", type=float, default=0.1)
    inspect.add_argument("--min-translation-span", type=float, default=0.05)
    inspect.add_argument("--min-rotation-span", type=float, default=0.2)
    inspect.add_argument("--fail-on-low-diversity", action="store_true")

    split = subparsers.add_parser(
        "split-dataset", help="create reproducible train and validation datasets"
    )
    split.add_argument("--dataset", type=Path, required=True)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--validation-output", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--validation-count", type=int)
    split.add_argument("--seed", type=int, default=1234)
    split.add_argument("--search-attempts", type=int, default=1)
    split.add_argument("--min-joint-span", type=float, default=0.1)
    split.add_argument("--min-translation-span", type=float, default=0.05)
    split.add_argument("--min-rotation-span", type=float, default=0.2)
    split.add_argument("--fail-on-low-diversity", action="store_true")
    split.add_argument("--force", action="store_true")

    deduplicate = subparsers.add_parser(
        "deduplicate-dataset",
        help="remove exact repeated observations while preserving first occurrences",
    )
    deduplicate.add_argument("--dataset", type=Path, required=True)
    deduplicate.add_argument("--output", type=Path, required=True)
    deduplicate.add_argument("--manifest", type=Path, required=True)
    deduplicate.add_argument("--force", action="store_true")

    upgrade = subparsers.add_parser(
        "upgrade-dataset", help="rewrite a legacy dataset in the current schema"
    )
    upgrade.add_argument("--dataset", type=Path, required=True)
    upgrade.add_argument("--output", type=Path, required=True)
    upgrade.add_argument("--manifest", type=Path, required=True)
    upgrade.add_argument("--force", action="store_true")

    solve = subparsers.add_parser("solve", help="run kinematic calibration")
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--urdf", type=Path, required=True)
    solve.add_argument(
        "--backend",
        choices=["rtb", "numpy"],
        default="rtb",
        help="model loader (default: rtb); numpy fits POE without Robotics Toolbox and does not export URDF link frames",
    )
    solve.add_argument(
        "--base-link",
        help="optional ancestor link defining the base frame with --backend numpy",
    )
    solve.add_argument(
        "--ee-name",
        required=True,
        help="terminal link of the serial revolute chain to calibrate",
    )
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument(
        "--model-mode",
        choices=["self-calibrating", "composite", "robot-body"],
        required=True,
        help=(
            "self-calibrating estimates a Camera/Target composite model without "
            "external transforms; robot-body uses known Base/Camera and EE/Target"
        ),
    )
    solve.add_argument("--base-t-camera", type=Path)
    solve.add_argument("--ee-t-target", type=Path)
    solve.add_argument("--max-iterations", type=int, default=100)
    solve.add_argument("--damping", type=float, default=0.0)
    solve.add_argument("--max-position-error", type=float, default=0.01)
    solve.add_argument("--max-orientation-error", type=float, default=0.05)
    solve.add_argument("--max-condition-number", type=float, default=1e8)
    solve.add_argument("--min-joint-span", type=float, default=0.1)
    solve.add_argument("--min-translation-span", type=float, default=0.05)
    solve.add_argument("--min-rotation-span", type=float, default=0.2)
    solve.add_argument("--fail-on-low-diversity", action="store_true")
    solve.add_argument("--quiet", action="store_true")
    solve.add_argument(
        "--experimental-nominal-allocation",
        action="store_true",
        help="enable experimental nominal-constrained SE(3) parameter allocation",
    )
    solve.add_argument(
        "--experimental-svd-gauge-allocation",
        action="store_true",
        help=(
            "project only regressor-nullspace geometry toward nominal; "
            "preserves end-effector FK, not intermediate link frames"
        ),
    )
    solve.add_argument(
        "--geometry-anchor-joint",
        help=(
            "experimental: fix this URDF joint to nominal during geometry "
            "post-processing (default: no anchor)"
        ),
    )
    solve.add_argument("--force", action="store_true")

    zero_offset = subparsers.add_parser(
        "zero-offset",
        help="estimate encoder zero offsets with URDF geometry held fixed",
    )
    zero_offset.add_argument("--dataset", type=Path, required=True)
    zero_offset.add_argument("--urdf", type=Path, required=True)
    zero_offset.add_argument("--ee-name", required=True)
    zero_offset.add_argument("--output", type=Path, required=True)
    zero_offset.add_argument(
        "--model-mode",
        choices=["self-calibrating", "robot-body"],
        required=True,
    )
    zero_offset.add_argument("--base-t-camera", type=Path)
    zero_offset.add_argument("--ee-t-target", type=Path)
    zero_offset.add_argument("--max-offset-rad", type=float, default=0.5)
    zero_offset.add_argument("--max-offset-std-rad", type=float, default=0.01)
    zero_offset.add_argument("--max-position-error", type=float, default=0.01)
    zero_offset.add_argument("--max-orientation-error", type=float, default=0.05)
    zero_offset.add_argument("--max-condition-number", type=float, default=1e8)
    zero_offset.add_argument("--min-joint-span", type=float, default=0.1)
    zero_offset.add_argument("--min-translation-span", type=float, default=0.05)
    zero_offset.add_argument("--min-rotation-span", type=float, default=0.2)
    zero_offset.add_argument("--fail-on-low-diversity", action="store_true")
    zero_offset.add_argument("--force", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate",
        help=(
            "run the post-collection hand-eye, geometry, zero-offset, "
            "verification, and URDF-export workflow"
        ),
    )
    calibrate.add_argument("--dataset", type=Path, required=True)
    calibrate.add_argument(
        "--urdf",
        type=Path,
        help="source URDF (default: repository UR10 asset)",
    )
    calibrate.add_argument("--ee-name", default="ee_link")
    calibrate.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory (default: DATASET parent/calibration)",
    )
    calibrate.add_argument(
        "--calibration-type",
        choices=["geometry", "zero-offset", "both"],
        default="both",
        help="calibration stages to run after hand-eye (default: both)",
    )
    calibrate.add_argument(
        "--handeye-samples",
        type=Path,
        help=(
            "archived Base_T_EE/Camera_T_Board observations; when supplied, "
            "OpenCV hand-eye is attempted before kinematic calibration"
        ),
    )
    calibrate.add_argument(
        "--handeye-result",
        type=Path,
        help=(
            "reuse a verified hand-eye JSON, or write the solve here; default "
            "is OUTPUT_DIR/handeye_result.json"
        ),
    )
    calibrate.add_argument("--opencv-method", choices=["shah", "li"], default="shah")
    calibrate.add_argument(
        "--no-use-handeye-extrinsics",
        action="store_true",
        help=(
            "still solve/verify hand-eye, but do not pass its transforms to "
            "geometry or zero-offset calibration"
        ),
    )
    calibrate.add_argument(
        "--skip-handeye",
        action="store_true",
        help="do not auto-discover or solve hand-eye observations",
    )
    calibrate.add_argument(
        "--output-detail",
        choices=["concise", "full"],
        default="concise",
        help="terminal output detail (default: concise final result only)",
    )
    calibrate.add_argument("--max-iterations", type=int, default=100)
    calibrate.add_argument("--damping", type=float, default=0.0)
    calibrate.add_argument("--max-offset-rad", type=float, default=0.5)
    calibrate.add_argument("--max-offset-std-rad", type=float, default=0.01)
    calibrate.add_argument("--max-position-error", type=float, default=0.01)
    calibrate.add_argument("--max-orientation-error", type=float, default=0.05)
    calibrate.add_argument("--max-condition-number", type=float, default=1e8)
    calibrate.add_argument("--min-joint-span", type=float, default=0.1)
    calibrate.add_argument("--min-translation-span", type=float, default=0.05)
    calibrate.add_argument("--min-rotation-span", type=float, default=0.2)
    calibrate.add_argument("--fail-on-low-diversity", action="store_true")
    calibrate.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    calibrate.add_argument(
        "--experimental-nominal-allocation",
        action="store_true",
        help="enable experimental nominal-constrained SE(3) parameter allocation",
    )
    calibrate.add_argument(
        "--experimental-svd-gauge-allocation",
        action="store_true",
        help=(
            "project only regressor-nullspace geometry toward nominal; "
            "preserves end-effector FK only"
        ),
    )
    calibrate.add_argument(
        "--geometry-anchor-joint",
        help=(
            "experimental: fix this URDF joint to nominal during geometry "
            "post-processing (default: no anchor)"
        ),
    )
    calibrate.add_argument("--force", action="store_true")

    export_urdf = subparsers.add_parser(
        "export-urdf", help="write an accepted calibration into a new URDF"
    )
    export_urdf.add_argument("--result", type=Path, required=True)
    export_urdf.add_argument("--urdf", type=Path, required=True)
    export_urdf.add_argument("--output", type=Path, required=True)
    export_urdf.add_argument("--force", action="store_true")
    export_urdf.add_argument("--allow-partial", action="store_true")

    verify_result = subparsers.add_parser(
        "verify-result", help="verify calibration artifact hashes and acceptance"
    )
    verify_result.add_argument("--result", type=Path, required=True)
    verify_result.add_argument("--dataset", type=Path, required=True)
    verify_result.add_argument("--urdf", type=Path, required=True)
    verify_result.add_argument("--parameters", type=Path)
    verify_result.add_argument("--base-t-camera", type=Path)
    verify_result.add_argument("--ee-t-target", type=Path)

    verify_collection = subparsers.add_parser(
        "verify-collection", help="verify a dataset against its collection report"
    )
    verify_collection.add_argument("--dataset", type=Path, required=True)
    verify_collection.add_argument("--report", type=Path, required=True)

    verify_split = subparsers.add_parser(
        "verify-split", help="verify train/validation datasets against a split manifest"
    )
    verify_split.add_argument("--manifest", type=Path, required=True)
    verify_split.add_argument("--source", type=Path, required=True)
    verify_split.add_argument("--train", type=Path, required=True)
    verify_split.add_argument("--validation", type=Path, required=True)

    verify_deduplication = subparsers.add_parser(
        "verify-deduplication",
        help="verify a deduplicated dataset and its derivation manifest",
    )
    verify_deduplication.add_argument("--manifest", type=Path, required=True)
    verify_deduplication.add_argument("--source", type=Path, required=True)
    verify_deduplication.add_argument("--output", type=Path, required=True)

    verify_validation = subparsers.add_parser(
        "verify-validation",
        help="verify an archived independent-validation artifact set",
    )
    verify_validation.add_argument("--report", type=Path, required=True)
    verify_validation.add_argument("--parameters", type=Path, required=True)
    verify_validation.add_argument("--dataset", type=Path, required=True)
    verify_validation.add_argument("--result", type=Path, required=True)

    validate_result = subparsers.add_parser(
        "validate-result", help="evaluate saved parameters on another dataset"
    )
    validate_result.add_argument("--parameters", type=Path, required=True)
    validate_result.add_argument(
        "--result",
        type=Path,
        help="accepted solve JSON (default: same stem as --parameters)",
    )
    validate_result.add_argument("--allow-unverified-parameters", action="store_true")
    validate_result.add_argument("--allow-implementation-mismatch", action="store_true")
    validate_result.add_argument("--dataset", type=Path, required=True)
    validate_result.add_argument("--output", type=Path)
    validate_result.add_argument("--base-t-camera", type=Path)
    validate_result.add_argument("--ee-t-target", type=Path)
    validate_result.add_argument("--allow-training-data", action="store_true")
    validate_result.add_argument("--max-position-error", type=float, default=0.01)
    validate_result.add_argument("--max-orientation-error", type=float, default=0.05)
    validate_result.add_argument("--min-validation-samples", type=int, default=5)
    validate_result.add_argument("--min-joint-span", type=float, default=0.1)
    validate_result.add_argument("--min-translation-span", type=float, default=0.05)
    validate_result.add_argument("--min-rotation-span", type=float, default=0.2)
    validate_result.add_argument("--force", action="store_true")

    validate_urdf_truth = subparsers.add_parser(
        "validate-urdf-truth",
        help="compare nominal and calibrated FK with a known physical URDF",
    )
    validate_urdf_truth.add_argument("--nominal-urdf", type=Path, required=True)
    validate_urdf_truth.add_argument("--calibrated-urdf", type=Path, required=True)
    validate_urdf_truth.add_argument("--ground-truth-urdf", type=Path, required=True)
    validate_urdf_truth.add_argument("--dataset", type=Path, required=True)
    validate_urdf_truth.add_argument("--ee-name", default="ee_link")
    validate_urdf_truth.add_argument("--samples", type=int, default=300)
    validate_urdf_truth.add_argument("--seed", type=int, default=1234)
    validate_urdf_truth.add_argument(
        "--sampling-domain",
        choices=["joint-limits", "observed-range"],
        default="joint-limits",
        help="random-joint domain (default: full ground-truth URDF limits)",
    )
    validate_urdf_truth.add_argument("--injected-joint-bias-rad", type=float, nargs="+")
    validate_urdf_truth.add_argument("--zero-offset-result", type=Path)
    validate_urdf_truth.add_argument(
        "--min-position-improvement", type=float, default=0.0
    )
    validate_urdf_truth.add_argument(
        "--min-orientation-improvement", type=float, default=0.0
    )
    validate_urdf_truth.add_argument("--output", type=Path, required=True)
    validate_urdf_truth.add_argument("--force", action="store_true")

    compare_offsets = subparsers.add_parser(
        "compare-zero-offsets", help="assess zero-offset repeatability across runs"
    )
    compare_offsets.add_argument(
        "--result", type=Path, action="append", required=True, dest="results"
    )
    compare_offsets.add_argument("--output", type=Path, required=True)
    compare_offsets.add_argument("--max-between-run-std-rad", type=float, default=0.01)
    compare_offsets.add_argument("--allow-duplicate-datasets", action="store_true")
    compare_offsets.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="recover a known perturbed kinematic chain"
    )
    benchmark.add_argument("--urdf", type=Path, required=True)
    benchmark.add_argument("--ee-name", required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--train-samples", type=int, default=40)
    benchmark.add_argument("--validation-samples", type=int, default=20)
    benchmark.add_argument("--perturbation", type=float, default=0.01)
    benchmark.add_argument(
        "--position-noise-std-m",
        type=float,
        default=0.0,
        help="training pose translation noise standard deviation in metres",
    )
    benchmark.add_argument(
        "--orientation-noise-std-rad",
        type=float,
        default=0.0,
        help="training pose RPY noise standard deviation in radians",
    )
    benchmark.add_argument(
        "--joint-noise-std-rad",
        type=float,
        default=0.0,
        help="training joint-angle noise standard deviation in radians",
    )
    benchmark.add_argument(
        "--outlier-ratio",
        type=float,
        default=0.0,
        help="fraction of training poses receiving additional outlier noise",
    )
    benchmark.add_argument("--outlier-position-std-m", type=float, default=0.02)
    benchmark.add_argument("--outlier-orientation-std-rad", type=float, default=0.05)
    benchmark.add_argument("--seed", type=int, default=1234)
    benchmark.add_argument("--max-iterations", type=int, default=100)
    benchmark.add_argument("--min-position-improvement", type=float, default=0.9)
    benchmark.add_argument("--min-orientation-improvement", type=float, default=0.9)
    benchmark.add_argument("--force", action="store_true")
    return parser
