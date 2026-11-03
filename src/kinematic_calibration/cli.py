"""Command-line entry for the dependency-light simulation and viewer."""

import argparse
import json
import math
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Engine-independent kinematic calibration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run a deterministic headless calibration experiment"
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--urdf", type=Path)
    demo.add_argument("--end-link", default="ee_link")
    demo.add_argument("--base-link")
    demo.add_argument("--samples", type=int, default=100)
    demo.add_argument("--validation-samples", type=int, default=40)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--position-noise-m", type=float, default=0.0001)
    demo.add_argument("--orientation-noise-rad", type=float, default=0.0002)
    demo.add_argument("--max-iterations", type=int, default=30)
    demo.add_argument("--force", action="store_true")
    view = commands.add_parser("view", help="Replay an experiment in Viser")
    view.add_argument("experiment", type=Path)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument(
        "--duration",
        type=float,
        help="Stop server after this many seconds (smoke checks)",
    )
    inspect = commands.add_parser(
        "inspect", help="Print an experiment's validation metrics"
    )
    inspect.add_argument("experiment", type=Path)
    capture = commands.add_parser(
        "import-captures",
        help="Import synchronized images or ordered target detections",
    )
    capture.add_argument("manifest", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--detector", choices=("auto", "opencv", "toolkit"), default="auto"
    )
    capture.add_argument("--max-sync-error-s", type=float, default=0.03)
    capture.add_argument("--max-reprojection-rms-px", type=float, default=2.0)
    capture.add_argument("--allow-partial", action="store_true")
    capture.add_argument("--allow-legacy-profile", action="store_true")
    capture.add_argument("--force", action="store_true")
    plan = commands.add_parser(
        "plan", help="Select informative joint configurations from a nominal URDF"
    )
    plan.add_argument("--urdf", type=Path)
    plan.add_argument("--end-link", default="ee_link")
    plan.add_argument("--base-link")
    plan.add_argument("--count", type=int, default=40)
    plan.add_argument("--candidates", type=int, default=400)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--characteristic-length-m", type=float, default=1.0)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .experiment import run_experiment
            from .simulation import MeasurementNoise

            report = run_experiment(
                args.output,
                urdf=args.urdf,
                end_link=args.end_link,
                base_link=args.base_link,
                samples=args.samples,
                validation_samples=args.validation_samples,
                seed=args.seed,
                noise=MeasurementNoise(
                    args.position_noise_m, args.orientation_noise_rad
                ),
                max_iterations=args.max_iterations,
                force=args.force,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["accepted"] else 1
        elif args.command == "inspect":
            from .experiment import load_experiment

            report, _, _ = load_experiment(args.experiment)
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "import-captures":
            from .acquisition import import_captures

            _, report = import_captures(
                args.manifest,
                args.output,
                detector=args.detector,
                max_sync_error_s=args.max_sync_error_s,
                max_reprojection_rms_px=args.max_reprojection_rms_px,
                allow_partial=args.allow_partial,
                allow_legacy_profile=args.allow_legacy_profile,
                force=args.force,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "plan":
            from .planning import save_pose_plan

            report = save_pose_plan(
                args.output,
                urdf=args.urdf,
                end_link=args.end_link,
                base_link=args.base_link,
                count=args.count,
                candidates=args.candidates,
                seed=args.seed,
                characteristic_length_m=args.characteristic_length_m,
                force=args.force,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "view":
            from .backends.viser import create_viewer

            if args.duration is not None and (
                args.duration <= 0 or not math.isfinite(args.duration)
            ):
                raise ValueError("duration must be finite and positive")
            server = create_viewer(args.experiment, host=args.host, port=args.port)
            try:
                deadline = (
                    None if args.duration is None else time.monotonic() + args.duration
                )
                while deadline is None or time.monotonic() < deadline:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                server.stop()
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"kincal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
