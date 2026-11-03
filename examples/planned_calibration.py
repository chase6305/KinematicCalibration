"""Select and visit calibration poses, then fit independent synthetic geometry.

Run with the base dependencies only::

    python examples/planned_calibration.py --plan outputs/planned-calibration.npz

The perturbed model supplies synthetic measurements; the nominal model supplies
only planning regressors and fit initialization. The result directory can be
replayed with ``kincal view <directory> --split training``.
"""

import argparse
import json
from pathlib import Path

from kinematic_calibration.experiment import run_experiment
from kinematic_calibration.planning import save_pose_plan
from kinematic_calibration.simulation import MeasurementNoise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    plan = save_pose_plan(
        args.plan, count=args.count, candidates=args.candidates, seed=args.seed
    )
    output = args.output or args.plan.parent / f"{args.plan.stem}-experiment"
    report = run_experiment(
        output,
        plan=args.plan,
        validation_samples=20,
        seed=args.seed,
        noise=MeasurementNoise(0, 0),
    )
    print(
        json.dumps(
            {
                "synthetic_only": True,
                "experiment": str(output),
                "training_poses": report["training_samples"],
                "independent_validation_poses": report["validation_samples"],
                "converged": report["converged"],
                "visit_distance_rad": plan["total_distance_rad"],
                "selection_order_distance_rad": plan["selection_order_distance_rad"],
                "nominal_position_rmse_m": report["nominal_vs_measurements"][
                    "position_rmse_m"
                ],
                "calibrated_position_rmse_m": report["calibrated_vs_measurements"][
                    "position_rmse_m"
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
