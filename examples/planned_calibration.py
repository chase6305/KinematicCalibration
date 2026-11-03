"""Select and visit calibration poses, then fit independent synthetic geometry.

Run with the base dependencies only::

    python examples/planned_calibration.py --plan outputs/planned-calibration.npz

The perturbed model supplies synthetic measurements; the nominal model supplies
only planning regressors and fit initialization. No robot motion is commanded.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kinematic_calib.dataset import KinematicDataset, ObservationFrame
from kinematic_calib.service import FitOptions, fit_dataset
from kinematic_calib.validation import evaluate_saved_poe
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from kinematic_calibration.planning import save_pose_plan
from kinematic_calibration.simulation import MeasurementNoise, sample_measurements
from robotkinecal import SerialRobotKineCal


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    plan = save_pose_plan(
        args.plan, count=args.count, candidates=args.candidates, seed=args.seed
    )
    with np.load(args.plan, allow_pickle=False) as archive:
        configurations = archive["planned_joint_positions"]
        sample_ids = archive["candidate_indices"]

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=args.seed + 1)
    training = KinematicDataset(
        configurations,
        np.asarray([truth.forward(row) for row in configurations]),
        ObservationFrame.BASE_T_EE,
        sample_ids,
        nominal.joint_names,
    )
    validation = sample_measurements(
        truth, count=20, seed=args.seed + 2, noise=MeasurementNoise(0, 0)
    )
    validation_ids = np.arange(args.candidates, args.candidates + 20)
    nominal_axes, nominal_home = nominal.poe_parameters()
    calibrator = SerialRobotKineCal.from_screw_axes(
        nominal_axes, nominal_home, joint_names=nominal.joint_names
    )
    fitted = fit_dataset(training, calibrator, options=FitOptions(max_iterations=30))
    evaluations = [
        evaluate_saved_poe(
            axes,
            home,
            validation.joint_positions,
            validation.observed_poses,
            validation_ids,
        )
        for axes, home in (
            (nominal_axes, nominal_home),
            (fitted.screw_axes, fitted.zero_pose.A),
        )
    ]
    print(
        json.dumps(
            {
                "synthetic_only": True,
                "training_poses": len(configurations),
                "independent_validation_poses": len(validation.joint_positions),
                "converged": fitted.result.has_converged,
                "visit_distance_rad": plan["total_distance_rad"],
                "selection_order_distance_rad": plan["selection_order_distance_rad"],
                "nominal_position_rmse_m": evaluations[0]["position_error_rms_m"],
                "calibrated_position_rmse_m": evaluations[1]["position_error_rms_m"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if fitted.result.has_converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
