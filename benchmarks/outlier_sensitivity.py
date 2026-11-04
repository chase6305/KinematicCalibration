"""Measure held-out calibration accuracy under corrupted training poses.

Run: python benchmarks/outlier_sensitivity.py --output outputs/outliers.json
This injects errors only into training observations. Validation uses untouched
independent joint configurations and exact truth poses.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset
from kinematic_calib.service import fit_dataset
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf, pose_metrics, predict_poe
from kinematic_calibration.simulation import MeasurementNoise, sample_measurements
from robotkinecal import SerialRobotKineCal


def measure(*, samples=100, validation_samples=40, seed=42):
    robot = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = robot.perturbed(seed=seed + 1)
    train = sample_measurements(truth, count=samples, seed=seed)
    validation = sample_measurements(
        truth, count=validation_samples, seed=seed + 2, noise=MeasurementNoise(0, 0)
    )
    rows = []
    for fraction in (0, 0.01, 0.05, 0.1):
        observed = train.observed_poses.copy()
        rng = np.random.default_rng(seed + 3)
        count = int(fraction * len(observed))
        indices = rng.choice(len(observed), count, replace=False)
        observed[indices, :3, 3] += rng.normal(0, 0.03, (count, 3))
        observed[indices, :3, :3] = (
            observed[indices, :3, :3]
            @ Rotation.from_rotvec(rng.normal(0, 0.05, (count, 3))).as_matrix()
        )
        data = KinematicDataset(
            train.joint_positions, observed, train.observation_frame
        )
        axes, home = robot.poe_parameters()
        solver = SerialRobotKineCal.from_screw_axes(axes, home)
        started = time.perf_counter()
        fit = fit_dataset(data, solver)
        rows.append(
            {
                "outlier_fraction": fraction,
                "corrupted_sample_indices": indices.tolist(),
                "seconds": time.perf_counter() - started,
                "iterations": len(fit.result.iteration_results),
                "termination": fit.result.termination_reason,
                **pose_metrics(
                    predict_poe(
                        fit.screw_axes, fit.zero_pose.A, validation.joint_positions
                    ),
                    validation.observed_poses,
                ),
            }
        )
    return {
        "seed": seed,
        "training_samples": samples,
        "validation_samples": validation_samples,
        "measurement_model": "synthetic_pose_sensor",
        "outlier_position_std_m": 0.03,
        "outlier_orientation_std_rad": 0.05,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = measure(
        samples=args.samples, validation_samples=args.validation_samples, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
