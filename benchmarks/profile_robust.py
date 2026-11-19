#!/usr/bin/env python3
"""Measure a complete Huber IRLS fit and Linux process peak RSS.

Run each measurement in a fresh process. Set BLAS/OMP threads before Python
starts, and follow the shared-machine resource and exclusive-lock convention.
Input generation and independent holdout evaluation are outside fit timing.
Peak RSS includes imports and input generation, up to completion of the fit.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from kinematic_calib import (
    FitOptions,
    HuberOptions,
    KinematicDataset,
    fit_dataset,
    robust,
    service,
)
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf, pose_metrics, predict_poe
from kinematic_calibration.simulation import MeasurementNoise, sample_measurements
from robotkinecal import SerialRobotKineCal, calibration


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def measure(samples, seed):
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=seed + 1)
    training = sample_measurements(truth, count=samples, seed=seed)
    validation = sample_measurements(
        truth, count=100, seed=seed + 2, noise=MeasurementNoise(0, 0)
    )
    observed = training.observed_poses.copy()
    rng = np.random.default_rng(seed + 3)
    corrupted = rng.choice(samples, samples // 20, replace=False)
    observed[corrupted, :3, 3] += rng.normal(0, 0.03, (len(corrupted), 3))
    if len(corrupted):
        observed[corrupted, :3, :3] = (
            observed[corrupted, :3, :3]
            @ Rotation.from_rotvec(rng.normal(0, 0.05, (len(corrupted), 3))).as_matrix()
        )
    dataset = KinematicDataset(
        training.joint_positions,
        observed,
        training.observation_frame,
        training.sample_indices + 1000,
        training.joint_names,
    )
    input_fingerprint = fingerprint(
        dataset.joint_positions, dataset.observed_poses, dataset.sample_indices
    )
    axes, home = nominal.poe_parameters()

    def factory():
        return SerialRobotKineCal.from_screw_axes(
            axes, home, joint_names=nominal.joint_names
        )

    gc.collect()
    wall, cpu = time.perf_counter(), time.process_time()
    fit = fit_dataset(
        dataset,
        calibrator_factory=factory,
        options=FitOptions(robust=HuberOptions(0.0001, 0.0002)),
    )
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    # Read the high-water mark before reporting/fingerprinting allocates memory.
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (
        fingerprint(
            dataset.joint_positions, dataset.observed_poses, dataset.sample_indices
        )
        == input_fingerprint
    ), "input data changed"
    diagnostics = fit.robust_diagnostics
    diagnostic_bytes = json.dumps(diagnostics, sort_keys=True, allow_nan=False).encode()
    return {
        "samples": samples,
        "seed": seed,
        "corrupted_samples": len(corrupted),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "process_max_rss_kib_at_fit_completion": peak_rss_kib,
        "outer_rounds": len(diagnostics["iterations"]),
        "inner_iterations": [
            row["inner_iterations"] for row in diagnostics["iterations"]
        ],
        "outer_termination": diagnostics["outer_termination_reason"],
        "inner_termination": fit.result.termination_reason,
        "inputs_sha256": input_fingerprint,
        "inputs_unchanged": True,
        "fitted_parameters_sha256": fingerprint(fit.screw_axes, fit.zero_pose.A),
        "robust_diagnostics_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "independent_holdout_metrics": pose_metrics(
            predict_poe(fit.screw_axes, fit.zero_pose.A, validation.joint_positions),
            validation.observed_poses,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 6 or args.seed < 0:
        parser.error("require at least 6 samples and a non-negative seed")
    if not sys.platform.startswith("linux"):
        parser.error("peak RSS units in this benchmark are defined for Linux")
    metadata = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_at_start": os.getloadavg(),
        "source_sha256": {
            str(Path(module.__file__).resolve()): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (robust, service, calibration)
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "timing_scope": "complete in-memory Huber IRLS fit, including observation loading; input generation, GC and holdout evaluation excluded",
        "memory_scope": "fresh Linux worker process peak RSS up to fit completion; includes imports and input generation, excludes later reporting",
        "measurement_model": "synthetic pose sensor from independently perturbed truth; 5% training contamination; untouched independent holdout",
    }
    report = {"metadata": metadata, "result": measure(args.samples, args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["result"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
