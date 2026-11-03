#!/usr/bin/env python3
"""Deterministic solver/regressor profiles with independent synthetic truth.

Example (set BLAS threads before Python imports NumPy)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
        benchmarks/profile_solver.py --output outputs/solver-profile.json

Input construction, observation loading, profiling, and held-out validation are
outside measured solve intervals. The reference solver is never modified here.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from robotkinecal import SerialRobotKineCal


def independent_poe(axes, home, configurations):
    """Revolute POE using closed-form rigid rotations, independent of Twist3."""
    result = np.repeat(np.eye(4)[None], len(configurations), axis=0)
    for axis, angles in zip(axes, configurations.T, strict=True):
        angular = axis[3:]
        point = np.cross(angular, axis[:3])
        rotations = Rotation.from_rotvec(angles[:, None] * angular).as_matrix()
        transforms = np.repeat(np.eye(4)[None], len(configurations), axis=0)
        transforms[:, :3, :3] = rotations
        transforms[:, :3, 3] = point - np.einsum("nij,j->ni", rotations, point)
        result = result @ transforms
    return result @ home


def problem(dof, samples, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, dof]))
    angular = rng.normal(size=(dof, 3))
    angular /= np.linalg.norm(angular, axis=1)[:, None]
    points = rng.uniform(-0.4, 0.4, size=(dof, 3))
    nominal_axes = np.column_stack((np.cross(points, angular), angular))
    home = np.eye(4)
    home[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
    home[:3, 3] = [0.5, 0.1, 0.6]
    truth_axes = []
    for axis in nominal_axes:
        rotation = Rotation.from_rotvec(rng.normal(scale=0.002, size=3)).as_matrix()
        translation = rng.normal(scale=0.002, size=3)
        direction = rotation @ axis[3:]
        truth_axes.append(
            np.r_[rotation @ axis[:3] + np.cross(translation, direction), direction]
        )
    truth_axes = np.asarray(truth_axes)
    perturbation = np.eye(4)
    perturbation[:3, :3] = Rotation.from_rotvec(
        rng.normal(scale=0.002, size=3)
    ).as_matrix()
    perturbation[:3, 3] = rng.normal(scale=0.002, size=3)
    truth_home = perturbation @ home
    # One common population makes the 100/500/2000 cases nested prefixes.
    population = rng.uniform(-2.0, 2.0, size=(max(2000, samples), dof))
    held_out = rng.uniform(-2.0, 2.0, size=(80, dof))
    joints = population[:samples]
    truth = independent_poe(truth_axes, truth_home, joints)
    observations = truth.copy()
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, dof, 1]))
    observations[:, :3, 3] += noise_rng.normal(scale=5e-5, size=(samples, 3))
    observations[:, :3, :3] = (
        Rotation.from_rotvec(
            noise_rng.normal(scale=5e-5, size=(samples, 3))
        ).as_matrix()
        @ observations[:, :3, :3]
    )
    return (
        nominal_axes,
        home,
        joints,
        observations,
        held_out,
        independent_poe(truth_axes, truth_home, held_out),
    )


def make_calibrator(inputs):
    axes, home, joints, observations, _, _ = inputs
    calibrator = SerialRobotKineCal.from_screw_axes(axes, home)
    calibrator.set_observations(joints, [SE3(pose) for pose in observations])
    return calibrator


def validate(calibrator, inputs):
    held_out, measured = inputs[4:]
    predictions = np.asarray(
        [calibrator.forward_kinematics(calibrator.N_JOINTS + 1, q).A for q in held_out]
    )
    differences = predictions[:, :3, 3] - measured[:, :3, 3]
    relative_rotations = predictions[:, :3, :3].transpose(0, 2, 1) @ measured[:, :3, :3]
    return {
        "held_out_samples": len(held_out),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(differences**2, axis=1)))),
        "orientation_rmse_rad": float(
            np.sqrt(np.mean(Rotation.from_matrix(relative_rotations).magnitude() ** 2))
        ),
    }


def profile_rows(profiler, limit=20):
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, function), (
        primitive_calls,
        total_calls,
        own_time,
        cumulative,
        _,
    ) in stats.stats.items():
        entries.append(
            {
                "function": f"{Path(filename).name}:{line}:{function}",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": own_time,
                "cumulative_seconds": cumulative,
            }
        )
    return {
        "total_profiled_seconds": stats.total_tt,
        "by_cumulative": sorted(
            entries, key=lambda row: row["cumulative_seconds"], reverse=True
        )[:limit],
        "by_self": sorted(entries, key=lambda row: row["self_seconds"], reverse=True)[
            :limit
        ],
    }


def measure_case(dof, samples, seed, repeats, max_iterations):
    inputs = problem(dof, samples, seed)
    rows = []
    regressor_times = []
    batch_regressor_times = []
    for _ in range(repeats):
        calibrator = make_calibrator(inputs)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        result = calibrator.solve(max_iterations=max_iterations)
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        final = result.iteration_results[-1]
        rows.append(
            {
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "iterations": result.nb_iterations_executed,
                "termination_reason": result.termination_reason,
                "matrix_rank": int(final.matrix_rank),
                "parameter_count": int(final.N_PARAMS),
                **validate(calibrator, inputs),
            }
        )
        # Build the exact regressor blocks used by planning, without data
        # generation, candidate scoring, or another solve in this interval.
        planning_calibrator = make_calibrator(inputs)
        regressor_start = time.perf_counter()
        blocks = np.asarray([planning_calibrator.A_Matrix(q) for q in inputs[2]])
        regressor_times.append(time.perf_counter() - regressor_start)
        assert blocks.shape == (samples, 6, 4 * dof + 6)
        if hasattr(planning_calibrator, "regressor_blocks"):
            batch_regressor_start = time.perf_counter()
            batch_blocks = planning_calibrator.regressor_blocks(inputs[2])
            batch_regressor_times.append(time.perf_counter() - batch_regressor_start)
            np.testing.assert_allclose(batch_blocks, blocks, atol=2e-12)
    return {
        "dof": dof,
        "samples": samples,
        "runs": rows,
        "median_solve_wall_seconds": statistics.median(
            row["wall_seconds"] for row in rows
        ),
        "median_solve_cpu_seconds": statistics.median(
            row["cpu_seconds"] for row in rows
        ),
        "regressor_wall_seconds": regressor_times,
        "median_regressor_wall_seconds": statistics.median(regressor_times),
        "batch_regressor_wall_seconds": batch_regressor_times,
        "median_batch_regressor_wall_seconds": statistics.median(batch_regressor_times)
        if batch_regressor_times
        else None,
    }


def metadata():
    import scipy
    import spatialmath

    core_path = Path(sys.modules[SerialRobotKineCal.__module__].__file__)
    batch_path = core_path.parent / "_batch.py"

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "spatialmath": getattr(spatialmath, "__version__", "unknown"),
        "core_source_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "batch_source_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest()
        if batch_path.exists()
        else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "blas_threads_env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "truth_evaluator": "independent closed-form rigid rotations, not SpatialMath Twist3",
        "timing_scope": "solve only; synthetic data, set_observations, held-out evaluation excluded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 500, 2000])
    parser.add_argument("--dof", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--profile-samples", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/solver-profile.json")
    )
    args = parser.parse_args()
    if (
        min(
            args.samples
            + args.dof
            + [args.repeats, args.max_iterations, args.profile_samples]
        )
        <= 0
    ):
        parser.error("counts must be positive")
    report = {
        "metadata": metadata(),
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
    }
    for dof in args.dof:
        for samples in args.samples:
            case = measure_case(
                dof, samples, args.seed, args.repeats, args.max_iterations
            )
            report["cases"].append(case)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in case.items()
                        if key
                        not in {
                            "runs",
                            "regressor_wall_seconds",
                            "batch_regressor_wall_seconds",
                        }
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    inputs = problem(max(args.dof), args.profile_samples, args.seed)
    calibrator = make_calibrator(inputs)
    profiler = cProfile.Profile()
    profiler.enable()
    calibrator.solve(max_iterations=args.max_iterations)
    profiler.disable()
    report["solve_profile"] = profile_rows(profiler)
    planning_calibrator = make_calibrator(inputs)
    planning_profiler = cProfile.Profile()
    planning_profiler.enable()
    _ = np.asarray([planning_calibrator.A_Matrix(q) for q in inputs[2]])
    planning_profiler.disable()
    report["regressor_profile"] = profile_rows(planning_profiler)
    if hasattr(planning_calibrator, "regressor_blocks"):
        batch_planning_profiler = cProfile.Profile()
        batch_planning_profiler.enable()
        _ = planning_calibrator.regressor_blocks(inputs[2])
        batch_planning_profiler.disable()
        report["batch_regressor_profile"] = profile_rows(batch_planning_profiler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
