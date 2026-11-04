#!/usr/bin/env python3
"""Time complete saved-POE validation against independent synthetic observations.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        benchmarks/profile_validation.py --repeats 3 --profile --output /tmp/validation.json

Use this same driver with another checkout on PYTHONPATH to compare revisions.
Truth construction, result checks, and optional profiling are outside timing.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import numpy as np
from profile_solver import metadata, problem, profile_rows

from kinematic_calib.validation import evaluate_saved_poe


def validation_inputs(dof, samples, seed):
    axes, home, joints, observed, _, _ = problem(dof, samples, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, dof, samples, 2]))
    source_indices = rng.permutation(np.arange(samples, dtype=np.int64) * 3 + 101)
    return axes, home, joints, observed, source_indices


def check_result(result, joints, source_indices):
    rows = result["sample_errors"]
    assert result["samples"] == len(joints)
    np.testing.assert_array_equal(
        [row["sample_index"] for row in rows], np.arange(len(joints))
    )
    np.testing.assert_array_equal(
        [row["source_sample_index"] for row in rows], source_indices
    )
    np.testing.assert_array_equal([row["joint_positions_rad"] for row in rows], joints)
    for quantity, unit in (("position", "m"), ("orientation", "rad")):
        index = result[f"{quantity}_error_worst_sample_index"]
        assert index == np.argmax([row[f"{quantity}_error_{unit}"] for row in rows])
        assert (
            result[f"{quantity}_error_worst_source_sample_index"]
            == source_indices[index]
        )


def measure_case(dof, samples, seed, repeats):
    inputs = validation_inputs(dof, samples, seed)
    runs = []
    for _ in range(repeats):
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        result = evaluate_saved_poe(*inputs)
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        check_result(result, inputs[2], inputs[4])
        metrics = {
            key: value for key, value in result.items() if key != "sample_errors"
        }
        runs.append({"wall_seconds": wall, "cpu_seconds": cpu, "metrics": metrics})
    return {
        "dof": dof,
        "samples": samples,
        "sample_ids_verified_each_run": True,
        "inputs_sha256": hashlib.sha256(
            b"".join(array.tobytes() for array in inputs)
        ).hexdigest(),
        "source_sample_indices": inputs[4].tolist(),
        "runs": runs,
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in runs),
        "median_cpu_seconds": statistics.median(row["cpu_seconds"] for row in runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--dof", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--profile", action="store_true", help="profile one extra largest case"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/validation-profile.json")
    )
    args = parser.parse_args()
    if min(args.samples + args.dof + [args.repeats]) <= 0 or args.seed < 0:
        parser.error("counts must be positive and seed must be non-negative")
    environment = metadata()
    for name, source in (
        ("validation", inspect.getsourcefile(evaluate_saved_poe)),
        ("driver", __file__),
        ("generator", inspect.getsourcefile(problem)),
    ):
        environment[f"{name}_source_sha256"] = hashlib.sha256(
            Path(source).read_bytes()
        ).hexdigest()
    environment["timing_scope"] = (
        "complete evaluate_saved_poe, including validation and result construction; "
        "input construction, result checks and cProfile excluded"
    )
    report = {
        "metadata": environment,
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
    }
    for dof in args.dof:
        for samples in args.samples:
            case = measure_case(dof, samples, args.seed, args.repeats)
            report["cases"].append(case)
            print(
                json.dumps(
                    {
                        key: case[key]
                        for key in (
                            "dof",
                            "samples",
                            "median_wall_seconds",
                            "median_cpu_seconds",
                        )
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    if args.profile:
        dof, samples = max(args.dof), max(args.samples)
        inputs = validation_inputs(dof, samples, args.seed)
        profiler = cProfile.Profile()
        result = profiler.runcall(evaluate_saved_poe, *inputs)
        check_result(result, inputs[2], inputs[4])
        report["validation_profile"] = {
            "dof": dof,
            "samples": samples,
            **profile_rows(profiler),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
