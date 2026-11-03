#!/usr/bin/env python3
"""Reproducible pose-selection and visit-order timings for the bundled UR10.

Set BLAS threads before running Python, for example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        benchmarks/profile_planning.py --cases 400:40 2000:80 5000:120 \
        --output outputs/planning-profile.json

Imports, candidate generation, warmup and validation are outside timed calls.
The route starts at zero joint angles; distances are joint-space radians.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from kinematic_calibration import planning, routing
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf


def _case(text):
    try:
        candidates, selected = map(int, text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use CANDIDATES:SELECTED, e.g. 400:40"
        ) from exc
    if not 1 <= selected <= candidates:
        raise argparse.ArgumentTypeError("require 1 <= SELECTED <= CANDIDATES")
    return candidates, selected


def measure_case(backend, candidate_count, selected_count, seed, repeats):
    limits = backend.joint_limits
    candidates = np.random.default_rng(seed).uniform(
        np.maximum(limits[:, 0], -np.pi),
        np.minimum(limits[:, 1], np.pi),
        (candidate_count, backend.dof),
    )
    start_configuration = np.zeros(backend.dof)
    warmup = planning.select_informative_poses(backend, candidates[:1], count=1)
    routing.plan_visit_order(warmup.joint_positions)
    runs = []
    for _ in range(repeats):
        started = time.perf_counter()
        selected = planning.select_informative_poses(
            backend, candidates, count=selected_count
        )
        selection_seconds = time.perf_counter() - started
        original_positions = selected.joint_positions.copy()
        started = time.perf_counter()
        visits = routing.plan_visit_order(
            selected.joint_positions, start_configuration=start_configuration
        )
        routing_seconds = time.perf_counter() - started
        visited_candidates = selected.candidate_indices[visits.indices]
        np.testing.assert_array_equal(selected.joint_positions, original_positions)
        np.testing.assert_array_equal(
            np.sort(visits.indices), np.arange(selected_count)
        )
        np.testing.assert_array_equal(
            np.sort(visited_candidates), np.sort(selected.candidate_indices)
        )
        assert len(np.unique(visited_candidates)) == selected_count
        path = np.vstack(
            [start_configuration, selected.joint_positions[visits.indices]]
        )
        checked_distance = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        np.testing.assert_allclose(visits.total_distance_rad, checked_distance)
        assert visits.total_distance_rad <= visits.initial_distance_rad
        runs.append(
            {
                "selection_wall_seconds": selection_seconds,
                "routing_wall_seconds": routing_seconds,
                "initial_distance_rad": visits.initial_distance_rad,
                "total_distance_rad": visits.total_distance_rad,
                "rank": selected.rank,
                "parameter_count": selected.parameter_count,
                "condition_number": selected.condition_number,
                "selection_candidate_indices": selected.candidate_indices.tolist(),
                "visit_candidate_indices": visited_candidates.tolist(),
                "unique_visits": True,
                "information_set_unchanged": True,
            }
        )
    return {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "median_selection_wall_seconds": statistics.median(
            row["selection_wall_seconds"] for row in runs
        ),
        "median_routing_wall_seconds": statistics.median(
            row["routing_wall_seconds"] for row in runs
        ),
        "runs": runs,
    }


def metadata():
    import robotkinecal.calibration as core

    sources = [Path(module.__file__) for module in (planning, routing, core)]
    sources += [
        sources[-1].with_name("_batch.py"),
        Path(planning.__file__).with_name("backends") / "numpy.py",
        example_urdf(),
        Path(__file__),
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name) for name in ("numpy", "scipy", "spatialmath-python")
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "blas_threads_env": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "timing_scope": "selection and routing separately; imports, warmup, candidate generation and validation excluded",
        "start_configuration_rad": [0.0] * 6,
        "return_to_start": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", type=_case, default=[(400, 40), (2000, 80)]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0:
        parser.error("repeats must be positive and seed must be non-negative")
    backend = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    report = {
        "metadata": metadata(),
        "seed": args.seed,
        "repeats": args.repeats,
        "cases": [],
    }
    for candidate_count, selected_count in args.cases:
        case = measure_case(
            backend, candidate_count, selected_count, args.seed, args.repeats
        )
        report["cases"].append(case)
        summary = {key: value for key, value in case.items() if key != "runs"}
        summary.update(
            {
                key: case["runs"][0][key]
                for key in ("initial_distance_rad", "total_distance_rad")
            }
        )
        print(json.dumps(summary, allow_nan=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
