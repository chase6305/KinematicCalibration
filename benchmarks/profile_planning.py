#!/usr/bin/env python3
"""Reproducible pose-selection and visit-order timings for the bundled UR10.

Set BLAS threads before running Python, for example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        benchmarks/profile_planning.py --cases 400:40 2000:80 5000:120 \
        --output outputs/planning-profile.json

For routing alone, including 6/7 DOF and implicit/open/closed starts::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        benchmarks/profile_planning.py --route-counts 40 200 1000 --repeats 5 \
        --output outputs/routing-profile.json

Imports, candidate generation, warmup and validation are outside timed calls.
Explicit starts use zero joint angles; distances are joint-space radians.
``--memory`` measures selection's peak traced allocations in a separate call,
outside the timings; candidate inputs are allocated before tracing starts.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import tracemalloc
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


def measure_case(backend, candidate_count, selected_count, seed, repeats, memory=False):
    limits = backend.joint_limits
    candidates = np.random.default_rng(seed).uniform(
        np.maximum(limits[:, 0], -np.pi),
        np.minimum(limits[:, 1], np.pi),
        (candidate_count, backend.dof),
    )
    start_configuration = np.zeros(backend.dof)
    warmup = planning.select_informative_poses(
        backend, candidates, count=selected_count
    )
    routing.plan_visit_order(warmup.joint_positions)
    runs = []
    for _ in range(repeats):
        started, cpu_started = time.perf_counter(), time.process_time()
        selected = planning.select_informative_poses(
            backend, candidates, count=selected_count
        )
        selection_seconds = time.perf_counter() - started
        selection_cpu_seconds = time.process_time() - cpu_started
        original_positions = selected.joint_positions.copy()
        started, cpu_started = time.perf_counter(), time.process_time()
        visits = routing.plan_visit_order(
            selected.joint_positions, start_configuration=start_configuration
        )
        routing_seconds = time.perf_counter() - started
        routing_cpu_seconds = time.process_time() - cpu_started
        np.testing.assert_array_equal(
            selected.candidate_indices, warmup.candidate_indices
        )
        np.testing.assert_array_equal(selected.singular_values, warmup.singular_values)
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
                "selection_cpu_seconds": selection_cpu_seconds,
                "routing_wall_seconds": routing_seconds,
                "routing_cpu_seconds": routing_cpu_seconds,
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
    result = {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "median_selection_wall_seconds": statistics.median(
            row["selection_wall_seconds"] for row in runs
        ),
        "median_selection_cpu_seconds": statistics.median(
            row["selection_cpu_seconds"] for row in runs
        ),
        "median_routing_wall_seconds": statistics.median(
            row["routing_wall_seconds"] for row in runs
        ),
        "median_routing_cpu_seconds": statistics.median(
            row["routing_cpu_seconds"] for row in runs
        ),
        "runs": runs,
    }
    if memory:
        tracemalloc.start()
        try:
            measured = planning.select_informative_poses(
                backend, candidates, count=selected_count
            )
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        np.testing.assert_array_equal(
            measured.candidate_indices, warmup.candidate_indices
        )
        np.testing.assert_array_equal(measured.singular_values, warmup.singular_values)
        result["selection_peak_traced_bytes"] = peak_bytes
    return result


def measure_route_case(route_count, dof, mode, seed, repeats):
    positions = np.random.default_rng(seed + route_count + dof).uniform(
        -3.0, 3.0, (route_count, dof)
    )
    original_positions = positions.copy()
    start = None if mode == "implicit_open" else np.zeros(dof)
    closed = mode == "explicit_closed"
    reference = routing.plan_visit_order(
        positions, start_configuration=start, return_to_start=closed
    )
    runs = []
    for _ in range(repeats):
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        visits = routing.plan_visit_order(
            positions, start_configuration=start, return_to_start=closed
        )
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        np.testing.assert_array_equal(positions, original_positions)
        np.testing.assert_array_equal(np.sort(visits.indices), np.arange(route_count))
        np.testing.assert_array_equal(visits.indices, reference.indices)
        np.testing.assert_array_equal(
            visits.segment_distances_rad, reference.segment_distances_rad
        )
        assert visits.total_distance_rad == reference.total_distance_rad
        path = positions[visits.indices]
        if start is not None:
            path = np.vstack([start, path])
            if closed:
                path = np.vstack([path, start])
        checked_segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
        np.testing.assert_allclose(visits.segment_distances_rad, checked_segments)
        assert visits.total_distance_rad <= visits.initial_distance_rad
        runs.append(
            {
                "routing_wall_seconds": wall_seconds,
                "routing_cpu_seconds": cpu_seconds,
                "initial_distance_rad": visits.initial_distance_rad,
                "total_distance_rad": visits.total_distance_rad,
                "visit_indices_sha256": hashlib.sha256(
                    visits.indices.tobytes()
                ).hexdigest(),
            }
        )
    return {
        "route_count": route_count,
        "dof": dof,
        "mode": mode,
        "median_routing_wall_seconds": statistics.median(
            row["routing_wall_seconds"] for row in runs
        ),
        "median_routing_cpu_seconds": statistics.median(
            row["routing_cpu_seconds"] for row in runs
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
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_average_at_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "timing_scope": "selection and routing separately; imports, warmup, candidate generation and validation excluded",
        "start_configuration_rad": [0.0] * 6,
        "return_to_start": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--cases", nargs="+", type=_case, default=[(400, 40), (2000, 80)]
    )
    modes.add_argument(
        "--route-counts",
        nargs="+",
        type=int,
        help="benchmark routing only for each N, with 6/7 DOF and all start modes",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--memory",
        action="store_true",
        help="measure selection's peak traced allocations in an extra untimed call",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0:
        parser.error("repeats must be positive and seed must be non-negative")
    if args.route_counts is not None and min(args.route_counts) < 1:
        parser.error("route counts must be positive")
    if args.memory and args.route_counts is not None:
        parser.error(
            "--memory measures selection; use --cases instead of --route-counts"
        )
    report = {
        "metadata": metadata(),
        "seed": args.seed,
        "repeats": args.repeats,
        "cases": [],
    }
    if args.route_counts is not None:
        report["metadata"].update(
            {
                "timing_scope": "routing only; imports, warmup, input generation and validation excluded",
                "joint_positions_rad": "uniform[-3,3], seed + route_count + dof",
                "start_configuration_rad": "none for implicit_open; zero vector otherwise",
                "return_to_start": "true for explicit_closed only",
                "load_average_at_start": os.getloadavg()
                if hasattr(os, "getloadavg")
                else None,
            }
        )
        cases = (
            measure_route_case(route_count, dof, mode, args.seed, args.repeats)
            for route_count in args.route_counts
            for dof in (6, 7)
            for mode in ("implicit_open", "explicit_open", "explicit_closed")
        )
    else:
        report["metadata"]["memory_scope"] = (
            (
                "peak tracemalloc allocations in a separate full selection call; "
                "candidate inputs excluded; not process RSS"
            )
            if args.memory
            else None
        )
        backend = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
        cases = (
            measure_case(
                backend,
                candidate_count,
                selected_count,
                args.seed,
                args.repeats,
                memory=args.memory,
            )
            for candidate_count, selected_count in args.cases
        )
    for case in cases:
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
