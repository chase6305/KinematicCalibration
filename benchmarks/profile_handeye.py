#!/usr/bin/env python3
"""Time complete public hand-eye CLI workflows with independent SE(3) truth.

    env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        .venv/bin/python benchmarks/profile_handeye.py --output /tmp/handeye.json

Influence is opt-in: --workflows influence --methods shah --repeats 1.
Imports, generation, one full warmup per case and driver checks are untimed.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from kinematic_calib import io_utils
from kinematic_calibration import handeye_cli
from kinematic_calibration.handeye import opencv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(rotvec, translation):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def write_observations(path, samples, seed):
    rng = np.random.default_rng(seed + samples)
    base_T_camera = pose([0.31, -0.43, 0.12], [0.54, -0.31, 0.26])
    ee_T_board = pose([-0.21, 0.13, 0.26], [0.03, -0.04, 0.12])
    # Direct ground-truth SE(3) poses; no robot model or nominal FK.
    base_T_ee = np.repeat(np.eye(4)[None], samples, axis=0)
    rotations = rng.normal(0, 0.6, (samples, 3))
    base_T_ee[:, :3, :3] = Rotation.from_rotvec(rotations).as_matrix()
    base_T_ee[:, :3, 3] = rng.uniform(-0.5, 0.5, (samples, 3))
    camera_T_board = np.linalg.inv(base_T_camera) @ base_T_ee @ ee_T_board
    noise = np.repeat(np.eye(4)[None], samples, axis=0)
    rotations = rng.normal(0, 1e-4, (samples, 3))
    noise[:, :3, :3] = Rotation.from_rotvec(rotations).as_matrix()
    noise[:, :3, 3] = rng.normal(0, 1e-4, (samples, 3))
    np.savez(
        path,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=base_T_ee,
        camera_T_board=camera_T_board @ noise,
        candidate_indices=np.arange(samples, dtype=np.int64),
        translation_candidate_count=np.asarray(samples // 2),
        rotation_candidate_count=np.asarray(samples - samples // 2),
    )


def measure_case(source, output, count, workflow, method, args):
    argv = [workflow, "--samples", str(source), "--output", str(output), "--force"]
    if method:
        argv += ["--method", method]
    if workflow == "bootstrap":
        argv += ["--replicates", str(args.replicates), "--seed", str(args.seed)]
    input_sha = sha256(source)

    def checked_result():
        result = json.loads(output.read_text())
        assert sha256(source) == input_sha, "input archive changed"
        assert result["observations_npz_sha256"] == input_sha
        assert result["samples"] == count
        if "candidate_indices" in result:
            assert result["candidate_indices"] == list(range(count))
        result.pop("provenance")
        return json.dumps(result, sort_keys=True, allow_nan=False)

    warmup_status = handeye_cli.main(argv)
    assert warmup_status == 0, "warmup failed"
    reference, runs = checked_result(), []
    for _ in range(args.repeats):
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        status = handeye_cli.main(argv)
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        assert status == 0, f"{workflow} failed"
        assert checked_result() == reference, "result changed"
        runs.append({"wall_seconds": wall, "cpu_seconds": cpu})
    return {
        "samples": count,
        "workflow": workflow,
        "method": method,
        "observations_npz_sha256": input_sha,
        "result_sha256_without_provenance": hashlib.sha256(
            reference.encode()
        ).hexdigest(),
        "inputs_unchanged": True,
        "results_exactly_stable": True,
        "warmup_calls": 1,
        "runs": runs,
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in runs),
        "median_cpu_seconds": statistics.median(row["cpu_seconds"] for row in runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 1000])
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=["compare", "bootstrap", "influence"],
        default=["compare", "bootstrap"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["shah", "li"],
        default=["shah", "li"],
        help="bootstrap/influence methods; compare always runs both",
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/handeye-profile.json")
    )
    args = parser.parse_args()
    if (
        min(args.samples) < (4 if "influence" in args.workflows else 3)
        or min(args.repeats, args.replicates) <= 0
        or args.seed < 0
    ):
        parser.error(
            "require N >= 3 (influence: 4), positive counts and non-negative seed"
        )
    cv.setNumThreads(1)
    logging.getLogger(handeye_cli.__name__).setLevel(logging.ERROR)
    sources = [__file__, opencv.__file__, handeye_cli.__file__, io_utils.__file__]
    source_hashes = {str(path): sha256(path) for path in sources}
    thread_keys = "OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONPATH".split()
    report = {
        "metadata": {
            "versions": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "opencv": cv.__version__,
            },
            "platform": platform.platform(),
            "source_sha256": source_hashes,
            "thread_environment": {key: os.environ.get(key) for key in thread_keys},
            "opencv_threads": cv.getNumThreads(),
            "timing_scope": "complete handeye_cli.main: loading, checks, solving and atomic writes; success logs suppressed; imports, generation, one full warmup per case and driver checks excluded",
            "truth": "independently specified SE(3) poses plus small camera noise; no nominal FK",
        },
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="kincal-handeye-benchmark-") as temporary:
        directory = Path(temporary)
        for samples in args.samples:
            source = directory / f"samples-{samples}.npz"
            write_observations(source, samples, args.seed)
            for workflow in args.workflows:
                for method in [None] if workflow == "compare" else args.methods:
                    case = measure_case(
                        source,
                        directory / "result.json",
                        samples,
                        workflow,
                        method,
                        args,
                    )
                    report["cases"].append(case)
                    print(
                        f"N={samples} {workflow}/{method or 'shah+li'}: "
                        f"{case['median_wall_seconds']:.6f}s wall, "
                        f"{case['median_cpu_seconds']:.6f}s CPU",
                        flush=True,
                    )
    assert source_hashes == {str(path): sha256(path) for path in sources}, (
        "source changed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
