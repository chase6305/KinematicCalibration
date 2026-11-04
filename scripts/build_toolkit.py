"""Build the pinned optional Toolkit without installing into a user environment.

Requires Conan 2.9+, CMake 3.23+, a C++17 compiler, and a configured Conan profile.
Run `python scripts/build_toolkit.py --help` for paths and offline-cache mode.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(command, *, env=None):
    print("+ " + " ".join(str(arg) for arg in command), flush=True)
    subprocess.run([str(arg) for arg in command], check=True, env=env)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--conan", default="conan")
    parser.add_argument(
        "--build-dir", type=Path, default=root / "outputs/native-toolkit"
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--offline", action="store_true", help="Use cached native dependencies only"
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    cmake = shutil.which(args.cmake)
    conan = shutil.which(args.conan)
    if not cmake or not conan:
        parser.error(
            "CMake and Conan must be installed; choose their paths with --cmake / --conan"
        )
    version_text = subprocess.check_output([cmake, "--version"], text=True)
    match = re.search(r"cmake version (\d+)\.(\d+)", version_text)
    if not match or tuple(map(int, match.groups())) < (3, 23):
        parser.error("Toolkit requires CMake 3.23 or newer")
    source = root / "third_party/HandEyeCalibrationToolkit"
    if not (source / "conanfile.py").is_file():
        parser.error(
            "initialize the Toolkit first: git submodule update --init --recursive"
        )
    target_python = args.python.absolute()
    subprocess.run(
        [str(target_python), "-c", "import sys; print(sys.version)"], check=True
    )
    build = args.build_dir.absolute()
    if build == source or source in build.parents:
        parser.error("build-dir must be outside the submodule source tree")
    environment = dict(
        os.environ, HANDEYE_CALIBRATION_PYTHON_EXECUTABLE=str(target_python)
    )
    install = [
        conan,
        "install",
        source,
        f"--output-folder={build}",
        "--build=never" if args.offline else "--build=missing",
        "-o",
        "&:with_python=True",
        "-o",
        f"&:with_tests={not args.skip_tests}",
        "-c",
        "tools.cmake.cmaketoolchain:user_presets=",
    ]
    if args.offline:
        install.append("--no-remote")
    run(install, env=environment)
    binary = build / "build/Release"
    toolchain = binary / "generators/conan_toolchain.cmake"
    run(
        [
            cmake,
            "-S",
            source,
            "-B",
            binary,
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        env=environment,
    )
    run([cmake, "--build", binary, "-j", args.jobs], env=environment)
    runtime = binary / "generators/conanrun.sh"
    if not args.skip_tests:
        ctest = Path(cmake).with_name("ctest")
        if not ctest.is_file():
            parser.error("ctest must be installed beside the chosen CMake executable")
        run(
            [
                "bash",
                "-c",
                'source "$1"; shift; exec "$@"',
                "kincal-native",
                runtime,
                ctest,
                "--test-dir",
                binary,
                "--output-on-failure",
            ],
            env=environment,
        )
    prefix = build / "install"
    run([cmake, "--install", binary, "--prefix", prefix], env=environment)
    commit = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", source, "status", "--porcelain"], text=True
        )
    )
    report = {
        "source_commit": commit,
        "source_modified": dirty,
        "python": str(target_python),
        "cmake": cmake,
        "install_prefix": str(prefix),
        "runtime_script": str(runtime),
        "tests_executed": not args.skip_tests,
    }
    (build / "native-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Toolkit build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
