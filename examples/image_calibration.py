"""Render real PNG observations, detect the board, then calibrate robot geometry.

Run after ``pip install -e '.[vision]'``::

    python examples/image_calibration.py --output runs/images
"""

import argparse
import json
from pathlib import Path

from kinematic_calibration.image_simulation import ImageNoise, run_image_experiment
from kinematic_calibration.targets import get_board_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--board", choices=["opencv-acircle", "cctag-multi"], default="opencv-acircle"
    )
    parser.add_argument(
        "--measurement-mode", choices=["image", "external-centers"], default="image"
    )
    parser.add_argument("--rgb-std", type=float, default=0.5)
    parser.add_argument("--blur-sigma", type=float, default=0.3)
    parser.add_argument("--supersample", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_image_experiment(
        args.output,
        samples=args.samples,
        validation_samples=args.validation_samples,
        seed=args.seed,
        board=get_board_config(args.board),
        measurement_mode=args.measurement_mode,
        noise=ImageNoise(args.rgb_std, args.blur_sigma, args.supersample),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
