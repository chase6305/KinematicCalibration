"""Reusable metric calibration targets, independent of a simulation engine.

The six-marker target is ported from ``auto_calibrate_sim``. Grid spacing is
expressed as the *physical* X and row pitches. The toolkit's legacy detector
uses twice the physical row pitch as its second spacing argument.
"""

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path

import numpy as np


class BoardType(str, Enum):
    CCTAG_SINGLE = "cctag-single"
    CCTAG_MULTI = "cctag-multi"
    OPENCV_ACIRCLE = "opencv-acircle"


@dataclass(frozen=True)
class BoardConfig:
    """A measured board with an origin at its first detected control point.

    X points right across the printed image and Y points down. Lengths are
    metres. Legacy CCTag artwork is retained as a reference and is not assumed
    compatible with every detector ring profile. A single marker provides an
    origin measurement and needs depth; its in-plane rotation is unobservable.
    """

    board_type: BoardType
    size_x: int
    size_y: int
    pitch_x_m: float
    row_pitch_m: float
    width_m: float
    height_m: float
    asset_stem: str
    origin_from_top_left_m: tuple[float, float]
    thickness_m: float = 0.001
    legacy_artwork: bool = True

    def __post_init__(self):
        object.__setattr__(self, "board_type", BoardType(self.board_type))
        for name in ("size_x", "size_y"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        lengths = (
            self.pitch_x_m,
            self.row_pitch_m,
            self.width_m,
            self.height_m,
            self.thickness_m,
        )
        if not all(np.isfinite(v) and v > 0 for v in lengths):
            raise ValueError(
                "board dimensions and pitches must be positive finite metres"
            )
        origin = np.asarray(self.origin_from_top_left_m, dtype=float)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_from_top_left_m must contain two finite lengths")

    @property
    def points(self) -> np.ndarray:
        """Row-major asymmetric-grid points in the detected board frame."""
        return np.asarray(
            [
                [
                    (column + 0.5 * (row % 2)) * self.pitch_x_m,
                    row * self.row_pitch_m,
                    0.0,
                ]
                for row in range(self.size_y)
                for column in range(self.size_x)
            ],
            dtype=float,
        )

    @property
    def toolkit_size(self) -> tuple[int, int]:
        return self.size_x, self.size_y

    @property
    def toolkit_spacing(self) -> tuple[float, float]:
        """Detector spacing: X pitch and twice the physical row pitch."""
        return self.pitch_x_m, 2.0 * self.row_pitch_m

    @property
    def size(self) -> tuple[int, int]:
        return self.toolkit_size

    @property
    def asset_path(self) -> Path:
        """Legacy OBJ path; standard wheel installs unpack these assets."""
        return self.asset("obj")

    def asset(self, extension: str = "svg") -> Path:
        if extension not in ("svg", "png", "obj", "mtl"):
            raise ValueError("unsupported board asset extension")
        return Path(
            str(
                files("kinematic_calibration").joinpath(
                    "assets", "boards", f"{self.asset_stem}.{extension}"
                )
            )
        )

    @property
    def board_T_mesh(self) -> np.ndarray:
        """Map centered legacy OBJ coordinates into the detected board frame.

        The OBJ has Y up and its printed face at positive Z. The detector
        frame has Y down and Z into the printed surface.
        """
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[:3, 3] = [
            self.width_m / 2 - self.origin_from_top_left_m[0],
            self.height_m / 2 - self.origin_from_top_left_m[1],
            self.thickness_m,
        ]
        if self.board_type == BoardType.OPENCV_ACIRCLE:
            transform[:3, 3] = [0.0, 0.20, self.thickness_m]
        return transform

    @property
    def corners(self) -> np.ndarray:
        """Printed face corners in board coordinates, suitable for previews."""
        ox, oy = self.origin_from_top_left_m
        return np.array(
            [
                [0, 0, 0],
                [self.width_m, 0, 0],
                [self.width_m, self.height_m, 0],
                [0, self.height_m, 0],
            ]
        ) - [ox, oy, 0]

    def mesh(self):
        """Load a textured mesh lazily; requires the visualization extra."""
        import trimesh

        mesh = trimesh.load(self.asset_path, force="mesh", process=False)
        mesh.apply_transform(self.board_T_mesh)
        return mesh


BOARD_CONFIGS = {
    BoardType.CCTAG_MULTI: BoardConfig(
        BoardType.CCTAG_MULTI,
        2,
        3,
        0.13,
        0.065,
        0.3,
        0.3,
        "A4-300_300_3",
        (0.052, 0.052),
    ),
    BoardType.CCTAG_SINGLE: BoardConfig(
        BoardType.CCTAG_SINGLE,
        1,
        1,
        0.12,
        0.12,
        0.12,
        0.12,
        "cctag00000",
        (0.06, 0.06),
        thickness_m=0.0005,
    ),
    BoardType.OPENCV_ACIRCLE: BoardConfig(
        BoardType.OPENCV_ACIRCLE,
        7,
        11,
        0.04,
        0.02,
        0.30,
        0.24,
        "opencv_acircle",
        (0.02, 0.02),
        thickness_m=0.0005,
    ),
}


def get_board_config(
    board_type: BoardType | str = BoardType.CCTAG_MULTI,
) -> BoardConfig:
    return BOARD_CONFIGS[BoardType(board_type)]


__all__ = ["BoardConfig", "BoardType", "BOARD_CONFIGS", "get_board_config"]
