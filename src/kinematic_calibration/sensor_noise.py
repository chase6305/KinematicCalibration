"""Deterministic sensor-noise models applied before board detection."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDNoiseConfig:
    """Gaussian RGB/depth measurement noise with explicit units."""

    rgb_std: float = 0.0
    depth_std_m: float = 0.0
    seed: int = 1234

    def __post_init__(self) -> None:
        for name, value in (
            ("rgb_std", self.rgb_std),
            ("depth_std_m", self.depth_std_m),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, bool):
            raise ValueError("noise seed must be an integer")

    @property
    def enabled(self) -> bool:
        return self.rgb_std > 0.0 or self.depth_std_m > 0.0


def add_rgbd_gaussian_noise(
    rgb: np.ndarray,
    depth: np.ndarray,
    config: RGBDNoiseConfig,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy copies while preserving RGB dtype and invalid depth pixels."""
    rgb_value = np.asarray(rgb)
    depth_value = np.asarray(depth)
    if rgb_value.ndim != 3 or rgb_value.shape[:2] != depth_value.shape:
        raise ValueError("RGB and depth image shapes are inconsistent")
    if not np.issubdtype(rgb_value.dtype, np.integer):
        raise ValueError("RGB image must use an integer dtype")
    if not np.issubdtype(depth_value.dtype, np.floating):
        raise ValueError("depth image must use a floating-point dtype")

    noisy_rgb = rgb_value.copy()
    if config.rgb_std > 0.0:
        channel_count = min(noisy_rgb.shape[2], 3)
        color = noisy_rgb[..., :channel_count].astype(np.float64)
        color += generator.normal(scale=config.rgb_std, size=color.shape)
        maximum = np.iinfo(noisy_rgb.dtype).max
        noisy_rgb[..., :channel_count] = np.rint(
            np.clip(color, 0.0, float(maximum))
        ).astype(noisy_rgb.dtype)

    noisy_depth = depth_value.copy()
    if config.depth_std_m > 0.0:
        valid = np.isfinite(noisy_depth) & (noisy_depth > 0.0)
        perturbation = generator.normal(
            scale=config.depth_std_m, size=noisy_depth.shape
        )
        noisy_depth[valid] = np.maximum(
            noisy_depth[valid] + perturbation[valid],
            np.finfo(noisy_depth.dtype).tiny,
        )
    return noisy_rgb, noisy_depth
