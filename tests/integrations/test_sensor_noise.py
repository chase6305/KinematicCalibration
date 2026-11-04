import numpy as np
import pytest

from kinematic_calibration.sensor_noise import (
    RGBDNoiseConfig,
    add_rgbd_gaussian_noise,
)


def test_rgbd_noise_is_reproducible_and_preserves_invalid_depth() -> None:
    rgb = np.full((8, 7, 4), 128, dtype=np.uint8)
    rgb[..., 3] = 255
    depth = np.full((8, 7), 1.2, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    config = RGBDNoiseConfig(rgb_std=5.0, depth_std_m=0.002, seed=17)

    first = add_rgbd_gaussian_noise(
        rgb, depth, config, np.random.default_rng(config.seed)
    )
    second = add_rgbd_gaussian_noise(
        rgb, depth, config, np.random.default_rng(config.seed)
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].dtype == rgb.dtype
    assert first[1].dtype == depth.dtype
    np.testing.assert_array_equal(first[0][..., 3], rgb[..., 3])
    assert first[1][0, 0] == 0.0
    assert np.isnan(first[1][0, 1])
    assert np.any(first[0][..., :3] != rgb[..., :3])
    assert np.any(first[1][1:] != depth[1:])


def test_disabled_rgbd_noise_returns_independent_equal_copies() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)

    noisy_rgb, noisy_depth = add_rgbd_gaussian_noise(
        rgb, depth, RGBDNoiseConfig(), np.random.default_rng(1)
    )

    np.testing.assert_array_equal(noisy_rgb, rgb)
    np.testing.assert_array_equal(noisy_depth, depth)
    assert not np.shares_memory(noisy_rgb, rgb)
    assert not np.shares_memory(noisy_depth, depth)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("rgb_std", -1.0), ("depth_std_m", float("nan")), ("seed", 1.5)],
)
def test_rgbd_noise_configuration_rejects_invalid_values(keyword, value) -> None:
    with pytest.raises(ValueError):
        RGBDNoiseConfig(**{keyword: value})
