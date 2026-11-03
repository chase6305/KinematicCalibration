"""Headless synthetic pose measurements with independent truth geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    invert_transform,
    validated_joint_names,
)

from .backends.protocol import KinematicsBackend


@dataclass(frozen=True)
class MeasurementNoise:
    position_std_m: float = 0.0001
    orientation_std_rad: float = 0.0002

    def __post_init__(self):
        values = [self.position_std_m, self.orientation_std_rad]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(
                "noise standard deviations must be finite and non-negative"
            )


def _random_generator(seed):
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(seed)


def _joint_limits(truth):
    limits = np.asarray(truth.joint_limits, dtype=float)
    if (
        limits.shape != (truth.dof, 2)
        or truth.dof < 1
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("backend joint_limits must be finite ordered [DOF,2] ranges")
    return limits


def sample_measurements(
    truth: KinematicsBackend,
    *,
    count: int = 100,
    seed: int = 42,
    noise: MeasurementNoise | None = None,
    base_t_camera: np.ndarray | None = None,
    ee_t_target: np.ndarray | None = None,
) -> KinematicDataset:
    """Sample synchronized stationary poses; this is a pose sensor model.

    No camera images, occlusion, dynamics or collision constraints are simulated.
    Translation noise acts in the observation frame, rotation noise on the right.
    A camera observation requires both fixed extrinsics.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    rng = _random_generator(seed)
    limits = _joint_limits(truth)
    lower = np.maximum(limits[:, 0], -np.pi)
    upper = np.minimum(limits[:, 1], np.pi)
    # A legitimate URDF range may lie outside [-pi,pi]. Keep that range intact.
    empty = lower >= upper
    lower[empty], upper[empty] = limits[empty, 0], limits[empty, 1]
    joints = rng.uniform(lower, upper, (count, truth.dof))
    return _measure_configurations(
        truth,
        joints,
        rng=rng,
        noise=noise,
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
    )


def measure_configurations(
    truth: KinematicsBackend,
    joint_positions: np.ndarray,
    *,
    seed: int = 42,
    noise: MeasurementNoise | None = None,
    base_t_camera: np.ndarray | None = None,
    ee_t_target: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
) -> KinematicDataset:
    """Measure stationary configurations in their supplied order.

    ``truth`` supplies synthetic sensor poses; a nominal planning model alone
    cannot establish calibration accuracy. Start and return legs are not poses
    to measure. Noise and camera frames follow :func:`sample_measurements`.
    """
    rng = _random_generator(seed)
    raw = np.asarray(joint_positions)
    if (
        raw.dtype.kind not in "iuf"
        or raw.ndim != 2
        or raw.shape[1] != truth.dof
        or not len(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("joint_positions must be finite real nonempty [N,DOF]")
    joints = raw.astype(float, copy=True)
    limits = _joint_limits(truth)
    if (
        not np.isfinite(joints).all()
        or np.any(joints < limits[:, 0])
        or np.any(joints > limits[:, 1])
    ):
        raise ValueError("joint_positions must respect backend joint limits")
    return _measure_configurations(
        truth,
        joints,
        rng=rng,
        noise=noise,
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
        sample_indices=sample_indices,
    )


def _measure_configurations(
    truth, joints, *, rng, noise, base_t_camera, ee_t_target, sample_indices=None
):
    if (base_t_camera is None) != (ee_t_target is None):
        raise ValueError("camera measurements require both fixed transforms")
    joint_names = validated_joint_names(
        getattr(truth, "joint_names", None), truth.dof, label="backend joint_names"
    )
    noise = noise or MeasurementNoise()
    camera_t_base = None
    if base_t_camera is not None:
        camera_t_base = invert_transform(base_t_camera)
        invert_transform(ee_t_target)  # validate before taking any measurements
    poses = []
    for q in joints:
        pose = truth.forward(q).copy()
        if camera_t_base is not None:
            pose = camera_t_base @ pose @ ee_t_target
        pose[:3, 3] += rng.normal(0, noise.position_std_m, 3)
        pose[:3, :3] = (
            pose[:3, :3]
            @ Rotation.from_rotvec(
                rng.normal(0, noise.orientation_std_rad, 3)
            ).as_matrix()
        )
        poses.append(pose)
    frame = (
        ObservationFrame.BASE_T_EE
        if camera_t_base is None
        else ObservationFrame.CAMERA_T_TARGET
    )
    return KinematicDataset(
        joints, np.asarray(poses), frame, sample_indices, joint_names
    )
