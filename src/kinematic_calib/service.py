"""Engine-independent, in-memory kinematic calibration service.

The CLI owns files and provenance; this module owns the fit. A caller supplies
an initialized RobotKineCal-compatible calibrator, so acquisition, visualization,
URDF loading and model selection remain outside the numerical workflow.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import KinematicDataset, ObservationFrame
from .robot_model import (
    estimate_composite_extrinsics,
    initialize_composite_model,
    validate_nominal_poe,
)
from .robust import HuberOptions


@dataclass(frozen=True)
class FitOptions:
    """Fit controls with metre/radian observation conventions."""

    model_mode: str = "robot-body"
    max_iterations: int = 100
    damping: float = 0.0
    robust: HuberOptions | None = None

    def __post_init__(self) -> None:
        if self.model_mode not in {"robot-body", "self-calibrating"}:
            raise ValueError("model_mode must be robot-body or self-calibrating")
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        if self.robust is not None and not isinstance(self.robust, HuberOptions):
            raise ValueError("robust must be HuberOptions or None")


@dataclass(frozen=True)
class CalibrationFit:
    """Fit products; numerical acceptance and export remain explicit steps.

    In self-calibrating mode the returned POE is in Camera/Target coordinates.
    Its base and terminal gauges are not separately identifiable.
    """

    result: Any
    screw_axes: np.ndarray
    zero_pose: Any
    dataset: KinematicDataset
    estimated_camera_t_base: np.ndarray | None = None
    estimated_ee_t_target: np.ndarray | None = None
    extrinsic_initialization: dict | None = None
    joint_identity: dict | None = None
    robust_diagnostics: dict | None = None


def fit_dataset(
    dataset: KinematicDataset,
    calibrator=None,
    *,
    options: FitOptions | None = None,
    base_t_camera: np.ndarray | None = None,
    ee_t_target: np.ndarray | None = None,
    model_joint_names: tuple[str, ...] | None = None,
    calibrator_factory=None,
    sample_weights: np.ndarray | None = None,
) -> CalibrationFit:
    """Fit synchronized observations without writing artifacts.

    ``calibrator`` is mutated by the upstream solver; create a fresh instance
    or call its ``reset()`` before fitting another dataset. For robot-body mode,
    Camera_T_Target inputs require both fixed transforms. Self-calibrating mode
    estimates initial Camera/Base and EE/Target gauges from the nominal model
    and observations, then fits the composite POE without an extrinsic prior.
    """
    from spatialmath import SE3

    options = options or FitOptions()
    source_observation_frame = dataset.observation_frame.value
    if options.robust is not None and options.model_mode != "robot-body":
        raise ValueError(
            "Huber IRLS supports robot-body observations with fixed extrinsics; unknown self-calibrating extrinsics are unsupported"
        )
    self_calibrating = options.model_mode == "self-calibrating"
    if self_calibrating:
        if base_t_camera is not None or ee_t_target is not None:
            raise ValueError("self-calibrating mode does not accept fixed extrinsics")
        if dataset.observation_frame != ObservationFrame.CAMERA_T_TARGET:
            raise ValueError(
                "self-calibrating mode requires camera_T_target observations"
            )
    elif dataset.observation_frame == ObservationFrame.CAMERA_T_TARGET:
        if base_t_camera is None or ee_t_target is None:
            raise ValueError(
                "robot-body camera observations require both fixed extrinsics"
            )
        dataset = dataset.as_robot_body(base_t_camera, ee_t_target)
    elif base_t_camera is not None or ee_t_target is not None:
        raise ValueError("base_T_ee observations do not accept fixed extrinsics")

    if options.robust is not None:
        if calibrator is not None or not callable(calibrator_factory):
            raise ValueError(
                "Huber IRLS requires calibrator_factory and no existing calibrator object"
            )
        if sample_weights is not None or model_joint_names is not None:
            raise ValueError(
                "Huber IRLS obtains weights internally and joint names from each fresh factory model"
            )
        from .robust import fit_huber_irls

        return fit_huber_irls(
            dataset,
            calibrator_factory,
            options=options,
            source_observation_frame=source_observation_frame,
        )
    if calibrator_factory is not None or calibrator is None:
        raise ValueError(
            "ordinary fitting requires an initialized calibrator; calibrator_factory is for opt-in Huber IRLS"
        )

    chain_dof = int(calibrator.N_JOINTS)
    if chain_dof != dataset.joint_positions.shape[1]:
        raise ValueError(
            f"selected chain has {chain_dof} revolute joints but dataset has "
            f"{dataset.joint_positions.shape[1]} columns"
        )
    identity = dataset.check_joint_identity(
        model_joint_names
        if model_joint_names is not None
        else getattr(calibrator, "joint_names", None)
    )
    if not identity["verified"]:
        logging.getLogger(__name__).warning("%s", identity["reason"])
    dataset.validate_solver_sample_count()
    if getattr(calibrator, "robot_model", None) is not None:
        validate_nominal_poe(calibrator, dataset.joint_positions[0])

    camera_t_base = ee_t_target_estimate = initialization = None
    if self_calibrating:
        camera_t_base, ee_t_target_estimate, initialization = (
            estimate_composite_extrinsics(
                calibrator, dataset.joint_positions, dataset.observed_poses
            )
        )
        initialize_composite_model(calibrator, camera_t_base, ee_t_target_estimate)

    # Dataset construction already checks rigid transforms. Avoid rejecting
    # valid near-pi rotations because of floating-point roundoff in composition.
    observations = [SE3(pose, check=False) for pose in dataset.observed_poses]
    calibrator.set_observations(dataset.joint_positions, observations)
    solve_options = {
        "max_iterations": options.max_iterations,
        "damping": options.damping,
    }
    if sample_weights is not None:
        solve_options["sample_weights"] = sample_weights
    result = calibrator.solve(**solve_options)
    axes, zero_pose = calibrator.get_calibration(result)
    return CalibrationFit(
        result=result,
        screw_axes=np.asarray(axes).copy(),
        zero_pose=zero_pose,
        dataset=dataset,
        estimated_camera_t_base=camera_t_base,
        estimated_ee_t_target=ee_t_target_estimate,
        extrinsic_initialization=initialization,
        joint_identity=identity,
    )
