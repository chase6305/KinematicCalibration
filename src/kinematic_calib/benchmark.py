"""Perturbation-recovery benchmark for kinematic calibration."""

from dataclasses import asdict, dataclass

import numpy as np

from .robot_model import initialize_erobot_screw_axes, validate_nominal_poe


@dataclass(frozen=True)
class ErrorStatistics:
    """Mean and worst-case pose errors."""

    position_mean_m: float
    position_max_m: float
    orientation_mean_rad: float
    orientation_max_rad: float


def _pose_error_statistics(predictions, references) -> ErrorStatistics:
    position_errors = []
    orientation_errors = []
    for prediction, reference in zip(predictions, references, strict=True):
        difference = prediction @ reference.inv()
        position_errors.append(np.linalg.norm(prediction.t - reference.t))
        orientation_errors.append(abs(float(difference.angvec()[0])))
    return ErrorStatistics(
        position_mean_m=float(np.mean(position_errors)),
        position_max_m=float(np.max(position_errors)),
        orientation_mean_rad=float(np.mean(orientation_errors)),
        orientation_max_rad=float(np.max(orientation_errors)),
    )


def _improvement(before: float, after: float) -> float:
    if before <= np.finfo(float).eps:
        return 0.0
    return float((before - after) / before)


def run_perturbation_benchmark(
    robot,
    ee_name: str,
    train_samples: int = 40,
    validation_samples: int = 20,
    perturbation: float = 0.01,
    seed: int = 1234,
    max_iterations: int = 100,
    position_noise_std_m: float = 0.0,
    orientation_noise_std_rad: float = 0.0,
    joint_noise_std_rad: float = 0.0,
    outlier_ratio: float = 0.0,
    outlier_position_std_m: float = 0.02,
    outlier_orientation_std_rad: float = 0.05,
) -> dict:
    """Recover a perturbed chain from noisy data and test on clean held-out poses."""
    from spatialmath import SE3, SO3

    from robotkinecal import SerialRobotKineCal

    if train_samples <= 0 or validation_samples <= 0:
        raise ValueError("train and validation sample counts must be positive")
    if not np.isfinite(perturbation) or perturbation <= 0.0:
        raise ValueError("perturbation must be a positive finite value")
    noise_scales = {
        "position_noise_std_m": position_noise_std_m,
        "orientation_noise_std_rad": orientation_noise_std_rad,
        "joint_noise_std_rad": joint_noise_std_rad,
        "outlier_position_std_m": outlier_position_std_m,
        "outlier_orientation_std_rad": outlier_orientation_std_rad,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in noise_scales.values()):
        raise ValueError("noise standard deviations must be finite and non-negative")
    if not np.isfinite(outlier_ratio) or not 0.0 <= outlier_ratio <= 1.0:
        raise ValueError("outlier_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)
    calibrator = SerialRobotKineCal(robot, ee_name, verbose=False)
    initialize_erobot_screw_axes(calibrator)
    validate_nominal_poe(calibrator, np.zeros(calibrator.N_JOINTS))
    nominal_axes = [axis.copy() for axis in calibrator.joint_screw_axis]
    nominal_zero_pose = SE3(calibrator.zero_conf_EE_pose.A.copy())
    perturbed_axes = []
    for axis in nominal_axes:
        direction = axis[3:].copy()
        point = np.cross(direction, axis[:3])
        direction += rng.normal(scale=perturbation, size=3)
        direction /= np.linalg.norm(direction)
        point += rng.normal(scale=perturbation, size=3)
        perturbed_axes.append(np.block([np.cross(point, direction), direction]))

    terminal_offset = SE3.Rt(
        R=SO3.RPY(rng.normal(scale=perturbation, size=3)),
        t=rng.normal(scale=perturbation, size=3),
    )
    perturbed_zero_pose = nominal_zero_pose @ terminal_offset
    dof = calibrator.N_JOINTS
    train_q_truth = rng.uniform(-np.pi, np.pi, (train_samples, dof))
    clean_train_observations = [
        _forward_poe(perturbed_axes, perturbed_zero_pose, q) for q in train_q_truth
    ]
    measured_train_q = train_q_truth + rng.normal(
        scale=joint_noise_std_rad, size=train_q_truth.shape
    )
    outlier_count = min(
        train_samples, int(np.floor(outlier_ratio * train_samples + 0.5))
    )
    outlier_indices = np.sort(
        rng.choice(train_samples, size=outlier_count, replace=False)
    )
    outlier_mask = np.zeros(train_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    train_observations = []
    for index, clean_pose in enumerate(clean_train_observations):
        translation_std = position_noise_std_m
        orientation_std = orientation_noise_std_rad
        if outlier_mask[index]:
            translation_std = np.hypot(translation_std, outlier_position_std_m)
            orientation_std = np.hypot(orientation_std, outlier_orientation_std_rad)
        noise = SE3.Rt(
            R=SO3.RPY(rng.normal(scale=orientation_std, size=3)),
            t=rng.normal(scale=translation_std, size=3),
        )
        train_observations.append(clean_pose @ noise)
    injected_pose_errors = _pose_error_statistics(
        train_observations, clean_train_observations
    )
    joint_noise = measured_train_q - train_q_truth
    calibrator.set_observations(measured_train_q, train_observations)
    result = calibrator.solve(max_iterations=max_iterations)
    estimated_axes, estimated_zero_pose = calibrator.get_calibration(result)

    initial_axis_errors = np.asarray(
        [np.linalg.norm(a - truth) for a, truth in zip(nominal_axes, perturbed_axes)]
    )
    final_axis_errors = np.asarray(
        [
            np.linalg.norm(estimate - truth)
            for estimate, truth in zip(estimated_axes, perturbed_axes)
        ]
    )
    initial_terminal_error = float(
        np.linalg.norm((nominal_zero_pose.inv() @ perturbed_zero_pose).log(twist=True))
    )
    final_terminal_error = float(
        np.linalg.norm(
            (estimated_zero_pose.inv() @ perturbed_zero_pose).log(twist=True)
        )
    )

    validation_q = rng.uniform(-np.pi, np.pi, (validation_samples, dof))
    truth_poses = [
        _forward_poe(perturbed_axes, perturbed_zero_pose, q) for q in validation_q
    ]
    nominal_poses = [
        _forward_poe(nominal_axes, nominal_zero_pose, q) for q in validation_q
    ]
    calibrated_poses = [
        _forward_poe(estimated_axes, estimated_zero_pose, q) for q in validation_q
    ]
    before = _pose_error_statistics(nominal_poses, truth_poses)
    after = _pose_error_statistics(calibrated_poses, truth_poses)
    final_iteration = result.iteration_results[-1]
    solver_statistics = final_iteration.get_statistics()

    return {
        "schema_version": 1,
        "report_type": "kinematic-perturbation-recovery-benchmark",
        "seed": seed,
        "dof": dof,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "perturbation": perturbation,
        "noise_configuration": {
            **noise_scales,
            "outlier_ratio": outlier_ratio,
            "outlier_count": outlier_count,
            "realized_outlier_ratio": outlier_count / train_samples,
            "outlier_indices": outlier_indices.tolist(),
            "noise_applied_to": "training observations only",
            "validation_data": "clean held-out ground truth",
        },
        "injected_noise_statistics": {
            "joint_error_rms_rad": float(np.sqrt(np.mean(joint_noise**2))),
            "joint_error_max_abs_rad": float(np.max(np.abs(joint_noise))),
            **asdict(injected_pose_errors),
        },
        "converged": bool(result.has_converged),
        "termination_reason": result.termination_reason,
        "regressor_rank": int(solver_statistics["matrix_rank"]),
        "parameter_count": int(final_iteration.N_PARAMS),
        "condition_number": float(solver_statistics["condition_number"]),
        "screw_axis_initial_errors": initial_axis_errors.tolist(),
        "screw_axis_final_errors": final_axis_errors.tolist(),
        "screw_axis_mean_improvement": _improvement(
            float(np.mean(initial_axis_errors)), float(np.mean(final_axis_errors))
        ),
        "terminal_initial_twist_error": initial_terminal_error,
        "terminal_final_twist_error": final_terminal_error,
        "terminal_error_improvement": _improvement(
            initial_terminal_error, final_terminal_error
        ),
        "validation_before": asdict(before),
        "validation_after": asdict(after),
        "validation_position_mean_improvement": _improvement(
            before.position_mean_m, after.position_mean_m
        ),
        "validation_orientation_mean_improvement": _improvement(
            before.orientation_mean_rad, after.orientation_mean_rad
        ),
    }


def _forward_poe(screw_axes, zero_pose, joint_positions):
    """Evaluate a POE model without relying on PoERobot version details."""
    from spatialmath import SE3, Twist3

    pose = SE3()
    for axis, position in zip(screw_axes, joint_positions, strict=True):
        pose *= Twist3(axis).exp(position)
    return pose @ zero_pose
