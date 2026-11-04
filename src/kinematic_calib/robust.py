"""Explicit sample-wise Huber IRLS for fixed-frame kinematic observations.

Weights use Euclidean endpoint and SO(3) angle errors. Each inner solver still
minimizes its weighted spatial-twist objective. This is a reweighting heuristic,
not an exact optimizer of the endpoint Huber objective or an uncertainty model.
"""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .dataset import KinematicDataset
from .validation import evaluate_saved_poe, validate_saved_poe_parameters


def _positive_real(value, name: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite real number")
    return float(value)


@dataclass(frozen=True)
class HuberOptions:
    """Fixed measurement scales and controls for opt-in IRLS.

    Scales are supplied by the caller in metres and radians. They are never
    estimated from residuals, and validation observations are never consulted.
    ``min_residual_multiplier`` keeps every sample strictly in the problem.
    """

    position_scale_m: float
    orientation_scale_rad: float
    huber_delta: float = 3.5
    max_outer_iterations: int = 12
    weight_tolerance: float = 1e-3
    min_residual_multiplier: float = 1e-3

    def __post_init__(self):
        for name in (
            "position_scale_m",
            "orientation_scale_rad",
            "huber_delta",
            "weight_tolerance",
            "min_residual_multiplier",
        ):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if (
            not isinstance(self.max_outer_iterations, int)
            or isinstance(self.max_outer_iterations, bool)
            or self.max_outer_iterations < 1
        ):
            raise ValueError("max_outer_iterations must be a positive integer")
        if self.weight_tolerance >= 1 or self.min_residual_multiplier >= 1:
            raise ValueError(
                "weight_tolerance and min_residual_multiplier must be below one"
            )


def huber_sample_weights(position_errors, orientation_errors, options: HuberOptions):
    """Return normalized errors, positive residual multipliers and Huber losses.

    One normalized error combines the two endpoint norms. Its IRLS precision
    is ``min(1, delta / error)``; the core receives the square root because it
    multiplies residual rows, rather than squared-error terms.
    """
    values = []
    for value in (position_errors, orientation_errors):
        raw = np.asarray(value)
        if (
            raw.dtype.kind not in "iuf"
            or raw.ndim != 1
            or not len(raw)
            or not np.isfinite(raw).all()
            or np.any(raw < 0)
        ):
            raise ValueError("endpoint errors must be finite nonnegative real vectors")
        values.append(raw.astype(float))
    position, orientation = values
    if position.shape != orientation.shape:
        raise ValueError("endpoint error counts must match")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.hypot(
            position / options.position_scale_m,
            orientation / options.orientation_scale_rad,
        )
    if not np.isfinite(normalized).all():
        raise ValueError("normalized endpoint errors exceed the finite numeric range")
    precision = np.ones(len(normalized))
    outside = normalized > options.huber_delta
    precision[outside] = options.huber_delta / normalized[outside]
    multipliers = np.maximum(np.sqrt(precision), options.min_residual_multiplier)
    losses = np.empty(len(normalized))
    with np.errstate(over="ignore", invalid="ignore"):
        losses[~outside] = 0.5 * normalized[~outside] ** 2
        losses[outside] = options.huber_delta * (
            normalized[outside] - 0.5 * options.huber_delta
        )
    if not np.isfinite(losses).all():
        raise ValueError("Huber loss exceeds the finite numeric range")
    return normalized, multipliers, losses


def _nominal_signature(calibrator):
    try:
        axes, home = validate_saved_poe_parameters(
            np.asarray(calibrator.joint_screw_axis), calibrator.zero_conf_EE_pose.A
        )
    except AttributeError as exc:
        raise ValueError(
            "calibrator_factory must return an initialized POE calibrator"
        ) from exc
    names = getattr(calibrator, "joint_names", None)
    return axes.copy(), home.copy(), tuple(names) if names is not None else None


def fit_huber_irls(
    dataset: KinematicDataset,
    calibrator_factory: Callable,
    *,
    options,
    source_observation_frame: str,
):
    """Run the internal IRLS loop; callers normally use ``fit_dataset``.

    The service has already converted known camera extrinsics into Base_T_EE.
    A fresh, geometrically identical nominal model is required on every pass.
    """
    from .service import fit_dataset

    robust = options.robust
    inner_options = replace(options, robust=None)
    weights = np.ones(len(dataset.joint_positions))
    # Retain references so object-id reuse after GC cannot defeat the fresh-model
    # check. Calibration results from previous rounds are not retained.
    previous_calibrators = []
    nominal = None
    trace = []
    outer_converged = False
    for iteration in range(robust.max_outer_iterations):
        calibrator = calibrator_factory()
        if any(calibrator is previous for previous in previous_calibrators):
            raise ValueError(
                "calibrator_factory reused an existing object; every IRLS round requires a fresh nominal calibrator"
            )
        previous_calibrators.append(calibrator)
        signature = _nominal_signature(calibrator)
        if nominal is None:
            nominal = signature
        elif (
            signature[0].shape != nominal[0].shape
            or not np.allclose(signature[0], nominal[0], atol=1e-12, rtol=0)
            or not np.allclose(signature[1], nominal[1], atol=1e-12, rtol=0)
            or signature[2] != nominal[2]
        ):
            raise ValueError(
                "calibrator_factory changed the initial nominal axes, home pose or joint names between IRLS rounds"
            )
        fitted = fit_dataset(
            dataset, calibrator, options=inner_options, sample_weights=weights
        )
        measured_errors = evaluate_saved_poe(
            fitted.screw_axes,
            fitted.zero_pose.A,
            dataset.joint_positions,
            dataset.observed_poses,
            dataset.sample_indices,
        )["sample_errors"]
        position = np.asarray([row["position_error_m"] for row in measured_errors])
        orientation = np.asarray(
            [row["orientation_error_rad"] for row in measured_errors]
        )
        normalized, proposed, losses = huber_sample_weights(
            position, orientation, robust
        )
        change = float(np.max(np.abs(proposed - weights)))
        last = fitted.result.iteration_results[-1]
        weighted_rank = int(last.weighted_matrix_rank)
        weighted_condition = float(last.weighted_condition_number)
        trace.append(
            {
                "outer_iteration": iteration + 1,
                "inner_converged": bool(fitted.result.has_converged),
                "inner_termination_reason": fitted.result.termination_reason,
                "inner_iterations": len(fitted.result.iteration_results),
                "endpoint_huber_objective": float(np.sum(losses)),
                "sample_residual_multipliers": weights.tolist(),
                "maximum_multiplier_change": change,
                "downweighted_samples": int(np.count_nonzero(weights < 1.0)),
                "raw_regressor_rank": int(last.matrix_rank),
                "weighted_regressor_rank": weighted_rank,
                "parameter_count": int(last.N_PARAMS),
                "weighted_condition_number": weighted_condition
                if np.isfinite(weighted_condition)
                else None,
            }
        )
        if change <= robust.weight_tolerance:
            outer_converged = True
            break
        if iteration + 1 < robust.max_outer_iterations:
            weights = proposed
    diagnostics = {
        "method": "sample-wise-huber-irls-heuristic",
        "residual_definition": "hypot(Euclidean endpoint position error / position_scale_m, SO(3) angle error / orientation_scale_rad)",
        "inner_objective": "weighted spatial-twist squared error; endpoint Huber cost is a diagnostic, not an exact minimized objective",
        "source_observation_frame": source_observation_frame,
        "fit_observation_frame": dataset.observation_frame.value,
        "position_scale_m": robust.position_scale_m,
        "orientation_scale_rad": robust.orientation_scale_rad,
        "huber_delta": robust.huber_delta,
        "min_residual_multiplier": robust.min_residual_multiplier,
        "weight_tolerance": robust.weight_tolerance,
        "outer_converged": outer_converged,
        "converged": outer_converged and bool(fitted.result.has_converged),
        "outer_termination_reason": "weights_stable"
        if outer_converged
        else "max_outer_iterations",
        "inner_converged": bool(fitted.result.has_converged),
        "weighted_regressor_rank": weighted_rank,
        "parameter_count": int(last.N_PARAMS),
        "weighted_condition_number": weighted_condition
        if np.isfinite(weighted_condition)
        else None,
        "samples_retained": len(dataset.joint_positions),
        "samples_removed": 0,
        "downweighted_samples": int(np.count_nonzero(weights < 1.0)),
        "uncertainty_interpretation": "inner weighted local-linear heuristic; data-dependent Huber weight estimation is not included in uncertainty",
        "iterations": trace,
        "samples": [
            {
                "sample_id": int(dataset.sample_indices[index]),
                "dataset_row": index,
                "residual_multiplier": float(weights[index]),
                "precision_weight": float(weights[index] ** 2),
                "normalized_endpoint_residual": float(normalized[index]),
                "position_error_m": float(position[index]),
                "orientation_error_rad": float(orientation[index]),
                "downweighted": bool(weights[index] < 1.0),
            }
            for index in range(len(weights))
        ],
    }
    return replace(fitted, robust_diagnostics=diagnostics)


__all__ = ["HuberOptions", "huber_sample_weights"]
