"""Shared numerical acceptance criteria for library and CLI consumers."""

import numpy as np


def evaluate_solution_quality(
    *,
    converged: bool,
    full_rank: bool,
    position_error_max_m: float,
    orientation_error_max_rad: float,
    condition_number: float,
    max_position_error: float,
    max_orientation_error: float,
    max_condition_number: float,
) -> tuple[bool, list[str]]:
    """Apply common numerical acceptance criteria to a calibration result."""
    thresholds = (max_position_error, max_orientation_error, max_condition_number)
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("quality thresholds must be positive finite values")
    failures = []
    if not converged:
        failures.append("solver did not converge")
    if not full_rank:
        failures.append("calibration Jacobian/regressor is rank deficient")
    if not np.isfinite(position_error_max_m):
        failures.append("maximum position error is not finite")
    elif position_error_max_m > max_position_error:
        failures.append(
            f"maximum position error {position_error_max_m:.6g} m exceeds "
            f"{max_position_error:.6g} m"
        )
    if not np.isfinite(orientation_error_max_rad):
        failures.append("maximum orientation error is not finite")
    elif orientation_error_max_rad > max_orientation_error:
        failures.append(
            f"maximum orientation error {orientation_error_max_rad:.6g} rad "
            f"exceeds {max_orientation_error:.6g} rad"
        )
    if not np.isfinite(condition_number) or condition_number > max_condition_number:
        failures.append(
            f"condition number {condition_number:.6g} exceeds "
            f"{max_condition_number:.6g}"
        )
    return not failures, failures
