"""Order calibration configurations by joint-space travel distance."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisitOrder:
    """An input-row permutation and travel distances in joint-space radians."""

    indices: np.ndarray
    segment_distances_rad: np.ndarray
    total_distance_rad: float
    initial_distance_rad: float


def _configurations(value, *, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real joint values")
    if shape is not None:
        if raw.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    elif raw.ndim != 2 or min(raw.shape) == 0:
        raise ValueError(f"{name} must have non-empty shape [N, DOF]")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(float, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real joint values")
    return result


def _distance_matrix(configurations):
    # Compute each unordered pair once, without an [N, N, DOF] temporary.
    distances = np.empty((len(configurations), len(configurations)))
    np.fill_diagonal(distances, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for index, configuration in enumerate(configurations[:-1]):
            row = np.hypot.reduce(configurations[index + 1 :] - configuration, axis=1)
            distances[index, index + 1 :] = row
            distances[index + 1 :, index] = row
    if not np.isfinite(distances).all():
        raise ValueError("joint-space distances exceed the finite numeric range")
    return distances


def _segments(order, distances, start_index, return_to_start):
    path = order
    if start_index is not None:
        path = np.r_[start_index, path]
        if return_to_start:
            path = np.r_[path, start_index]
    return distances[path[:-1], path[1:]]


def _nearest_neighbor(distances, count, start_index):
    remaining = np.ones(count, dtype=bool)
    order = np.empty(count, dtype=np.int64)
    current = 0 if start_index is None else start_index
    for index in range(count):
        available = np.flatnonzero(remaining)
        # The first input pose stays fixed when there is no explicit start.
        chosen = int(available[np.argmin(distances[current, available])])
        order[index] = chosen
        remaining[chosen] = False
        current = chosen
    return order


def _improve_order(order, distances, start_index, return_to_start):
    """Apply up to 20 deterministic 2-opt passes, including open-path tails."""
    count = len(order)
    first_movable = 1 if start_index is None else 0
    length = float(_segments(order, distances, start_index, return_to_start).sum())
    tolerance = 16 * np.finfo(float).eps * max(1.0, length)
    for _ in range(20):
        changed = False
        for first in range(first_movable, count - 1):
            preceding = order[first - 1] if first else start_index
            candidates = order[first + 1 :]
            successors = order[first + 2 :]
            old_edges = np.full(len(candidates), distances[preceding, order[first]])
            new_edges = distances[preceding, candidates]
            # Every candidate except the final visit has a successor.
            old_edges[:-1] += distances[candidates[:-1], successors]
            new_edges[:-1] += distances[order[first], successors]
            if return_to_start:
                old_edges[-1] += distances[candidates[-1], start_index]
                new_edges[-1] += distances[order[first], start_index]
            improvements = old_edges - new_edges
            best = int(np.argmax(improvements))
            if improvements[best] > tolerance:
                end = first + best + 2
                order[first:end] = order[first:end][::-1]
                changed = True
        if not changed:
            break
    return order


def plan_visit_order(
    joint_positions: np.ndarray,
    *,
    start_configuration: np.ndarray | None = None,
    return_to_start: bool = False,
) -> VisitOrder:
    """Visit every input configuration once using a short deterministic route.

    Distances are Euclidean joint displacements in radians, without angular
    wrapping. This respects the supplied joint values but does not check limits,
    collisions, visibility or motion feasibility. No trajectory timing is added.

    Without an explicit start, input row zero remains the first visit. With a
    start, its distance to the first visit is included; ``return_to_start`` also
    includes the final return segment and requires an explicit start. Duplicate
    configurations retain separate row identities and may have zero-length legs.

    Nearest-neighbor initialization and at most 20 2-opt passes are a heuristic,
    not a global shortest-path guarantee. The result is never longer than the
    original input order under the same start/return conditions. Inputs are not
    modified; ``indices`` maps the visits back to their original input rows.
    """
    positions = _configurations(joint_positions, name="joint_positions")
    if not isinstance(return_to_start, bool):
        raise ValueError("return_to_start must be a boolean")
    if return_to_start and start_configuration is None:
        raise ValueError("return_to_start requires start_configuration")
    count, dof = positions.shape
    start_index = None
    if start_configuration is not None:
        start = _configurations(
            start_configuration, name="start_configuration", shape=(dof,)
        )
        positions = np.vstack([positions, start])
        start_index = count
    distances = _distance_matrix(positions)
    original = np.arange(count, dtype=np.int64)
    original_segments = _segments(original, distances, start_index, return_to_start)
    with np.errstate(over="ignore"):
        initial_distance = float(original_segments.sum())
    if not np.isfinite(initial_distance):
        raise ValueError("total joint-space distance exceeds the finite numeric range")
    nearest = _nearest_neighbor(distances, count, start_index)
    nearest_distance = float(
        _segments(nearest, distances, start_index, return_to_start).sum()
    )
    order = nearest if nearest_distance < initial_distance else original.copy()
    order = _improve_order(order, distances, start_index, return_to_start)
    segments = _segments(order, distances, start_index, return_to_start)
    total_distance = float(segments.sum())
    # Recompute the complete distance after local edge comparisons to guard
    # against rounding accumulation in unusually scaled joint coordinates.
    if total_distance > initial_distance:
        order, segments, total_distance = original, original_segments, initial_distance
    return VisitOrder(order, segments.copy(), total_distance, initial_distance)
