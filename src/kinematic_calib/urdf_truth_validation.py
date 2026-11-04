"""CPU-only FK validation against a known ground-truth URDF."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _known_input_perturbations(
    nominal_urdf: Path, ground_truth_urdf: Path
) -> dict | None:
    """Load the exact generator inputs stored beside a known truth URDF."""
    manifest_path = ground_truth_urdf.with_suffix(".perturbation.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid perturbation manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("nominal_urdf") != nominal_urdf.name:
        raise ValueError("perturbation manifest nominal URDF does not match validation")
    if manifest.get("perturbed_urdf") != ground_truth_urdf.name:
        raise ValueError("perturbation manifest truth URDF does not match validation")
    joints = manifest.get("joints")
    if not isinstance(joints, dict) or not joints:
        raise ValueError("perturbation manifest has no joint inputs")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "preset": manifest.get("preset"),
        "scale_relative_to_small": manifest.get("scale_relative_to_small"),
        "convention": manifest.get("convention"),
        "joints": {
            name: {
                "delta_xyz_m": value.get("delta_xyz_m"),
                "delta_rpy_rad": value.get("delta_rpy_rad"),
            }
            for name, value in joints.items()
        },
    }


def _vector(element, name: str, default: str) -> np.ndarray:
    values = np.fromstring(
        default if element is None else element.get(name, default), sep=" "
    )
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid URDF {name} vector")
    return values


def _chain(path: Path, ee_name: str) -> list[dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML in {path}: {exc}") from exc
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            by_child[child.get("link")] = joint
    chain = []
    link = ee_name
    seen = set()
    while link in by_child:
        if link in seen:
            raise ValueError(f"URDF chain to {ee_name!r} contains a cycle")
        seen.add(link)
        joint = by_child[link]
        origin = joint.find("origin")
        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            raise ValueError("URDF joint has no parent link")
        joint_type = joint.get("type")
        if joint_type not in {"revolute", "continuous", "fixed"}:
            raise ValueError(f"unsupported joint type {joint_type!r}")
        limit = joint.find("limit")
        lower = upper = None
        if joint_type != "fixed" and limit is not None:
            try:
                lower = float(limit.get("lower"))
                upper = float(limit.get("upper"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"joint {joint.get('name')!r} has invalid limits"
                ) from exc
        xyz = _vector(origin, "xyz", "0 0 0")
        rpy = _vector(origin, "rpy", "0 0 0")
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        local[:3, 3] = xyz
        chain.append(
            {
                "name": joint.get("name"),
                "type": joint_type,
                "local": local,
                "xyz": xyz,
                "rpy": rpy,
                "axis": _vector(joint.find("axis"), "xyz", "1 0 0"),
                "lower": lower,
                "upper": upper,
            }
        )
        link = parent.get("link")
    if not chain:
        raise ValueError(f"no joint chain ends at {ee_name!r} in {path}")
    return list(reversed(chain))


def _fk(chain: list[dict], q: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    position = 0
    for joint in chain:
        pose = pose @ joint["local"]
        if joint["type"] != "fixed":
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(
                joint["axis"] * q[position]
            ).as_matrix()
            pose = pose @ motion
            position += 1
    if position != len(q):
        raise ValueError(f"URDF chain expects {position} joints, got {len(q)}")
    return pose


def _errors(candidate_chain, truth_chain, candidate_q, truth_q):
    position = []
    orientation = []
    for q_candidate, q_truth in zip(candidate_q, truth_q, strict=True):
        candidate = _fk(candidate_chain, q_candidate)
        truth = _fk(truth_chain, q_truth)
        delta = np.linalg.inv(truth) @ candidate
        position.append(np.linalg.norm(delta[:3, 3]))
        orientation.append(
            np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        )
    return np.asarray(position), np.asarray(orientation)


def _statistics(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _transform_error(candidate: np.ndarray, truth: np.ndarray) -> dict:
    delta = np.linalg.inv(truth) @ candidate
    translation = delta[:3, 3]
    rotation_vector = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    return {
        "translation_vector_m": translation.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_vector_rad": rotation_vector.tolist(),
        "rotation_angle_rad": float(np.linalg.norm(rotation_vector)),
    }


def _relative_improvement(before: float, after: float) -> float | None:
    if before == 0.0:
        return 1.0 if after == 0.0 else None
    return (before - after) / before


def _joint_geometry_comparison(nominal_chain, calibrated_chain, truth_chain):
    output = []
    for nominal, calibrated, truth in zip(
        nominal_chain, calibrated_chain, truth_chain, strict=True
    ):
        before = _transform_error(nominal["local"], truth["local"])
        after = _transform_error(calibrated["local"], truth["local"])
        output.append(
            {
                "joint_name": truth["name"],
                "joint_type": truth["type"],
                "nominal_origin": {
                    "xyz_m": nominal["xyz"].tolist(),
                    "rpy_rad": nominal["rpy"].tolist(),
                },
                "ground_truth_origin": {
                    "xyz_m": truth["xyz"].tolist(),
                    "rpy_rad": truth["rpy"].tolist(),
                },
                "calibrated_origin": {
                    "xyz_m": calibrated["xyz"].tolist(),
                    "rpy_rad": calibrated["rpy"].tolist(),
                },
                "injected_geometry_error": before,
                "remaining_calibrated_error": after,
                "improvement": {
                    "translation": _relative_improvement(
                        before["translation_norm_m"],
                        after["translation_norm_m"],
                    ),
                    "rotation": _relative_improvement(
                        before["rotation_angle_rad"],
                        after["rotation_angle_rad"],
                    ),
                },
            }
        )
    return output


def _joint_geometry_summary(comparison: list[dict]) -> dict:
    before_translation = np.asarray(
        [item["injected_geometry_error"]["translation_norm_m"] for item in comparison]
    )
    after_translation = np.asarray(
        [
            item["remaining_calibrated_error"]["translation_norm_m"]
            for item in comparison
        ]
    )
    before_rotation = np.asarray(
        [item["injected_geometry_error"]["rotation_angle_rad"] for item in comparison]
    )
    after_rotation = np.asarray(
        [
            item["remaining_calibrated_error"]["rotation_angle_rad"]
            for item in comparison
        ]
    )
    return {
        "joint_count": len(comparison),
        "translation_error_m": {
            "before": _statistics(before_translation),
            "after": _statistics(after_translation),
            "rms_improvement": _relative_improvement(
                float(np.sqrt(np.mean(before_translation**2))),
                float(np.sqrt(np.mean(after_translation**2))),
            ),
            "improved_joint_count": int(
                np.count_nonzero(after_translation < before_translation)
            ),
        },
        "rotation_error_rad": {
            "before": _statistics(before_rotation),
            "after": _statistics(after_rotation),
            "rms_improvement": _relative_improvement(
                float(np.sqrt(np.mean(before_rotation**2))),
                float(np.sqrt(np.mean(after_rotation**2))),
            ),
            "improved_joint_count": int(
                np.count_nonzero(after_rotation < before_rotation)
            ),
        },
    }


def compare_urdf_fk_truth(
    nominal_urdf: Path,
    calibrated_urdf: Path,
    ground_truth_urdf: Path,
    joint_min_rad: np.ndarray,
    joint_max_rad: np.ndarray,
    *,
    ee_name: str,
    samples: int = 300,
    seed: int = 1234,
    injected_bias_rad: np.ndarray | None = None,
    estimated_offset_rad: np.ndarray | None = None,
    sampling_domain: str = "joint-limits",
) -> dict:
    """Compare nominal and calibrated FK with known physical ground truth."""
    paths = [
        Path(value).resolve(strict=True)
        for value in (nominal_urdf, calibrated_urdf, ground_truth_urdf)
    ]
    chains = [_chain(path, ee_name) for path in paths]
    signatures = [[(item["name"], item["type"]) for item in chain] for chain in chains]
    if signatures[1:] != signatures[:-1]:
        raise ValueError("nominal, calibrated, and ground-truth chains do not match")
    dof = sum(item["type"] != "fixed" for item in chains[0])
    if sampling_domain == "joint-limits":
        movable = [item for item in chains[2] if item["type"] != "fixed"]
        if any(item["lower"] is None or item["upper"] is None for item in movable):
            raise ValueError(
                "ground-truth URDF requires finite lower/upper limits for all joints"
            )
        lower = np.asarray([item["lower"] for item in movable], dtype=float)
        upper = np.asarray([item["upper"] for item in movable], dtype=float)
    elif sampling_domain == "observed-range":
        lower = np.asarray(joint_min_rad, dtype=float)
        upper = np.asarray(joint_max_rad, dtype=float)
    else:
        raise ValueError(f"unsupported FK sampling domain: {sampling_domain!r}")
    if lower.shape != (dof,) or upper.shape != (dof,) or np.any(upper <= lower):
        raise ValueError("joint bounds must be ordered one-dimensional DOF vectors")
    if samples <= 0:
        raise ValueError("samples must be positive")
    bias = (
        np.zeros(dof)
        if injected_bias_rad is None
        else np.asarray(injected_bias_rad, dtype=float)
    )
    offset = (
        np.zeros(dof)
        if estimated_offset_rad is None
        else np.asarray(estimated_offset_rad, dtype=float)
    )
    if bias.shape != (dof,) or offset.shape != (dof,):
        raise ValueError("joint bias and estimated offset must match chain DOF")
    if not all(np.all(np.isfinite(value)) for value in (lower, upper, bias, offset)):
        raise ValueError("FK validation inputs must be finite")

    measured_q = np.random.default_rng(seed).uniform(lower, upper, (samples, dof))
    truth_q = measured_q - bias
    calibrated_q = measured_q + offset
    before_position, before_orientation = _errors(
        chains[0], chains[2], measured_q, truth_q
    )
    after_position, after_orientation = _errors(
        chains[1], chains[2], calibrated_q, truth_q
    )
    before = {
        "position_error_m": _statistics(before_position),
        "orientation_error_rad": _statistics(before_orientation),
    }
    after = {
        "position_error_m": _statistics(after_position),
        "orientation_error_rad": _statistics(after_orientation),
    }
    joint_geometry = _joint_geometry_comparison(*chains)
    known_inputs = _known_input_perturbations(paths[0], paths[2])

    def improvement(field: str) -> float:
        initial = before[field]["rms"]
        final = after[field]["rms"]
        return (
            1.0
            if initial == 0.0 and final == 0.0
            else (float("-inf") if initial == 0.0 else (initial - final) / initial)
        )

    return {
        "schema_version": 1,
        "report_type": "known-urdf-fk-validation",
        "ee_name": ee_name,
        "joint_names": [item["name"] for item in chains[0] if item["type"] != "fixed"],
        "samples": samples,
        "seed": seed,
        "sampling_domain": (
            "uniform per joint over ground-truth URDF limits"
            if sampling_domain == "joint-limits"
            else "uniform per joint over observed dataset bounds"
        ),
        "joint_min_rad": lower.tolist(),
        "joint_max_rad": upper.tolist(),
        "injected_bias_rad": bias.tolist(),
        "estimated_offset_rad": offset.tolist(),
        "conventions": {
            "ground_truth": "q_actual = q_measured - injected_bias_rad",
            "calibrated": "q_effective = q_measured + estimated_offset_rad",
        },
        "urdf": {
            "nominal": str(paths[0]),
            "calibrated": str(paths[1]),
            "ground_truth": str(paths[2]),
            "nominal_sha256": _sha256(paths[0]),
            "calibrated_sha256": _sha256(paths[1]),
            "ground_truth_sha256": _sha256(paths[2]),
        },
        "known_input_geometry_perturbations": known_inputs,
        "joint_geometry_comparison": joint_geometry,
        "joint_geometry_summary": _joint_geometry_summary(joint_geometry),
        "joint_geometry_interpretation": (
            "Local joint-origin comparison is diagnostic because serial-chain "
            "frame gauge can redistribute equivalent geometry. End-effector FK "
            "statistics are the gauge-invariant acceptance evidence."
        ),
        "before": before,
        "after": after,
        "rms_improvement": {
            "position": improvement("position_error_m"),
            "orientation": improvement("orientation_error_rad"),
        },
    }
