"""Presentation of fitted geometry in the unified calibration report."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _joint_metadata(source_urdf: Path | str) -> dict:
    metadata = {}
    joint_index = 0
    for element in ET.parse(source_urdf).getroot().findall("joint"):
        parent, child = element.find("parent"), element.find("child")
        if parent is None or child is None:
            continue
        joint_type = element.get("type")
        rotating = joint_type in {"revolute", "continuous"}
        origin = element.find("origin")
        xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
        rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
        axis = element.find("axis")
        relation = f"{parent.get('link')}-{child.get('link')}"
        metadata[relation] = {
            "zero_based_index": joint_index if rotating else None,
            "joint_name": element.get("name"),
            "joint_type": joint_type,
            "axis": (
                np.fromstring(axis.get("xyz", "1 0 0"), sep=" ").tolist()
                if axis is not None
                else None
            ),
            "nominal_xyz": np.fromstring(xyz, sep=" "),
            "nominal_rpy": np.fromstring(rpy, sep=" "),
        }
        if rotating:
            joint_index += 1
    return metadata


def _joint_definitions(report: dict, source_urdf: Path | str) -> list:
    definitions = report.get("urdf_joint_definitions")
    if definitions is None:
        return report.get("identifiable_urdf_joint_definitions") or []
    if not definitions or "name" not in definitions[0]:
        return definitions

    from scipy.spatial.transform import Rotation

    metadata = _joint_metadata(source_urdf)
    normalized = []
    for definition in definitions:
        joint = metadata.get(definition["name"], {})
        xyz = np.asarray(definition["xyz"], dtype=float)
        rpy = np.asarray(definition["rpy"], dtype=float)
        nominal_xyz = joint.get("nominal_xyz", np.zeros(3))
        nominal_rotation = Rotation.from_euler(
            "xyz", joint.get("nominal_rpy", np.zeros(3))
        ).as_matrix()
        calibrated_rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        # Express the calibrated origin correction in the nominal joint frame.
        delta_xyz = nominal_rotation.T @ (xyz - nominal_xyz)
        delta_rpy = Rotation.from_matrix(
            nominal_rotation.T @ calibrated_rotation
        ).as_euler("xyz")
        normalized.append(
            {
                **{
                    key: joint.get(key)
                    for key in ("zero_based_index", "joint_name", "joint_type", "axis")
                },
                "xyz_m": xyz.tolist(),
                "rpy_rad": rpy.tolist(),
                "delta_xyz_m": delta_xyz.tolist(),
                "delta_rpy_rad": delta_rpy.tolist(),
            }
        )
    return normalized


def _joint_vector(joint: dict, translation_key: str, rotation_key: str):
    translation, rotation = joint.get(translation_key), joint.get(rotation_key)
    if translation is None or rotation is None:
        return None
    return [*translation, *rotation]


def summarize_geometry(report: dict, source_urdf: Path | str, stage: dict) -> dict:
    """Build the concise geometry result without changing fit or export state.

    Full URDF origins are converted to joint metadata and local corrections;
    already-normalized identifiable definitions pass through. A POE-only result
    has empty joint-definition arrays and retains its parameter artifact path.
    """
    definitions = _joint_definitions(report, source_urdf)
    return {
        "accepted": stage["accepted"],
        "position_error_mean_m": report.get("position_error_mean_m"),
        "position_error_max_m": report.get("position_error_max_m"),
        "orientation_error_mean_rad": report.get("orientation_error_mean_rad"),
        "orientation_error_max_rad": report.get("orientation_error_max_rad"),
        "joint_names": [joint.get("joint_name") for joint in definitions],
        "joint_indices": [joint.get("zero_based_index") for joint in definitions],
        "joint_types": [joint.get("joint_type") for joint in definitions],
        "xyz_rpy": [_joint_vector(joint, "xyz_m", "rpy_rad") for joint in definitions],
        "xyz_rpy_units": ["m", "m", "m", "rad", "rad", "rad"],
        "axis": [joint.get("axis") for joint in definitions],
        "delta_xyz_rpy": [
            _joint_vector(joint, "delta_xyz_m", "delta_rpy_rad")
            for joint in definitions
        ],
        "postprocessing": report.get("geometry_postprocessing"),
        "urdf": stage["exported_urdf"],
        "parameters": stage["parameters"],
        "export_skipped_reason": stage["export_skipped_reason"],
    }
