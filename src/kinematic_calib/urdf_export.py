"""Safe export of accepted calibration results into a new URDF."""

import hashlib
import io
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_vector(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _validated_vector(values, field: str, identifier: str) -> list[float]:
    """Return a finite three-vector from an untrusted JSON report."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(
            f"calibrated joint {identifier!r} {field} must contain 3 values"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"calibrated joint {identifier!r} {field} contains non-finite values"
        )
    return vector.tolist()


def _load_tree(path: Path) -> ET.ElementTree:
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        return ET.parse(path, parser=parser)
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML in {path}: {exc}") from exc


def _joint_elements(root: ET.Element):
    by_name = {}
    by_relation = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if name is None or parent is None or child is None:
            raise ValueError("URDF contains a joint without name, parent, or child")
        parent_name = parent.get("link")
        child_name = child.get("link")
        if parent_name is None or child_name is None:
            raise ValueError(f"URDF joint {name!r} has invalid parent/child metadata")
        by_name[name] = joint
        by_relation[f"{parent_name}-{child_name}"] = joint
    return by_name, by_relation


def _rewrite_relative_resources(
    root: ET.Element, source_dir: Path, output_dir: Path
) -> int:
    """Rebase relative mesh/texture paths for the exported URDF location."""
    rewritten = 0
    for element in [*root.iter("mesh"), *root.iter("texture")]:
        filename = element.get("filename")
        if not filename or "://" in filename or Path(filename).is_absolute():
            continue
        resource = (source_dir / filename).resolve()
        rebased = Path(os.path.relpath(resource, output_dir.resolve())).as_posix()
        element.set("filename", rebased)
        rewritten += 1
    return rewritten


def export_calibrated_urdf(
    result_path: Path | str,
    source_urdf: Path | str,
    output_urdf: Path | str,
    *,
    force: bool = False,
    allow_partial: bool = False,
) -> dict:
    """Apply an accepted calibration report to a new URDF file."""
    result_path = Path(result_path)
    source_urdf = Path(source_urdf)
    output_urdf = Path(output_urdf)
    if result_path.suffix.lower() != ".json":
        raise ValueError("calibration result must be a .json file")
    if source_urdf.suffix.lower() != ".urdf":
        raise ValueError("source robot model must be a .urdf file")
    if output_urdf.suffix.lower() != ".urdf":
        raise ValueError("output robot model must use the .urdf extension")
    if source_urdf.resolve() == output_urdf.resolve():
        raise ValueError("refusing to overwrite the source URDF")
    if output_urdf.exists() and not force:
        raise ValueError(f"output URDF already exists: {output_urdf}; use --force")

    report = json.loads(result_path.read_text())
    schema_version = report.get("schema_version")
    if schema_version not in (None, 1):
        raise ValueError(f"unsupported calibration schema version: {schema_version!r}")
    if report.get("accepted") is not True:
        raise ValueError("calibration report is not accepted; refusing URDF export")
    if (
        report.get("full_urdf_export_supported") is False
        and report.get("partial_urdf_export_supported") is False
    ):
        reason = report.get(
            "full_urdf_export_disabled_reason", "calibrated link frames are unavailable"
        )
        raise ValueError(f"URDF export unavailable: {reason}")
    expected_hash = report.get("urdf_sha256")
    actual_hash = _sha256(source_urdf)
    if expected_hash != actual_hash:
        raise ValueError("source URDF SHA-256 does not match the calibration report")
    calibration_type = report.get("calibration_type")
    if calibration_type == "joint-zero-offset-only":
        raise ValueError(
            "zero-offset results are runtime joint corrections and cannot be "
            "exported as URDF geometry"
        )
    if calibration_type not in (None, "kinematic-geometry"):
        raise ValueError(f"unsupported calibration result type: {calibration_type!r}")

    tree = _load_tree(source_urdf)
    root = tree.getroot()
    by_name, by_relation = _joint_elements(root)
    mode = report.get("model_mode")
    if mode == "self-calibrating":
        if not allow_partial:
            raise ValueError(
                "self-calibrating export is only partially identifiable; "
                "use --allow-partial to acknowledge that the exported URDF "
                "is not equivalent to the full composite model"
            )
        definitions = report.get("identifiable_urdf_joint_definitions")
        if not definitions:
            raise ValueError("self-calibrating report has no identifiable joints")
        entries = [
            (definition["joint_name"], definition["xyz_m"], definition["rpy_rad"])
            for definition in definitions
        ]
        lookup = by_name
        export_scope = "identifiable-joints-only"
        export_warning = (
            "Joint1 and the fixed terminal transform remain nominal; this "
            "partial URDF is not equivalent to the full Camera/Target model"
        )
    elif mode == "robot-body":
        definitions = report.get("urdf_joint_definitions")
        if not definitions:
            raise ValueError("robot-body report has no URDF joint definitions")
        entries = [
            (definition["name"], definition["xyz"], definition["rpy"])
            for definition in definitions
        ]
        lookup = by_relation
        export_scope = "full-selected-chain"
        export_warning = None
    else:
        raise ValueError(f"unsupported calibration model mode: {mode!r}")

    identifiers = [identifier for identifier, _, _ in entries]
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(
            "calibration report contains duplicate joint definitions: "
            + ", ".join(repr(identifier) for identifier in duplicates)
        )

    updated = []
    for identifier, xyz, rpy in entries:
        joint = lookup.get(identifier)
        if joint is None:
            raise ValueError(f"calibrated joint {identifier!r} is absent from URDF")
        xyz = _validated_vector(xyz, "xyz", identifier)
        rpy = _validated_vector(rpy, "rpy", identifier)
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", _format_vector(xyz))
        origin.set("rpy", _format_vector(rpy))
        updated.append(joint.attrib["name"])

    rewritten_resources = _rewrite_relative_resources(
        root, source_urdf.parent, output_urdf.parent
    )
    ET.indent(tree, space="  ")
    rendered = io.BytesIO()
    tree.write(rendered, encoding="utf-8", xml_declaration=True)
    atomic_write_bytes(output_urdf, rendered.getvalue())
    _load_tree(output_urdf)
    return {
        "output_urdf": str(output_urdf),
        "model_mode": mode,
        "export_scope": export_scope,
        "export_warning": export_warning,
        "updated_joint_names": updated,
        "rewritten_relative_resources": rewritten_resources,
        "resource_path_policy": "relative-to-output-urdf",
        "source_urdf_sha256": actual_hash,
        "output_urdf_sha256": _sha256(output_urdf),
    }
