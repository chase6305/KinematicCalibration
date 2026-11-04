import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from kinematic_calib.urdf_export import export_calibrated_urdf


def _write_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="test">
  <link name="base"><visual><geometry><mesh filename="mesh.stl"/></geometry></visual></link>
  <link name="link1"/><link name="link2"/>
  <joint name="Joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="Joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>
"""
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_self_calibrating_export_updates_only_identifiable_joints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "robot.urdf"
    source.parent.mkdir()
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
                "identifiable_urdf_joint_definitions": [
                    {
                        "joint_name": "Joint2",
                        "xyz_m": [0.01, 0.02, 0.3],
                        "rpy_rad": [0.001, 0.002, 0.003],
                    }
                ],
            }
        )
    )
    output = tmp_path / "export" / "calibrated.urdf"

    report = export_calibrated_urdf(result, source, output, allow_partial=True)

    root = ET.parse(output).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    assert joints["Joint1"].find("origin").attrib["xyz"] == "0 0 0.1"
    assert joints["Joint2"].find("origin").attrib["xyz"] == "0.01 0.02 0.3"
    mesh_filename = root.find(".//mesh").attrib["filename"]
    assert not Path(mesh_filename).is_absolute()
    assert (output.parent / mesh_filename).resolve() == (
        source.parent / "mesh.stl"
    ).resolve()
    assert report["updated_joint_names"] == ["Joint2"]
    assert report["export_scope"] == "identifiable-joints-only"
    assert report["export_warning"] is not None
    assert report["resource_path_policy"] == "relative-to-output-urdf"


def test_export_rejects_unaccepted_result(tmp_path: Path) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": False,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
            }
        )
    )

    with pytest.raises(ValueError, match="not accepted"):
        export_calibrated_urdf(result, source, tmp_path / "output.urdf")


def test_export_in_source_directory_preserves_resource_filename(tmp_path: Path) -> None:
    source = tmp_path / "UR10.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
                "identifiable_urdf_joint_definitions": [
                    {
                        "joint_name": "Joint2",
                        "xyz_m": [0.0, 0.0, 0.2],
                        "rpy_rad": [0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    )
    output = tmp_path / "UR10_self_calibrated.urdf"

    report = export_calibrated_urdf(result, source, output, allow_partial=True)

    assert ET.parse(output).find(".//mesh").attrib["filename"] == "mesh.stl"
    assert report["resource_path_policy"] == "relative-to-output-urdf"


def test_self_calibrating_export_requires_partial_acknowledgement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
                "identifiable_urdf_joint_definitions": [
                    {
                        "joint_name": "Joint2",
                        "xyz_m": [0.0, 0.0, 0.2],
                        "rpy_rad": [0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="allow-partial"):
        export_calibrated_urdf(result, source, tmp_path / "output.urdf")


def test_export_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": "0" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="SHA-256"):
        export_calibrated_urdf(result, source, tmp_path / "output.urdf")


@pytest.mark.parametrize(
    ("xyz", "message"),
    [([0.0, 0.0], "must contain 3"), ([0.0, float("nan"), 0.0], "non-finite")],
)
def test_export_rejects_invalid_joint_vectors(
    tmp_path: Path, xyz: list[float], message: str
) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
                "identifiable_urdf_joint_definitions": [
                    {
                        "joint_name": "Joint2",
                        "xyz_m": xyz,
                        "rpy_rad": [0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        export_calibrated_urdf(
            result, source, tmp_path / "output.urdf", allow_partial=True
        )


def test_export_rejects_duplicate_joint_definitions(tmp_path: Path) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    definition = {
        "joint_name": "Joint2",
        "xyz_m": [0.0, 0.0, 0.2],
        "rpy_rad": [0.0, 0.0, 0.0],
    }
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "accepted": True,
                "model_mode": "self-calibrating",
                "urdf_sha256": _sha256(source),
                "identifiable_urdf_joint_definitions": [definition, definition],
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate joint"):
        export_calibrated_urdf(
            result, source, tmp_path / "output.urdf", allow_partial=True
        )


def test_export_rejects_unknown_schema_version(tmp_path: Path) -> None:
    source = tmp_path / "robot.urdf"
    _write_urdf(source)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"schema_version": 999, "accepted": True}))

    with pytest.raises(ValueError, match="schema version"):
        export_calibrated_urdf(result, source, tmp_path / "output.urdf")
