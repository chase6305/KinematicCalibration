"""Geometry presentation against known URDF frame corrections."""

import copy
import json

import numpy as np

from kinematic_calib.reporting import summarize_geometry


def _stage():
    return {
        "accepted": True,
        "exported_urdf": None,
        "parameters": "geometry_parameters.npz",
        "export_skipped_reason": "POE model has no calibrated link frames",
    }


def test_poe_summary_keeps_metrics_and_artifact_with_empty_joint_arrays(tmp_path):
    report = {
        "position_error_mean_m": 0.001,
        "position_error_max_m": 0.002,
        "orientation_error_mean_rad": 0.003,
        "orientation_error_max_rad": 0.004,
        "urdf_joint_definitions": None,
        "identifiable_urdf_joint_definitions": None,
    }
    # POE-only reporting needs no physical URDF origins and must not read a file.
    result = summarize_geometry(report, tmp_path / "unavailable.urdf", _stage())
    assert result["accepted"]
    assert result["position_error_mean_m"] == 0.001
    assert result["position_error_max_m"] == 0.002
    assert result["orientation_error_mean_rad"] == 0.003
    assert result["orientation_error_max_rad"] == 0.004
    for field in (
        "joint_names",
        "joint_indices",
        "joint_types",
        "axis",
        "xyz_rpy",
        "delta_xyz_rpy",
    ):
        assert result[field] == []
    assert result["parameters"] == "geometry_parameters.npz"
    assert result["urdf"] is None
    assert result["export_skipped_reason"] == _stage()["export_skipped_reason"]
    assert result["xyz_rpy_units"] == ["m", "m", "m", "rad", "rad", "rad"]
    json.dumps(result, allow_nan=False)


def test_full_origins_report_local_corrections_and_continuous_joint_indices(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot name="test">
    <link name="base"/><link name="arm"/><link name="spacer"/><link name="tool"/>
    <joint name="spin" type="continuous">
      <parent link="base"/><child link="arm"/>
      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/>
    </joint>
    <joint name="spacer" type="fixed">
      <parent link="arm"/><child link="spacer"/><origin xyz="0 0 0.1"/>
    </joint>
    <joint name="hinge" type="revolute">
      <parent link="spacer"/><child link="tool"/>
      <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1"/>
    </joint></robot>""")
    report = {
        "urdf_joint_definitions": [
            {"name": "spacer-tool", "xyz": [0.2, 0, 0], "rpy": [0, 0, 0]},
            {"name": "arm-spacer", "xyz": [0, 0, 0.1], "rpy": [0, 0, 0]},
            {"name": "base-arm", "xyz": [0.8, 2.1, 3.3], "rpy": [0.2, 0, np.pi / 2]},
        ],
        "geometry_postprocessing": {"method": "nominal-frame"},
    }
    original = copy.deepcopy(report)
    stage = {
        **_stage(),
        "exported_urdf": "calibrated.urdf",
        "export_skipped_reason": None,
    }

    result = summarize_geometry(report, source, stage)

    assert result["joint_names"] == ["hinge", "spacer", "spin"]
    assert result["joint_indices"] == [1, None, 0]
    assert result["joint_types"] == ["revolute", "fixed", "continuous"]
    assert result["axis"] == [[0, 1, 0], None, [0, 0, 1]]
    # A 90-degree nominal Z rotation maps local (0.1,0.2,0.3) into (-0.2,0.1,0.3).
    # The calibrated RPY composes this nominal rotation with a local 0.2 rad X turn.
    np.testing.assert_allclose(
        result["delta_xyz_rpy"][-1], [0.1, 0.2, 0.3, 0.2, 0, 0], atol=1e-14
    )
    np.testing.assert_allclose(
        result["delta_xyz_rpy"][:2], np.zeros((2, 6)), atol=1e-14
    )
    np.testing.assert_allclose(
        result["xyz_rpy"][-1], [0.8, 2.1, 3.3, 0.2, 0, np.pi / 2]
    )
    assert result["postprocessing"] == {"method": "nominal-frame"}
    assert result["urdf"] == "calibrated.urdf"
    assert report == original


def test_identifiable_definitions_keep_their_metadata_and_missing_corrections(tmp_path):
    definition = {
        "joint_name": "joint_2",
        "zero_based_index": 1,
        "joint_type": "revolute",
        "axis": [0, 1, 0],
        "xyz_m": [0.2, 0.3, 0.4],
        "rpy_rad": [0, 0, 0.1],
        "delta_xyz_m": None,
        "delta_rpy_rad": None,
    }
    report = {
        "urdf_joint_definitions": None,
        "identifiable_urdf_joint_definitions": [definition],
    }
    original = copy.deepcopy(report)

    result = summarize_geometry(report, tmp_path / "unused.urdf", _stage())

    assert result["joint_names"] == ["joint_2"]
    assert result["joint_indices"] == [1]
    assert result["xyz_rpy"] == [[0.2, 0.3, 0.4, 0, 0, 0.1]]
    assert result["delta_xyz_rpy"] == [None]
    assert report == original
