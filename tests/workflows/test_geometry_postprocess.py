import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.geometry_postprocess import postprocess_urdf_definitions


def test_postprocess_chooses_nominal_rpy_branch_and_reports_parallel_axes(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0" rpy="0 0 3.1"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
</robot>"""
    )
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0], "rpy": [0, 0, 3.1 - 2 * np.pi]},
        {"name": "a-b", "xyz": [0.1, 0, 0], "rpy": [0, 0, 0]},
    ]

    processed, report = postprocess_urdf_definitions(definitions, urdf)

    assert processed[0]["rpy"][2] == pytest.approx(3.1)
    assert report["transform_preserving"] is True
    relation = report["adjacent_axis_relations"][0]
    assert relation["relation"] == "parallel"
    assert relation["axis_distance_m"] == pytest.approx(0.1)
    before = Rotation.from_euler("xyz", definitions[0]["rpy"]).as_matrix()
    after = Rotation.from_euler("xyz", processed[0]["rpy"]).as_matrix()
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_postprocess_reports_intersecting_axes(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/></joint>
</robot>"""
    )
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0], "rpy": [0, 0, 0]},
        {"name": "a-b", "xyz": [0, 0, 0], "rpy": [0, 0, 0]},
    ]

    _, report = postprocess_urdf_definitions(definitions, urdf)

    relation = report["adjacent_axis_relations"][0]
    assert relation["relation"] == "intersecting"
    assert relation["axis_distance_m"] == pytest.approx(0.0)


def test_postprocess_snaps_numerical_noise_near_right_angles(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test"><link name="base"/><link name="a"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0" rpy="3.141592653589793 0 1.5707963267948966"/>
    <axis xyz="0 0 1"/></joint></robot>"""
    )
    definitions = [
        {
            "name": "base-a",
            "xyz": [0, 0, 0],
            "rpy": [np.pi - 2e-12, 3e-12, np.pi / 2 + 4e-12],
        }
    ]

    processed, _ = postprocess_urdf_definitions(definitions, urdf)

    np.testing.assert_array_equal(processed[0]["rpy"], [np.pi, 0.0, np.pi / 2])


def test_postprocess_does_not_snap_real_rotation_corrections(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test"><link name="base"/><link name="a"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint></robot>"""
    )
    definitions = [{"name": "base-a", "xyz": [0, 0, 0], "rpy": [0.0, 0.0, 2e-6]}]

    processed, _ = postprocess_urdf_definitions(definitions, urdf)

    assert processed[0]["rpy"][2] == pytest.approx(2e-6, abs=1e-12)


def test_postprocess_allocates_parallel_axis_translation_toward_nominal(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.1 0 0.2"/><axis xyz="0 0 1"/></joint>
</robot>"""
    )
    # Moving equal and opposite distances along two coincident Z directions is
    # a pure frame gauge and leaves FK unchanged for every joint position.
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0.01], "rpy": [0, 0, 0]},
        {"name": "a-b", "xyz": [0.1, 0, 0.19], "rpy": [0, 0, 0]},
    ]
    joint_positions = np.array([[0.0, 0.0], [0.3, -0.4], [-0.7, 0.8]])

    processed, report = postprocess_urdf_definitions(
        definitions, urdf, joint_positions, enable_nominal_allocation=True
    )

    allocation = report["allocation"]
    assert allocation["accepted"] is True
    assert allocation["fallback_to_raw_solver"] is False
    assert allocation["fk_position_max_m"] <= 1e-6
    assert allocation["fk_orientation_max_rad"] <= 1e-6
    assert processed[0]["xyz"][2] == pytest.approx(0.0, abs=1e-7)
    assert processed[1]["xyz"][2] == pytest.approx(0.2, abs=1e-7)


def test_postprocess_can_anchor_one_joint_to_nominal(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.1 0 0.2"/><axis xyz="0 0 1"/></joint>
</robot>"""
    )
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0.01], "rpy": [0, 0, 0]},
        {"name": "a-b", "xyz": [0.1, 0, 0.19], "rpy": [0, 0, 0]},
    ]
    joint_positions = np.array([[0.0, 0.0], [0.3, -0.4], [-0.7, 0.8]])

    processed, report = postprocess_urdf_definitions(
        definitions,
        urdf,
        joint_positions,
        anchor_joint="J1",
        enable_nominal_allocation=True,
    )

    assert report["allocation_policy"] == "nominal-constrained-se3-explicit-anchor"
    assert report["allocation"]["accepted"] is True
    assert report["allocation"]["anchor_joint"] == "J1"
    assert report["allocation"]["selection_mode"] == "explicit_anchor"
    np.testing.assert_allclose(processed[0]["xyz"], [0, 0, 0], atol=1e-8)
    assert processed[1]["xyz"][2] == pytest.approx(0.2, abs=1e-7)


def test_postprocess_rejects_unknown_anchor_joint(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test"><link name="base"/><link name="a"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint></robot>"""
    )

    with pytest.raises(ValueError, match="absent from the calibrated chain"):
        postprocess_urdf_definitions(
            [{"name": "base-a", "xyz": [0, 0, 0], "rpy": [0, 0, 0]}],
            urdf,
            np.array([[0.0]]),
            anchor_joint="missing",
            enable_nominal_allocation=True,
        )


def test_postprocess_does_not_automatically_select_an_anchor(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.1 0 0.2"/><axis xyz="0 0 1"/></joint>
</robot>"""
    )
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0.01], "rpy": [0, 0, 0]},
        {"name": "a-b", "xyz": [0.1, 0, 0.19], "rpy": [0, 0, 0]},
    ]

    processed, report = postprocess_urdf_definitions(
        definitions,
        urdf,
        np.array([[0.0, 0.0], [0.3, -0.4], [-0.7, 0.8]]),
    )

    allocation = report["allocation"]
    assert report["allocation_policy"] == "raw-solver-rpy-normalized"
    assert allocation["selection_mode"] == "raw_solver"
    assert allocation["anchor_joint"] is None
    assert allocation["attempted"] is False
    assert allocation["accepted"] is False
    assert allocation["reason"] == "nominal allocation is disabled by default"
    np.testing.assert_allclose(processed[0]["xyz"], definitions[0]["xyz"])
    np.testing.assert_allclose(processed[1]["xyz"], definitions[1]["xyz"])
    verification = report["equivalence_verification"]
    assert verification["verified"] is True
    assert verification["dataset_samples"] == 3
    assert verification["random_samples"] == 2048
    assert verification["total_samples"] == 2051
    assert verification["fk_all_links_position_max_m"] <= 1e-12
    assert verification["fk_all_links_orientation_max_rad"] <= 1e-9
    assert report["nominal_urdf_comparison"]["purpose"].startswith(
        "calibration-change magnitude"
    )


def test_postprocess_svd_projects_gauge_toward_nominal(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="J1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="J2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.1 0 0.2"/><axis xyz="0 0 1"/></joint>
</robot>"""
    )
    definitions = [
        {"name": "base-a", "xyz": [0, 0, 0.01], "rpy": [0, 0, 0]},
        {"name": "a-b", "xyz": [0.1, 0, 0.19], "rpy": [0, 0, 0]},
    ]
    joints = np.array([[0.0, 0.0], [0.3, -0.4], [-0.7, 0.8]])

    processed, report = postprocess_urdf_definitions(
        definitions,
        urdf,
        joints,
        enable_svd_gauge_allocation=True,
    )

    allocation = report["allocation"]
    assert report["allocation_policy"] == "svd-identifiable-nominal-gauge"
    assert allocation["accepted"] is True
    assert allocation["equivalence_scope"] == "end_effector_only"
    assert allocation["gauge_dimension"] > 0
    assert allocation["nominal_distance_final"] < allocation["nominal_distance_initial"]
    assert allocation["fk_position_max_m"] <= 1e-6
    assert processed != definitions
    verification = report["equivalence_verification"]
    assert verification["verified"] is True
    assert verification["random_samples"] == 2048
    assert verification["scope"].startswith("Base_T_EE only")
    assert verification["fk_end_effector_position_max_m"] <= 1e-6
    comparisons = report["joint_parameter_comparison"]
    assert [item["joint_name"] for item in comparisons] == ["J1", "J2"]
    assert comparisons[0]["raw_solver_to_selected"]["translation_norm_m"] > 0
    coupling = report["adjacent_parameter_coupling"]
    assert coupling[0]["joint_pair"] == ["J1", "J2"]
