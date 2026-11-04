"""Gaussian-sampling validation for SVD nominal-gauge allocation."""

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.geometry_postprocess import postprocess_urdf_definitions


def _transform(xyz, rpy):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def _forward_kinematics(definitions, joint_positions):
    pose = np.eye(4)
    for definition, position in zip(definitions, joint_positions, strict=True):
        pose = pose @ _transform(definition["xyz"], definition["rpy"])
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec(
            np.array([0.0, 0.0, position])
        ).as_matrix()
        pose = pose @ motion
    return pose


def _pose_error(actual, expected):
    delta = np.linalg.inv(expected) @ actual
    return (
        np.linalg.norm(delta[:3, 3]),
        np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()),
    )


def test_svd_gauge_allocation_on_independent_gaussian_joint_samples(tmp_path):
    """Gauge projection must generalize beyond its Gaussian training poses."""
    urdf = tmp_path / "gaussian_validation_robot.urdf"
    urdf.write_text(
        """<robot name="gaussian-validation">
  <link name="base"/><link name="link1"/><link name="link2"/>
  <joint name="Joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="Joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.35 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>"""
    )

    # Equal and opposite displacement along two parallel joint axes is a
    # coordinate gauge: it changes the two local origins but not Base_T_Link2.
    raw_definitions = [
        {"name": "base-link1", "xyz": [0.0, 0.0, 0.0008], "rpy": [0, 0, 0]},
        {
            "name": "link1-link2",
            "xyz": [0.35, 0.0, 0.1992],
            "rpy": [0, 0, 0],
        },
    ]

    rng = np.random.default_rng(20260818)
    training_positions = rng.normal(
        loc=np.array([0.15, -0.20]),
        scale=np.array([0.45, 0.35]),
        size=(128, 2),
    )
    training_positions = np.clip(training_positions, -1.2, 1.2)

    selected, report = postprocess_urdf_definitions(
        raw_definitions,
        urdf,
        training_positions,
        enable_svd_gauge_allocation=True,
    )

    allocation = report["allocation"]
    assert allocation["accepted"] is True
    assert allocation["gauge_dimension"] > 0
    assert allocation["nominal_distance_final"] < allocation["nominal_distance_initial"]

    # This set is independent of both the training data and the postprocessor's
    # deterministic uniform verification samples.
    validation_positions = rng.normal(
        loc=np.array([0.15, -0.20]),
        scale=np.array([0.45, 0.35]),
        size=(10_000, 2),
    )
    validation_positions = np.clip(validation_positions, -1.2, 1.2)
    errors = np.asarray(
        [
            _pose_error(
                _forward_kinematics(selected, q),
                _forward_kinematics(raw_definitions, q),
            )
            for q in validation_positions
        ]
    )

    assert errors[:, 0].max() <= 1e-6
    assert errors[:, 1].max() <= 1e-6
    assert np.quantile(errors[:, 0], 0.999) <= 1e-6
    assert np.quantile(errors[:, 1], 0.999) <= 1e-6
