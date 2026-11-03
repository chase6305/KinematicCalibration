import numpy as np

from kinematic_calib.dataset import KinematicDataset, ObservationFrame


def test_dataset_quality_reports_weak_joint_and_pose_coverage() -> None:
    joints = np.array([[0.0, 0.0], [0.2, 0.01], [0.4, 0.02]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    dataset = KinematicDataset(joints, poses, ObservationFrame.CAMERA_T_TARGET)

    warnings = dataset.get_quality_warnings()

    assert any("joints: 1" in warning for warning in warnings)
    assert "target translation coverage is below threshold" in warnings
    assert "target rotation coverage is below threshold" in warnings


def test_dataset_quality_identifies_duplicate_source_samples() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[2, 0, 3] = 0.2
    dataset = KinematicDataset(
        np.array([[0.0], [0.0], [0.2]]),
        poses,
        ObservationFrame.CAMERA_T_TARGET,
        sample_indices=np.array([4, 9, 15]),
    )

    summary = dataset.get_summary()

    assert summary["unique_sample_count"] == 2
    assert summary["duplicate_sample_count"] == 1
    assert summary["duplicate_sample_positions"] == [1]
    assert summary["duplicate_of_positions"] == [0]
    assert summary["duplicate_source_sample_indices"] == [9]
    assert any(
        "source sample indices: 9" in item for item in dataset.get_quality_warnings()
    )
