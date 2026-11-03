"""Replay measured datasets against a bound POE fit, without a simulation engine."""

import hashlib
import io
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    _validate_transforms,
    sample_sha256,
)
from kinematic_calib.saved_model import load_fitted_model
from kinematic_calib.validation import predict_saved_poe

from .numpy import NumpyBackend
from .playback import PoseState


def _bound_transform(path, report, hash_field):
    content = Path(path).read_bytes()
    if report.get(hash_field) != hashlib.sha256(content).hexdigest():
        raise ValueError(f"fixed transform {hash_field} does not match the fitted pair")
    value = np.load(io.BytesIO(content), allow_pickle=False)
    return _validate_transforms(hash_field, np.asarray(value)[None])[0]


def _measurement_role(dataset, dataset_hash, fitted):
    report = fitted.report
    hashes = sample_sha256(dataset)
    hash_overlap = np.isin(hashes, fitted.source_sample_sha256)
    id_overlap = np.isin(dataset.sample_indices, fitted.source_sample_indices)
    if dataset_hash == report["dataset_sha256"]:
        if not np.array_equal(
            dataset.sample_indices, fitted.source_sample_indices
        ) or not np.array_equal(hashes, fitted.source_sample_sha256):
            raise ValueError(
                "training dataset does not match the fitted source sample table"
            )
        role = "training measurements"
    elif hash_overlap.any() or id_overlap.any():
        role = "measurements with training overlap"
    elif dataset.observation_frame.value != report["source_observation_frame"]:
        role = "measurements (independence unverified)"
    else:
        role = "held-out measurements"
    return {
        "data_role": role,
        "training_sample_overlap_count": int(np.count_nonzero(hash_overlap)),
        "training_id_overlap_count": int(np.count_nonzero(id_overlap)),
    }


def _metrics(predicted, measured):
    position_errors = np.linalg.norm(predicted[:, :3, 3] - measured[:, :3, 3], axis=1)
    rotations = predicted[:, :3, :3].transpose(0, 2, 1) @ measured[:, :3, :3]
    orientation_errors = Rotation.from_matrix(rotations).magnitude()
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation_errors**2))),
    }


class MeasuredPlayback:
    """Display a measured dataset in the fitted model's observation coordinates.

    A robot-body fit uses Base/EE coordinates. Camera observations require the
    same fixed extrinsic files bound to that fit before conversion. A composite
    self-calibrating fit uses Camera/Target coordinates; its nominal skeleton
    uses initialization gauges, without inventing fitted physical link frames.
    """

    def __init__(
        self,
        parameters,
        dataset,
        urdf,
        *,
        result=None,
        base_t_camera=None,
        ee_t_target=None,
    ):
        fitted = load_fitted_model(parameters, urdf, result_path=result)
        source = KinematicDataset.load(dataset)
        self.nominal = NumpyBackend.from_urdf(
            urdf,
            end_link=fitted.report["ee_name"],
            base_link=fitted.report["base_link"],
        )
        if source.joint_positions.shape[1] != self.nominal.dof:
            raise ValueError("dataset joint columns do not match the fitted chain")
        identity = source.check_joint_identity(self.nominal.joint_names)
        limits = self.nominal.joint_limits
        outside = np.any(
            (source.joint_positions < limits[:, 0])
            | (source.joint_positions > limits[:, 1]),
            axis=1,
        )
        if outside.any():
            sample = int(source.sample_indices[np.flatnonzero(outside)[0]])
            raise ValueError(f"dataset sample {sample} is outside nominal joint limits")

        self._display_base = np.eye(4)
        self._nominal_target = np.eye(4)
        note = None
        if fitted.report["model_mode"] == "self-calibrating":
            if base_t_camera is not None or ee_t_target is not None:
                raise ValueError(
                    "self-calibrating replay does not accept fixed extrinsics"
                )
            if source.observation_frame != ObservationFrame.CAMERA_T_TARGET:
                raise ValueError(
                    "self-calibrating replay requires camera_T_target data"
                )
            if (
                fitted.camera_t_base_initialization is None
                or fitted.ee_t_target_initialization is None
            ):
                raise ValueError(
                    "self-calibrating replay requires initialization gauges"
                )
            self._display_base = fitted.camera_t_base_initialization.copy()
            self._nominal_target = fitted.ee_t_target_initialization.copy()
            displayed = source
            coordinate_frame = "camera"
            note = (
                "Nominal skeleton and target use initialization Camera_T_Base and "
                "EE_T_Target gauges, not calibrated physical extrinsics."
            )
        elif source.observation_frame == ObservationFrame.CAMERA_T_TARGET:
            if base_t_camera is None or ee_t_target is None:
                raise ValueError("camera replay requires both fixed extrinsic files")
            camera = _bound_transform(
                base_t_camera, fitted.report, "base_T_camera_sha256"
            )
            target = _bound_transform(ee_t_target, fitted.report, "ee_T_target_sha256")
            displayed = source.as_robot_body(camera, target)
            coordinate_frame = "base"
        else:
            if base_t_camera is not None or ee_t_target is not None:
                raise ValueError("base_T_ee replay does not accept fixed extrinsics")
            displayed = source
            coordinate_frame = "base"

        self.dataset = displayed
        self._fitted_axes = fitted.screw_axes
        self._fitted_home = fitted.zero_pose
        configurations = self.dataset.joint_positions
        nominal = np.asarray([self._nominal_pose(row) for row in configurations])
        calibrated = predict_saved_poe(
            self._fitted_axes, self._fitted_home, configurations
        )
        self.model = {
            "nominal_predictions": nominal,
            "calibrated_predictions": calibrated,
        }
        if coordinate_frame == "camera":
            self.model["camera_T_base_initialization"] = self._display_base.copy()
        self.report = {
            **fitted.report,
            **_measurement_role(
                source, hashlib.sha256(Path(dataset).read_bytes()).hexdigest(), fitted
            ),
            "coordinate_frame": coordinate_frame,
            "display_note": note,
            "replay_joint_identity_verified": identity["verified"],
            "nominal_vs_measurements": _metrics(nominal, displayed.observed_poses),
            "calibrated_vs_measurements": _metrics(
                calibrated, displayed.observed_poses
            ),
        }

    def _nominal_pose(self, configuration):
        return (
            self._display_base
            @ self.nominal.forward(configuration)
            @ self._nominal_target
        )

    def _state(self, configuration, *, index):
        raw = np.asarray(configuration)
        if raw.dtype.kind not in "iuf" or raw.shape != (self.nominal.dof,):
            raise ValueError("joint configuration must be a finite real [DOF] vector")
        with np.errstate(over="ignore"):
            positions = raw.astype(float, copy=True)
        if not np.isfinite(positions).all():
            raise ValueError("joint configuration must be a finite real [DOF] vector")
        limits = self.nominal.joint_limits
        if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
            raise ValueError("joint configuration is outside nominal joint limits")
        links = {
            name: self._display_base @ pose
            for name, pose in self.nominal.link_transforms(positions).items()
        }
        poses = {
            "nominal": links[self.nominal.end_link] @ self._nominal_target,
            "calibrated": predict_saved_poe(
                self._fitted_axes, self._fitted_home, positions[None]
            )[0],
        }
        measured = None if index is None else self.dataset.observed_poses[index].copy()
        position_error = orientation_error = None
        if measured is not None:
            predicted = poses["calibrated"]
            position_error = float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
            orientation_error = float(
                Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude()
            )
        return PoseState(
            positions,
            {"nominal": links},
            poses,
            measured,
            position_error,
            orientation_error,
            "no measurement" if index is None else self.report["data_role"],
            None if index is None else int(self.dataset.sample_indices[index]),
        )

    def sample(self, index: int) -> PoseState:
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < len(self.dataset.joint_positions)
        ):
            raise ValueError("sample index must identify a measured dataset row")
        return self._state(self.dataset.joint_positions[index], index=int(index))

    def manual(self, joint_positions) -> PoseState:
        return self._state(joint_positions, index=None)
