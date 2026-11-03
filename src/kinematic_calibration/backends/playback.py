"""Verified replay and manual geometry exploration, independent of a viewer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import KinematicDataset

from ..artifacts import load_experiment_bundle
from .numpy import NumpyBackend


@dataclass(frozen=True)
class PoseState:
    joint_positions: np.ndarray
    links: dict[str, dict[str, np.ndarray]]
    poses: dict[str, np.ndarray]
    measured_pose: np.ndarray | None
    position_error_m: float | None
    orientation_error_rad: float | None
    error_reference: str
    sample_index: int | None


class ExperimentPlayback:
    """Read an experiment once, then replay samples or evaluate manual joints.

    Validation is the default; training replay retains its own measurement
    labels and metrics and does not constitute held-out validation.
    Manual configurations have no sensor observation. Their errors are measured
    against the synthetic truth model, and cannot extend held-out validation.
    """

    def __init__(self, path: str | Path, *, split: str = "validation"):
        if not isinstance(split, str) or split not in ("validation", "training"):
            raise ValueError("split must be 'validation' or 'training'")

        from robotkinecal import SerialRobotKineCal

        self.path = Path(path)
        self.split = split
        self.report, self.dataset, self.model = load_experiment_bundle(path)
        self.nominal = NumpyBackend.from_urdf(
            self.path / "nominal.urdf",
            end_link=self.report["end_link"],
            base_link=self.report["base_link"],
        )
        self.truth = self.nominal.perturbed(seed=self.report["truth_seed"])
        if split == "training":
            from ..experiment import pose_metrics, predict_poe

            # Verify the complete bundle before selecting archived training rows.
            self.dataset = KinematicDataset.load(self.path / "train.npz")
            self.model = self.model.copy()
            self.report = self.report.copy()
            joints = self.dataset.joint_positions
            for name, (axes, home) in (
                ("nominal", self.nominal.poe_parameters()),
                ("truth", self.truth.poe_parameters()),
                ("calibrated", (self.model["screw_axes"], self.model["home"])),
            ):
                self.model[f"{name}_predictions"] = predict_poe(axes, home, joints)
            self.report["data_role"] = "training"
            for key, predicted, observed in (
                (
                    "nominal_vs_measurements",
                    self.model["nominal_predictions"],
                    self.dataset.observed_poses,
                ),
                (
                    "calibrated_vs_measurements",
                    self.model["calibrated_predictions"],
                    self.dataset.observed_poses,
                ),
                (
                    "calibrated_vs_truth",
                    self.model["calibrated_predictions"],
                    self.model["truth_predictions"],
                ),
                (
                    "nominal_vs_truth",
                    self.model["nominal_predictions"],
                    self.model["truth_predictions"],
                ),
                (
                    "measurement_vs_truth",
                    self.dataset.observed_poses,
                    self.model["truth_predictions"],
                ),
            ):
                if key in self.report:
                    self.report[key] = pose_metrics(predicted, observed)
        self.fitted = SerialRobotKineCal.from_screw_axes(
            self.model["screw_axes"],
            self.model["home"],
            joint_names=self.nominal.joint_names,
        )

    def _state(self, q, *, index: int | None) -> PoseState:
        raw = np.asarray(q)
        if (
            raw.dtype.kind not in "iuf"
            or raw.shape != (self.nominal.dof,)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("joint configuration must be a finite real [DOF] vector")
        q = raw.astype(float, copy=True)
        limits = self.nominal.joint_limits
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("joint configuration is outside model limits")
        links = {
            "nominal": self.nominal.link_transforms(q),
            "truth": self.truth.link_transforms(q),
        }
        poses = {
            "nominal": links["nominal"][self.nominal.end_link],
            "truth": links["truth"][self.truth.end_link],
            "calibrated": self.fitted.forward_kinematics_batch(q[None])[0],
        }
        measured = None if index is None else self.dataset.observed_poses[index].copy()
        reference = poses["truth"] if measured is None else measured
        measurement_reference = (
            "training measurement" if self.split == "training" else "held-out measurement"
        )
        predicted = poses["calibrated"]
        return PoseState(
            q,
            links,
            poses,
            measured,
            float(np.linalg.norm(predicted[:3, 3] - reference[:3, 3])),
            float(
                Rotation.from_matrix(
                    predicted[:3, :3].T @ reference[:3, :3]
                ).magnitude()
            ),
            "synthetic truth" if measured is None else measurement_reference,
            None if index is None else int(self.dataset.sample_indices[index]),
        )

    def sample(self, index: int) -> PoseState:
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < len(self.dataset.joint_positions)
        ):
            raise ValueError(f"sample index must identify a {self.split} dataset row")
        return self._state(self.dataset.joint_positions[index], index=int(index))

    def manual(self, joint_positions) -> PoseState:
        return self._state(joint_positions, index=None)
