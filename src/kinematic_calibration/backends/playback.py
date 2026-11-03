"""Verified replay and manual geometry exploration, independent of a viewer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..artifacts import load_experiment_bundle
from .numpy import NumpyBackend


@dataclass(frozen=True)
class PoseState:
    joint_positions: np.ndarray
    links: dict[str, dict[str, np.ndarray]]
    poses: dict[str, np.ndarray]
    measured_pose: np.ndarray | None
    position_error_m: float
    orientation_error_rad: float
    error_reference: str
    sample_index: int | None


class ExperimentPlayback:
    """Read an experiment once, then replay samples or evaluate manual joints.

    Manual configurations have no sensor observation. Their errors are measured
    against the synthetic truth model, and cannot extend held-out validation.
    """

    def __init__(self, path: str | Path):
        from robotkinecal import SerialRobotKineCal

        self.path = Path(path)
        self.report, self.dataset, self.model = load_experiment_bundle(path)
        self.nominal = NumpyBackend.from_urdf(
            self.path / "nominal.urdf",
            end_link=self.report["end_link"],
            base_link=self.report["base_link"],
        )
        self.truth = self.nominal.perturbed(seed=self.report["truth_seed"])
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
            "synthetic truth" if measured is None else "held-out measurement",
            None if index is None else int(self.dataset.sample_indices[index]),
        )

    def sample(self, index: int) -> PoseState:
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < len(self.dataset.joint_positions)
        ):
            raise ValueError("sample index must identify a validation dataset row")
        return self._state(self.dataset.joint_positions[index], index=int(index))

    def manual(self, joint_positions) -> PoseState:
        return self._state(joint_positions, index=None)
