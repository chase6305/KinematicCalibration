"""Validated interchange format for kinematic-calibration observations."""

from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path

import numpy as np

from .io_utils import atomic_savez_compressed

DATASET_SCHEMA_VERSION = 2


def validated_joint_names(
    value, dof: int, *, label: str = "joint_names"
) -> tuple[str, ...]:
    """Normalize an ordered, unique joint-column identity without reordering."""
    names = np.asarray(value)
    if (
        names.ndim != 1
        or names.shape != (dof,)
        or names.dtype.kind != "U"
        or any(not name.strip() for name in names.tolist())
        or len(set(names.tolist())) != dof
    ):
        raise ValueError(
            f"{label} must contain {dof} unique nonempty strings in joint-column order"
        )
    return tuple(names.tolist())


def read_dataset_schema_version(path: Path | str) -> int | None:
    """Return the declared schema, or ``None`` for a legacy archive."""
    with np.load(path, allow_pickle=False) as data:
        if "schema_version" not in data.files:
            return None
        schema_value = data["schema_version"]
        if schema_value.shape != ():
            raise ValueError("dataset schema_version must be a scalar")
        try:
            raw_schema_version = schema_value.item()
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset schema_version must be an integer") from exc
        if isinstance(raw_schema_version, bool) or raw_schema_version != schema_version:
            raise ValueError("dataset schema_version must be an integer")
        return schema_version


class ObservationFrame(str, Enum):
    """Physical meaning of the observed pose matrices."""

    BASE_T_EE = "base_T_ee"
    CAMERA_T_TARGET = "camera_T_target"


def minimum_full_pose_samples(dof: int) -> int:
    """Return the equation-count lower bound for full SE(3) observations."""
    if not isinstance(dof, int) or isinstance(dof, bool) or dof <= 0:
        raise ValueError("dof must be a positive integer")
    return ceil((4 * dof + 6) / 6)


def _validate_transforms(name: str, transforms: np.ndarray) -> np.ndarray:
    raw = np.asarray(transforms)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N, 4, 4]")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    expected_bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0], values[:, 3].shape)
    if not np.allclose(values[:, 3], expected_bottom, rtol=0, atol=1e-8):
        raise ValueError(f"{name} contains invalid homogeneous transforms")
    rotations = values[:, :3, :3]
    identities = np.matmul(rotations.transpose(0, 2, 1), rotations)
    if not np.allclose(identities, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError(f"{name} contains non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"{name} contains improper rotations")
    return values


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one 4x4 rigid transform."""
    transform = _validate_transforms("transform", np.asarray(transform)[None])[0]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def camera_target_to_base_ee(
    camera_t_target: np.ndarray,
    base_t_camera: np.ndarray,
    ee_t_target: np.ndarray,
) -> np.ndarray:
    """Convert Camera_T_Target observations into Base_T_EE observations."""
    observations = _validate_transforms("camera_T_target", camera_t_target)
    base_t_camera = _validate_transforms(
        "base_T_camera", np.asarray(base_t_camera)[None]
    )[0]
    target_t_ee = invert_transform(ee_t_target)
    converted = base_t_camera @ observations @ target_t_ee
    # Project rotations within the accepted rigid-transform tolerance onto
    # SO(3), so strict downstream checks also accept compositions near pi.
    left, _, right = np.linalg.svd(converted[:, :3, :3])
    correction = np.repeat(np.eye(3)[None], len(converted), axis=0)
    correction[:, 2, 2] = np.linalg.det(left @ right)
    converted[:, :3, :3] = left @ correction @ right
    converted[:, 3] = [0.0, 0.0, 0.0, 1.0]
    return converted


@dataclass(frozen=True)
class KinematicDataset:
    """Synchronized joint states and observed terminal poses."""

    joint_positions: np.ndarray
    observed_poses: np.ndarray
    observation_frame: ObservationFrame
    sample_indices: np.ndarray | None = None
    joint_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        raw_joints = np.asarray(self.joint_positions)
        if raw_joints.dtype.kind not in "iuf":
            raise ValueError("joint_positions must contain real numeric values")
        joints = np.asarray(raw_joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[0] == 0 or joints.shape[1] == 0:
            raise ValueError("joint_positions must have shape [N, DOF]")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint_positions contains non-finite values")
        poses = _validate_transforms("observed_poses", self.observed_poses)
        if len(joints) != len(poses):
            raise ValueError("joint and pose sample counts must match")
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "observed_poses", poses)
        object.__setattr__(
            self, "observation_frame", ObservationFrame(self.observation_frame)
        )
        if self.sample_indices is None:
            sample_indices = np.arange(len(joints), dtype=np.int64)
        else:
            raw_indices = np.asarray(self.sample_indices)
            if raw_indices.ndim != 1 or len(raw_indices) != len(joints):
                raise ValueError("sample_indices must have shape [N]")
            if raw_indices.dtype.kind not in "iuf" or not np.all(
                np.isfinite(raw_indices)
            ):
                raise ValueError("sample_indices must contain finite real integers")
            if (
                np.any(raw_indices < 0)
                or (raw_indices.dtype.kind == "f" and np.any(raw_indices >= 2.0**63))
                or (
                    raw_indices.dtype.kind in "iu"
                    and np.any(raw_indices > np.iinfo(np.int64).max)
                )
            ):
                raise ValueError("sample_indices must fit non-negative int64 integers")
            try:
                sample_indices = raw_indices.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("sample_indices must contain integers") from exc
            if not np.array_equal(raw_indices, sample_indices):
                raise ValueError("sample_indices must contain integers")
            if np.any(sample_indices < 0) or len(np.unique(sample_indices)) != len(
                sample_indices
            ):
                raise ValueError("sample_indices must be unique and non-negative")
        object.__setattr__(self, "sample_indices", sample_indices)
        if self.joint_names is not None:
            object.__setattr__(
                self,
                "joint_names",
                validated_joint_names(self.joint_names, joints.shape[1]),
            )

    def save(self, path: Path | str) -> None:
        """Save a portable NPZ without pickle-backed objects."""
        metadata = (
            {"joint_names": np.asarray(self.joint_names)}
            if self.joint_names is not None
            else {}
        )
        atomic_savez_compressed(
            path,
            schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
            joint_positions=self.joint_positions,
            observed_poses=self.observed_poses,
            observation_frame=np.asarray(self.observation_frame.value),
            sample_indices=self.sample_indices,
            **metadata,
        )

    @classmethod
    def load(cls, path: Path | str) -> "KinematicDataset":
        """Load and validate a dataset from disk."""
        schema_version = read_dataset_schema_version(path)
        if schema_version not in (None, 1, DATASET_SCHEMA_VERSION):
            raise ValueError(f"unsupported dataset schema version: {schema_version}")
        with np.load(path, allow_pickle=False) as data:
            required = {"joint_positions", "observed_poses", "observation_frame"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"dataset is missing fields: {sorted(missing)}")
            return cls(
                joint_positions=data["joint_positions"],
                observed_poses=data["observed_poses"],
                observation_frame=str(data["observation_frame"].item()),
                sample_indices=(
                    data["sample_indices"] if "sample_indices" in data.files else None
                ),
                joint_names=data["joint_names"]
                if "joint_names" in data.files
                else None,
            )

    def as_robot_body(
        self, base_t_camera: np.ndarray, ee_t_target: np.ndarray
    ) -> "KinematicDataset":
        """Return Base_T_EE data suitable for physical-model calibration."""
        if self.observation_frame == ObservationFrame.BASE_T_EE:
            return self
        poses = camera_target_to_base_ee(
            self.observed_poses, base_t_camera, ee_t_target
        )
        return KinematicDataset(
            self.joint_positions,
            poses,
            ObservationFrame.BASE_T_EE,
            self.sample_indices,
            self.joint_names,
        )

    def check_joint_identity(self, expected_joint_names) -> dict:
        """Check exact column order; legacy missing names stay explicitly unverified.

        This method never guesses joint identity from column count and never
        reorders data. A named dataset requires declared model joint names.
        """
        expected = (
            validated_joint_names(
                expected_joint_names,
                self.joint_positions.shape[1],
                label="model joint_names",
            )
            if expected_joint_names is not None
            else None
        )
        if self.joint_names is not None:
            if expected is None:
                raise ValueError(
                    "named dataset requires model joint_names to verify column identity"
                )
            if self.joint_names != expected:
                raise ValueError(
                    "dataset joint_names order does not match model: "
                    f"dataset={list(self.joint_names)}, model={list(expected)}; "
                    "columns are never automatically reordered"
                )
        verified = self.joint_names is not None and expected is not None
        return {
            "verified": verified,
            "dataset_joint_names": list(self.joint_names)
            if self.joint_names is not None
            else None,
            "model_joint_names": list(expected) if expected is not None else None,
            "reason": "exact ordered names match"
            if verified
            else "dataset has no joint_names; joint-column identity is unverified",
        }

    def validate_solver_sample_count(self) -> None:
        """Reject data below RobotKineCal's full-pose equation lower bound."""
        minimum = minimum_full_pose_samples(self.joint_positions.shape[1])
        unique_count = self.get_summary()["unique_sample_count"]
        if unique_count < minimum:
            raise ValueError(
                f"at least {minimum} unique full-pose samples are required for "
                f"{self.joint_positions.shape[1]} joints; got {unique_count} unique "
                f"samples from {len(self.joint_positions)} rows"
            )

    def get_summary(self) -> dict:
        """Return dependency-free coverage metrics for acquisition review."""
        joint_min = np.min(self.joint_positions, axis=0)
        joint_max = np.max(self.joint_positions, axis=0)
        translations = self.observed_poses[:, :3, 3]
        translation_span = np.max(translations, axis=0) - np.min(translations, axis=0)
        rotations = self.observed_poses[:, :3, :3]
        reference_rotation = rotations[0]
        relative_rotations = np.matmul(reference_rotation.T, rotations)
        traces = np.trace(relative_rotations, axis1=1, axis2=2)
        cos_angles = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        seen_samples: dict[bytes, int] = {}
        duplicate_positions = []
        duplicate_of_positions = []
        for position, (joints, pose) in enumerate(
            zip(self.joint_positions, self.observed_poses, strict=True)
        ):
            key = joints.tobytes() + pose.tobytes()
            original_position = seen_samples.setdefault(key, position)
            if original_position != position:
                duplicate_positions.append(position)
                duplicate_of_positions.append(original_position)
        return {
            "samples": int(len(self.joint_positions)),
            "unique_sample_count": int(
                len(self.joint_positions) - len(duplicate_positions)
            ),
            "duplicate_sample_count": len(duplicate_positions),
            "duplicate_sample_positions": duplicate_positions,
            "duplicate_of_positions": duplicate_of_positions,
            "duplicate_source_sample_indices": [
                int(self.sample_indices[position]) for position in duplicate_positions
            ],
            "dof": int(self.joint_positions.shape[1]),
            "joint_names": list(self.joint_names)
            if self.joint_names is not None
            else None,
            "joint_names_present": self.joint_names is not None,
            "observation_frame": self.observation_frame.value,
            "source_sample_index_min": int(self.sample_indices.min()),
            "source_sample_index_max": int(self.sample_indices.max()),
            "joint_min_rad": joint_min.tolist(),
            "joint_max_rad": joint_max.tolist(),
            "joint_span_rad": (joint_max - joint_min).tolist(),
            "translation_span_m": translation_span.tolist(),
            "translation_span_norm_m": float(np.linalg.norm(translation_span)),
            "max_rotation_from_first_rad": float(np.max(angles)),
        }

    def get_quality_warnings(
        self,
        *,
        min_joint_span_rad: float = 0.1,
        min_translation_span_norm_m: float = 0.05,
        min_rotation_from_first_rad: float = 0.2,
    ) -> list[str]:
        """Return coverage failures using the shared acquisition/inspect rules."""
        thresholds = (
            min_joint_span_rad,
            min_translation_span_norm_m,
            min_rotation_from_first_rad,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("coverage thresholds must be finite and non-negative")
        summary = self.get_summary()
        warnings = []
        weak_joints = [
            index
            for index, span in enumerate(summary["joint_span_rad"])
            if span < min_joint_span_rad
        ]
        if weak_joints:
            warnings.append(
                "joint span below threshold for zero-based joints: "
                + ", ".join(str(index) for index in weak_joints)
            )
        if summary["translation_span_norm_m"] < min_translation_span_norm_m:
            warnings.append("target translation coverage is below threshold")
        if summary["max_rotation_from_first_rad"] < min_rotation_from_first_rad:
            warnings.append("target rotation coverage is below threshold")
        if summary["duplicate_sample_count"]:
            warnings.append(
                "duplicate synchronized observations at source sample indices: "
                + ", ".join(
                    str(index) for index in summary["duplicate_source_sample_indices"]
                )
            )
        return warnings
