"""Read existing fitted POE parameters without CLI or optional engine imports."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import ObservationFrame, _validate_transforms, validated_joint_names
from .validation import validate_saved_poe_parameters


@dataclass(frozen=True)
class FittedModel:
    """Verified parameter/report/URDF pairing, without reading training records.

    ``report['accepted']`` is the recorded fit decision, not a new validation.
    Optional initialization transforms express a self-calibrating display gauge;
    they do not identify calibrated physical camera or tool extrinsics.
    """

    report: dict
    screw_axes: np.ndarray
    zero_pose: np.ndarray
    source_sample_indices: np.ndarray
    source_sample_sha256: np.ndarray
    camera_t_base_initialization: np.ndarray | None = None
    ee_t_target_initialization: np.ndarray | None = None


def _hash(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    return value


def _string(archive, name):
    value = archive[name]
    if value.shape != () or value.dtype.kind != "U" or not value.item().strip():
        raise ValueError(f"parameter {name} must be a nonempty string scalar")
    return value.item()


def load_fitted_model(
    parameters_path: Path | str,
    urdf_path: Path | str,
    *,
    result_path: Path | str | None = None,
) -> FittedModel:
    """Read a solve NPZ and companion JSON, checking their model identities.

    Parameter bytes, the companion report and the supplied URDF are bound by
    hashes and cross-checked metadata. Source sample IDs and fingerprints are
    validated structurally; no source dataset or image is read or revalidated.
    An internally consistent rejected fit can be loaded for diagnosis.
    """
    parameters_path, urdf_path = Path(parameters_path), Path(urdf_path)
    result_path = (
        Path(result_path)
        if result_path is not None
        else parameters_path.with_suffix(".json")
    )
    report = json.loads(result_path.read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("calibration_type") != "kinematic-geometry"
        or type(report.get("accepted")) is not bool
    ):
        raise ValueError("unsupported kinematic geometry report or acceptance status")
    parameter_bytes = parameters_path.read_bytes()
    parameter_hash = hashlib.sha256(parameter_bytes).hexdigest()
    urdf_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if (
        _hash(report.get("parameters_npz_sha256"), "parameters_npz_sha256")
        != parameter_hash
    ):
        raise ValueError("parameter NPZ SHA-256 does not match the result")
    if _hash(report.get("urdf_sha256"), "urdf_sha256") != urdf_hash:
        raise ValueError("source URDF SHA-256 does not match the result")
    required = {
        "schema_version",
        "parameter_type",
        "screw_axes",
        "zero_configuration_pose",
        "model_mode",
        "observation_frame",
        "source_observation_frame",
        "backend",
        "base_link",
        "joint_names",
        "joint_identity_verified",
        "source_dataset_sha256",
        "source_urdf_sha256",
        "implementation_sha256",
        "source_sample_indices",
        "source_sample_sha256",
    }
    loaded = np.load(io.BytesIO(parameter_bytes), allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("parameters must be an NPZ archive")
    with loaded as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"parameter archive is missing fields: {sorted(missing)}")
        version = archive["schema_version"]
        if version.shape != () or version.dtype.kind not in "iu" or version.item() != 1:
            raise ValueError("unsupported parameter archive schema version")
        if _string(archive, "parameter_type") != "kinematic-poe":
            raise ValueError("unsupported parameter archive type")
        axes, zero_pose = validate_saved_poe_parameters(
            archive["screw_axes"], archive["zero_configuration_pose"]
        )
        for parameter_field, report_field in (
            ("model_mode", "model_mode"),
            ("backend", "backend"),
            ("base_link", "base_link"),
            ("observation_frame", "observation_frame"),
            ("source_observation_frame", "source_observation_frame"),
            ("source_dataset_sha256", "dataset_sha256"),
            ("source_urdf_sha256", "urdf_sha256"),
        ):
            value = _string(archive, parameter_field)
            if value != report.get(report_field):
                raise ValueError(
                    f"parameter {parameter_field} does not match the result"
                )
        _hash(report["dataset_sha256"], "dataset_sha256")
        provenance = report.get("provenance")
        implementation_hash = _hash(
            _string(archive, "implementation_sha256"), "implementation_sha256"
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("implementation_sha256") != implementation_hash
        ):
            raise ValueError(
                "parameter implementation fingerprint does not match the result"
            )
        names = validated_joint_names(archive["joint_names"], len(axes))
        identity = archive["joint_identity_verified"]
        if (
            list(names) != report.get("joint_names")
            or identity.shape != ()
            or identity.dtype.kind != "b"
            or report.get("joint_identity_verified") is not identity.item()
            or type(report.get("dof")) is not int
            or report["dof"] != len(axes)
        ):
            raise ValueError(
                "parameter joint names, identity or DOF do not match the result"
            )
        indices = archive["source_sample_indices"]
        fingerprints = archive["source_sample_sha256"]
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices > np.iinfo(np.int64).max)
            or len(np.unique(indices)) != len(indices)
            or fingerprints.shape != indices.shape
            or fingerprints.dtype.kind != "U"
            or type(report.get("samples")) is not int
            or report["samples"] != len(indices)
        ):
            raise ValueError(
                "source sample indices or fingerprints have invalid shape or identity"
            )
        for fingerprint in fingerprints:
            _hash(fingerprint, "source_sample_sha256")
        mode = report["model_mode"]
        frame = ObservationFrame(report["observation_frame"])
        source_frame = ObservationFrame(report["source_observation_frame"])
        if report["backend"] not in {"numpy", "rtb"}:
            raise ValueError("unsupported fitted model backend")
        if mode == "robot-body":
            if frame != ObservationFrame.BASE_T_EE:
                raise ValueError(
                    "robot-body parameters require base_T_ee fit observations"
                )
        elif (
            mode != "self-calibrating"
            or frame != ObservationFrame.CAMERA_T_TARGET
            or source_frame != frame
        ):
            raise ValueError("unsupported fitted model mode or observation frame")
        fixed_camera = (
            mode == "robot-body" and source_frame == ObservationFrame.CAMERA_T_TARGET
        )
        for parameter_field, report_field in (
            ("source_base_T_camera_sha256", "base_T_camera_sha256"),
            ("source_EE_T_target_sha256", "ee_T_target_sha256"),
        ):
            if fixed_camera:
                if parameter_field not in archive.files:
                    raise ValueError(f"camera fit is missing {parameter_field}")
                if _hash(_string(archive, parameter_field), parameter_field) != _hash(
                    report.get(report_field), report_field
                ):
                    raise ValueError(
                        "fixed extrinsic fingerprints do not match the result"
                    )
            elif parameter_field in archive.files or report_field in report:
                raise ValueError(
                    "fixed extrinsic fingerprints do not match the observation mode"
                )
        initializations = []
        if mode == "self-calibrating":
            for field in ("estimated_camera_T_base", "estimated_EE_T_target"):
                if field not in archive.files:
                    raise ValueError(f"self-calibrating parameters are missing {field}")
                pose = _validate_transforms(field, archive[field][None])[0]
                reported = _validate_transforms(
                    field, np.asarray(report.get(field))[None]
                )[0]
                if not np.array_equal(pose, reported):
                    raise ValueError(f"parameter {field} does not match the result")
                initializations.append(pose.copy())
        else:
            initializations = [None, None]
        indices, fingerprints = indices.astype(np.int64), fingerprints.copy()

    end_link = report.get("ee_name")
    if not isinstance(end_link, str) or not end_link.strip():
        raise ValueError("result ee_name must identify the selected end link")
    from kinematic_calibration.backends.numpy import NumpyBackend

    nominal = NumpyBackend.from_urdf(
        urdf_path, end_link=end_link, base_link=report["base_link"]
    )
    if nominal.joint_names != names:
        raise ValueError("fitted joint names do not match the selected URDF chain")
    return FittedModel(
        report, axes.copy(), zero_pose.copy(), indices, fingerprints, *initializations
    )
