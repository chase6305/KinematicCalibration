"""Reusable synthetic experiments and portable, verified result artifacts."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    _validate_transforms,
    minimum_full_pose_samples,
)
from kinematic_calib.io_utils import atomic_savez_compressed, atomic_write_json
from kinematic_calib.service import FitOptions, fit_dataset
from kinematic_calib.validation import (
    predict_saved_poe,
    validate_saved_poe_parameters,
)

from .backends.numpy import NumpyBackend
from .simulation import MeasurementNoise, sample_measurements

EXPERIMENT_SCHEMA_VERSION = 2
_DATA_FILES = (
    "train.npz",
    "validation.npz",
    "model.npz",
    "nominal.urdf",
    "metrics.json",
)
_ARTIFACTS = (*_DATA_FILES, "manifest.json")
_MAX_CONDITION_NUMBER = 1e8


def example_urdf() -> Path:
    return Path(str(files("kinematic_calibration").joinpath("assets/robots/ur10.urdf")))


def _integer(value, name: str, minimum: int = 1) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _poses(value, name: str) -> np.ndarray:
    result = _validate_transforms(name, value)
    if not len(result):
        raise ValueError(f"{name} must contain at least one pose")
    return result


def predict_poe(axes: np.ndarray, home: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Evaluate a validated space-frame POE model in linear/angular order."""
    axes, home = validate_saved_poe_parameters(axes, home)
    joints = np.asarray(joints, dtype=float)
    if (
        joints.ndim != 2
        or joints.shape[1] != len(axes)
        or not len(joints)
        or not np.isfinite(joints).all()
    ):
        raise ValueError("joints must be a finite, nonempty [N, DOF] array")
    return predict_saved_poe(axes, home, joints)


def pose_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    predicted = _poses(predicted, "predicted poses")
    observed = _poses(observed, "observed poses")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed pose counts must match")
    errors = np.linalg.norm(predicted[:, :3, 3] - observed[:, :3, 3], axis=1)
    relative = predicted[:, :3, :3].transpose(0, 2, 1) @ observed[:, :3, :3]
    rotations = Rotation.from_matrix(relative).magnitude()
    return {
        "samples": len(errors),
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "position_max_m": float(np.max(errors)),
        "orientation_rmse_rad": float(np.sqrt(np.mean(rotations**2))),
        "orientation_max_rad": float(np.max(rotations)),
    }


def _preflight_output(output: Path, force: bool) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("experiment output must be a directory, not a file or symlink")
    existing = []
    for name in _ARTIFACTS:
        path = output / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"reserved artifact path is not a regular file: {path}")
        if path.exists():
            existing.append(name)
    if existing and not force:
        raise FileExistsError(
            f"experiment artifacts exist in {output}; use --force to replace"
        )


def _publish(stage: Path, output: Path, force: bool) -> None:
    """Publish a checked set, restoring previous files on ordinary I/O failures.

    The manifest is installed last; interrupted processes leave a set that the
    loader rejects. Unrelated files in an existing output directory are retained.
    """
    _preflight_output(output, force)
    created = not output.exists()
    output.mkdir(parents=True, exist_ok=True)
    backup = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-previous-", dir=output.parent)
    )
    moved, installed = [], []
    try:
        for name in _ARTIFACTS:
            destination = output / name
            if destination.exists():
                os.replace(destination, backup / name)
                moved.append(name)
        for name in _ARTIFACTS:
            os.replace(stage / name, output / name)
            installed.append(name)
    except BaseException:
        try:
            for name in reversed(installed):
                (output / name).unlink()
            for name in moved:
                os.replace(backup / name, output / name)
            if created:
                output.rmdir()
        except BaseException as recovery_error:
            raise OSError(
                f"experiment rollback failed; previous files retained at {backup}"
            ) from recovery_error
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)


def _diagnostics(singular_values: np.ndarray, samples: int, parameters: int):
    values = np.asarray(singular_values, dtype=float)
    if (
        values.shape != (parameters,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(np.diff(values) > 0.0)
    ):
        raise ValueError(
            "regressor singular values must be finite, nonnegative and descending"
        )
    tolerance = values[0] * max(6 * samples, parameters) * np.finfo(float).eps
    rank = int(np.count_nonzero(values > tolerance))
    condition = float(values[0] / values[-1]) if values[-1] > 0 else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    return rank, condition


def _quality_failures(
    converged: bool, rank: int, parameters: int, condition
) -> list[str]:
    failures = []
    if not converged:
        failures.append("solver did not converge")
    if rank != parameters:
        failures.append("calibration regressor is rank deficient")
    if condition is None or condition > _MAX_CONDITION_NUMBER:
        failures.append("calibration regressor exceeds the condition-number limit")
    return failures


def run_experiment(
    output: Path | str,
    *,
    urdf: Path | str | None = None,
    end_link: str = "ee_link",
    base_link: str | None = None,
    samples: int = 100,
    validation_samples: int = 40,
    seed: int = 42,
    noise: MeasurementNoise | None = None,
    max_iterations: int = 30,
    force: bool = False,
) -> dict:
    """Fit synthetic training poses, then evaluate independent held-out poses.

    Validation requires at least five poses. Numerical acceptance reports solver
    convergence and observability only; this synthetic experiment does not certify
    physical robot accuracy. Known output files are replaced only with ``force``.
    """
    from robotkinecal import SerialRobotKineCal

    samples = _integer(samples, "samples")
    validation_samples = _integer(validation_samples, "validation_samples", 5)
    seed = _integer(seed, "seed", 0)
    max_iterations = _integer(max_iterations, "max_iterations")
    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    if noise is not None and not isinstance(noise, MeasurementNoise):
        raise ValueError("noise must be a MeasurementNoise instance")
    noise = noise or MeasurementNoise()
    output = Path(output)
    _preflight_output(output, force)
    model_path = Path(urdf) if urdf is not None else example_urdf()
    if model_path.resolve() in {
        output.joinpath(name).resolve() for name in _ARTIFACTS if name != "nominal.urdf"
    }:
        raise ValueError("source URDF conflicts with a reserved experiment output")
    source_bytes = model_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        # Parse the exact bytes being archived, even when the source lies in output.
        (stage / "nominal.urdf").write_bytes(source_bytes)
        nominal = NumpyBackend.from_urdf(
            stage / "nominal.urdf", end_link=end_link, base_link=base_link
        )
        minimum = minimum_full_pose_samples(nominal.dof)
        if samples < minimum:
            raise ValueError(f"samples must be >= {minimum} for {nominal.dof} joints")
        truth = nominal.perturbed(seed=seed + 1)
        train = sample_measurements(truth, count=samples, seed=seed, noise=noise)
        raw_validation = sample_measurements(
            truth, count=validation_samples, seed=seed + 2, noise=noise
        )
        validation = KinematicDataset(
            raw_validation.joint_positions,
            raw_validation.observed_poses,
            raw_validation.observation_frame,
            np.arange(samples, samples + validation_samples),
            raw_validation.joint_names,
        )
        axes, home = nominal.poe_parameters()
        calibrator = SerialRobotKineCal.from_screw_axes(
            axes, home, joint_names=nominal.joint_names
        )
        fitted = fit_dataset(
            train, calibrator, options=FitOptions(max_iterations=max_iterations)
        )
        fitted_home = fitted.zero_pose.A
        nominal_predictions = np.asarray(
            [nominal.forward(q) for q in validation.joint_positions]
        )
        fitted_predictions = predict_poe(
            fitted.screw_axes, fitted_home, validation.joint_positions
        )
        truth_predictions = np.asarray(
            [truth.forward(q) for q in validation.joint_positions]
        )
        last = fitted.result.iteration_results[-1]
        parameters = int(last.N_PARAMS)
        rank, condition = _diagnostics(last.singular_values, samples, parameters)
        failures = _quality_failures(
            bool(fitted.result.has_converged), rank, parameters, condition
        )
        report = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "backend": "numpy",
            "measurement_model": "synthetic_pose_sensor",
            "sample_id_namespace": "experiment",
            "source_urdf": model_path.name,
            "source_urdf_sha256": source_hash,
            "seed": seed,
            "truth_seed": seed + 1,
            "validation_seed": seed + 2,
            "dof": nominal.dof,
            "joint_names": list(nominal.joint_names),
            "joint_identity_verified": True,
            "joint_identity": {
                "training": fitted.joint_identity,
                "validation": validation.check_joint_identity(nominal.joint_names),
            },
            "base_link": nominal.base_link,
            "end_link": nominal.end_link,
            "training_samples": samples,
            "validation_samples": validation_samples,
            "position_noise_std_m": float(noise.position_std_m),
            "orientation_noise_std_rad": float(noise.orientation_std_rad),
            "termination_reason": fitted.result.termination_reason,
            "iterations": len(fitted.result.iteration_results),
            "converged": bool(fitted.result.has_converged),
            "regressor_rank": rank,
            "parameter_count": parameters,
            "condition_number": condition,
            "full_rank": rank == parameters,
            "accepted": not failures,
            "quality_failures": failures,
            "acceptance_scope": "synthetic_solver_diagnostics",
            "max_condition_number": _MAX_CONDITION_NUMBER,
            "nominal_vs_measurements": pose_metrics(
                nominal_predictions, validation.observed_poses
            ),
            "calibrated_vs_measurements": pose_metrics(
                fitted_predictions, validation.observed_poses
            ),
            "calibrated_vs_truth": pose_metrics(fitted_predictions, truth_predictions),
        }
        train.save(stage / "train.npz")
        validation.save(stage / "validation.npz")
        truth_axes, truth_home = truth.poe_parameters()
        atomic_savez_compressed(
            stage / "model.npz",
            schema_version=np.asarray(1),
            screw_axes=fitted.screw_axes,
            home=fitted_home,
            nominal_axes=axes,
            nominal_home=home,
            truth_axes=truth_axes,
            truth_home=truth_home,
            joint_names=np.asarray(nominal.joint_names),
            base_link=np.asarray(nominal.base_link),
            end_link=np.asarray(nominal.end_link),
            source_urdf_sha256=np.asarray(source_hash),
            regressor_singular_values=last.singular_values,
            nominal_predictions=nominal_predictions,
            calibrated_predictions=fitted_predictions,
            truth_predictions=truth_predictions,
        )
        atomic_write_json(stage / "metrics.json", report)
        atomic_write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "kinematic-calibration-experiment",
                "files_sha256": {name: _sha256(stage / name) for name in _DATA_FILES},
            },
        )
        load_experiment(stage)
        _publish(stage, output, force)
    return report


def _scalar(model: dict, key: str):
    value = model[key]
    if value.shape != ():
        raise ValueError(f"model {key} must be a scalar")
    return value.item()


def _check_metrics(report: dict, key: str, expected: dict) -> None:
    actual = report.get(key)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"invalid {key} metric fields")
    if _integer(actual["samples"], f"{key} samples") != expected["samples"]:
        raise ValueError(f"{key} sample count does not match data")
    for name, value in expected.items():
        if name == "samples":
            continue
        metric = actual[name]
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not np.isfinite(metric)
            or not np.isclose(metric, value, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(f"{key} {name} does not match saved predictions")


def load_experiment(path: Path | str) -> tuple[dict, KinematicDataset, dict]:
    """Load a complete experiment, checking hashes and numerical consistency.

    SHA-256 detects accidental modification; it is not a signature from a trusted
    authority. Schema-1 experiments lack this identity contract and must be rerun.
    """
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("experiment path must be a directory")
    for name in _ARTIFACTS:
        file_path = path / name
        if not file_path.is_file() or file_path.is_symlink():
            raise ValueError(f"experiment is missing a regular {name} artifact")
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("artifact_type") != "kinematic-calibration-experiment"
    ):
        raise ValueError("unsupported experiment manifest")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_DATA_FILES):
        raise ValueError("manifest must identify every experiment artifact")
    for name in _DATA_FILES:
        if hashes[name] != _sha256(path / name):
            raise ValueError(f"experiment artifact SHA-256 mismatch: {name}")
    report = json.loads((path / "metrics.json").read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != EXPERIMENT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported experiment schema; rerun schema-1 experiments")
    if (
        report.get("backend") != "numpy"
        or report.get("measurement_model") != "synthetic_pose_sensor"
        or report.get("sample_id_namespace") != "experiment"
    ):
        raise ValueError("unsupported experiment model or sample identity")
    dof = _integer(report.get("dof"), "dof")
    training_count = _integer(report.get("training_samples"), "training_samples")
    validation_count = _integer(
        report.get("validation_samples"), "validation_samples", 5
    )
    seed = _integer(report.get("seed"), "seed", 0)
    if (
        _integer(report.get("truth_seed"), "truth_seed", 0) != seed + 1
        or _integer(report.get("validation_seed"), "validation_seed", 0) != seed + 2
    ):
        raise ValueError("experiment seed identities do not match")
    _integer(report.get("iterations"), "iterations")
    if (
        not isinstance(report.get("termination_reason"), str)
        or not report["termination_reason"]
    ):
        raise ValueError("termination_reason must be a nonempty string")
    for key in ("position_noise_std_m", "orientation_noise_std_rad"):
        value = report.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{key} must be a finite nonnegative number")
    train = KinematicDataset.load(path / "train.npz")
    dataset = KinematicDataset.load(path / "validation.npz")
    for name, value, count in (
        ("train", train, training_count),
        ("validation", dataset, validation_count),
    ):
        if (
            value.joint_positions.shape != (count, dof)
            or value.observation_frame != ObservationFrame.BASE_T_EE
        ):
            raise ValueError(
                f"{name} joint dimensions or observation frame do not match report"
            )
    train.validate_solver_sample_count()
    if not np.array_equal(
        train.sample_indices, np.arange(training_count)
    ) or not np.array_equal(
        dataset.sample_indices,
        np.arange(training_count, training_count + validation_count),
    ):
        raise ValueError("training and validation sample identities must be disjoint")
    if set(map(tuple, train.joint_positions)) & set(
        map(tuple, dataset.joint_positions)
    ):
        raise ValueError("held-out joint configurations overlap training data")
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    required = {
        "schema_version",
        "screw_axes",
        "home",
        "nominal_axes",
        "nominal_home",
        "truth_axes",
        "truth_home",
        "joint_names",
        "base_link",
        "end_link",
        "source_urdf_sha256",
        "regressor_singular_values",
        "nominal_predictions",
        "calibrated_predictions",
        "truth_predictions",
    }
    if not required.issubset(model):
        raise ValueError(
            f"model archive is missing fields: {sorted(required.difference(model))}"
        )
    if (
        type(_scalar(model, "schema_version")) is not int
        or _scalar(model, "schema_version") != 1
    ):
        raise ValueError("unsupported experiment model schema")
    for axes_key, home_key in (
        ("screw_axes", "home"),
        ("nominal_axes", "nominal_home"),
        ("truth_axes", "truth_home"),
    ):
        axes, _ = validate_saved_poe_parameters(model[axes_key], model[home_key])
        if axes.shape != (dof, 6):
            raise ValueError(f"model {axes_key} does not match experiment DOF")
    names = report.get("joint_names")
    if (
        not isinstance(names, list)
        or len(names) != dof
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != dof
        or model["joint_names"].shape != (dof,)
        or model["joint_names"].tolist() != names
    ):
        raise ValueError("model joint identities do not match experiment")
    joint_identity = {
        "training": train.check_joint_identity(names),
        "validation": dataset.check_joint_identity(names),
    }
    identity_verified = all(item["verified"] for item in joint_identity.values())
    if (
        "joint_identity_verified" in report
        and report["joint_identity_verified"] is not identity_verified
    ):
        raise ValueError(
            "report joint-column verification does not match saved datasets"
        )
    if "joint_identity" in report and report["joint_identity"] != joint_identity:
        raise ValueError("report joint identity details do not match saved datasets")
    report["joint_identity_verified"] = identity_verified
    report["joint_identity"] = joint_identity
    if not identity_verified:
        logging.getLogger(__name__).warning(
            "legacy experiment datasets have no joint_names; column identity is unverified"
        )
    for key in ("base_link", "end_link"):
        if (
            not isinstance(report.get(key), str)
            or not report[key]
            or _scalar(model, key) != report[key]
        ):
            raise ValueError(f"model {key} does not match experiment")
    if (
        report.get("source_urdf_sha256") != hashes["nominal.urdf"]
        or _scalar(model, "source_urdf_sha256") != hashes["nominal.urdf"]
    ):
        raise ValueError("source URDF identity does not match archived model")
    nominal = NumpyBackend.from_urdf(
        path / "nominal.urdf",
        base_link=report["base_link"],
        end_link=report["end_link"],
    )
    expected_axes, expected_home = nominal.poe_parameters()
    revolute = np.asarray(
        [joint.kind == "revolute" for joint in nominal.joints if joint.kind != "fixed"]
    )
    for value in (train, dataset):
        joints = value.joint_positions[:, revolute]
        if np.any(joints < nominal.joint_limits[revolute, 0] - 1e-9) or np.any(
            joints > nominal.joint_limits[revolute, 1] + 1e-9
        ):
            raise ValueError("experiment joint states exceed the archived URDF limits")
    if (
        list(nominal.joint_names) != names
        or not np.allclose(model["nominal_axes"], expected_axes, atol=1e-10, rtol=1e-10)
        or not np.allclose(model["nominal_home"], expected_home, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("nominal POE parameters do not match archived URDF")
    truth_axes, truth_home = nominal.perturbed(
        seed=report["truth_seed"]
    ).poe_parameters()
    if not np.allclose(
        model["truth_axes"], truth_axes, atol=1e-10, rtol=1e-10
    ) or not np.allclose(model["truth_home"], truth_home, atol=1e-10, rtol=1e-10):
        raise ValueError("truth model does not match the recorded perturbation seed")
    for key, axes_key, home_key in (
        ("nominal_predictions", "nominal_axes", "nominal_home"),
        ("calibrated_predictions", "screw_axes", "home"),
        ("truth_predictions", "truth_axes", "truth_home"),
    ):
        predictions = _poses(model[key], key)
        if predictions.shape != dataset.observed_poses.shape:
            raise ValueError(f"{key} must match validation sample count")
        expected = predict_poe(
            model[axes_key], model[home_key], dataset.joint_positions
        )
        if not np.allclose(predictions, expected, atol=1e-9, rtol=1e-9):
            raise ValueError(f"{key} do not match saved POE and joint states")
    for key, predictions, observed in (
        (
            "nominal_vs_measurements",
            model["nominal_predictions"],
            dataset.observed_poses,
        ),
        (
            "calibrated_vs_measurements",
            model["calibrated_predictions"],
            dataset.observed_poses,
        ),
        (
            "calibrated_vs_truth",
            model["calibrated_predictions"],
            model["truth_predictions"],
        ),
    ):
        _check_metrics(report, key, pose_metrics(predictions, observed))
    parameters = _integer(report.get("parameter_count"), "parameter_count")
    if parameters != 4 * dof + 6:
        raise ValueError("parameter count does not match the full-pose model")
    rank, condition = _diagnostics(
        model["regressor_singular_values"], training_count, parameters
    )
    if (
        report.get("regressor_rank") != rank
        or report.get("full_rank") is not (rank == parameters)
        or (condition is None and report.get("condition_number") is not None)
        or (
            condition is not None
            and (
                not isinstance(report.get("condition_number"), (int, float))
                or not np.isclose(report["condition_number"], condition)
            )
        )
    ):
        raise ValueError("regressor diagnostics do not match saved singular values")
    if type(report.get("converged")) is not bool:
        raise ValueError("converged must be a boolean")
    failures = _quality_failures(report["converged"], rank, parameters, condition)
    expected_acceptance = (
        report["converged"]
        and rank == parameters
        and condition is not None
        and condition <= _MAX_CONDITION_NUMBER
    )
    if (
        type(report.get("accepted")) is not bool
        or report["accepted"] != expected_acceptance
        or report.get("acceptance_scope") != "synthetic_solver_diagnostics"
        or report.get("max_condition_number") != _MAX_CONDITION_NUMBER
    ):
        raise ValueError("experiment acceptance diagnostics are inconsistent")
    if report.get("quality_failures") != failures:
        raise ValueError("quality failures do not match experiment diagnostics")
    return report, dataset, model
