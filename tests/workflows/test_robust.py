"""Opt-in Huber weights and fresh-model robust-service behavior."""

import numpy as np
import pytest

from kinematic_calib import FitOptions, HuberOptions, KinematicDataset, fit_dataset
from kinematic_calib.robust import huber_sample_weights


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("position_scale_m", 0),
        ("position_scale_m", float("nan")),
        ("position_scale_m", True),
        ("orientation_scale_rad", -1),
        ("orientation_scale_rad", "0.01"),
        ("huber_delta", np.inf),
        ("max_outer_iterations", 0),
        ("max_outer_iterations", True),
        ("weight_tolerance", 1),
        ("min_residual_multiplier", 0),
        ("min_residual_multiplier", 1),
    ],
)
def test_huber_requires_fixed_finite_noise_scales_and_valid_controls(keyword, value):
    arguments = {"position_scale_m": 0.001, "orientation_scale_rad": 0.002}
    arguments[keyword] = value
    with pytest.raises(ValueError):
        HuberOptions(**arguments)


def test_irls_uses_square_root_precision_weights_and_preserves_positive_floor():
    options = HuberOptions(1, 1, huber_delta=2, min_residual_multiplier=0.1)
    norm, multipliers, losses = huber_sample_weights(
        [0, 1, 8, 1000], [0, 0, 0, 0], options
    )
    np.testing.assert_allclose(norm, [0, 1, 8, 1000])
    np.testing.assert_allclose(multipliers, [1, 1, 0.5, 0.1])
    np.testing.assert_allclose(losses, [0, 0.5, 14, 1998])
    assert np.all(multipliers > 0)
    assert multipliers[2] ** 2 == pytest.approx(2 / 8)


def test_normalization_distinguishes_metres_and_radians():
    options = HuberOptions(position_scale_m=0.001, orientation_scale_rad=0.002)
    norm, _, _ = huber_sample_weights([0.003], [0.008], options)
    assert norm[0] == pytest.approx(5)


@pytest.mark.parametrize(
    "positions,orientations",
    [
        ([], []),
        ([1, 2], [1]),
        ([-1], [0]),
        ([np.nan], [0]),
        ([True], [0]),
        ([1 + 2j], [0]),
        (["1"], [0]),
    ],
)
def test_invalid_endpoint_error_vectors_are_rejected(positions, orientations):
    with pytest.raises(ValueError):
        huber_sample_weights(positions, orientations, HuberOptions(1, 1))


def _dataset(frame="base_T_ee"):
    q = np.array([[0.0], [0.4], [0.8]])
    return KinematicDataset(
        q, np.repeat(np.eye(4)[None], 3, axis=0), frame, joint_names=("joint",)
    )


def test_robust_path_is_explicit_and_does_not_accept_existing_model_objects():
    dataset = _dataset()
    options = FitOptions(robust=HuberOptions(0.001, 0.002))
    with pytest.raises(ValueError, match="calibrator_factory"):
        fit_dataset(dataset, object(), options=options)
    with pytest.raises(ValueError, match="calibrator_factory"):
        fit_dataset(dataset, options=options)
    with pytest.raises(ValueError, match="robust must"):
        FitOptions(robust="huber")


def test_unknown_self_calibrating_extrinsics_are_rejected_before_factory_call():
    called = []
    options = FitOptions(
        model_mode="self-calibrating", robust=HuberOptions(0.001, 0.002)
    )
    with pytest.raises(ValueError, match="unknown self-calibrating"):
        fit_dataset(
            _dataset("camera_T_target"),
            options=options,
            calibrator_factory=lambda: called.append(1),
        )
    assert not called


def test_camera_robust_fit_requires_both_known_fixed_transforms():
    options = FitOptions(robust=HuberOptions(0.001, 0.002))
    with pytest.raises(ValueError, match="both fixed extrinsics"):
        fit_dataset(
            _dataset("camera_T_target"),
            options=options,
            calibrator_factory=lambda: None,
        )


def test_robust_path_rejects_uncontrolled_external_weight_combinations():
    options = FitOptions(robust=HuberOptions(0.001, 0.002))
    with pytest.raises(ValueError, match="weights internally"):
        fit_dataset(
            _dataset(),
            options=options,
            calibrator_factory=lambda: None,
            sample_weights=np.ones(3),
        )


def _stub_inner_fit(dataset, calibrator, **kwargs):
    from types import SimpleNamespace

    from kinematic_calib.service import CalibrationFit

    last = SimpleNamespace(
        weighted_matrix_rank=10,
        weighted_condition_number=2.0,
        matrix_rank=10,
        N_PARAMS=10,
    )
    result = SimpleNamespace(
        has_converged=True, termination_reason="converged", iteration_results=[last]
    )
    return CalibrationFit(
        result=result,
        screw_axes=np.asarray(calibrator.joint_screw_axis),
        zero_pose=calibrator.zero_conf_EE_pose,
        dataset=dataset,
    )


def _one_joint_calibrator(*, axis=None, home=None, name="joint"):
    from robotkinecal import SerialRobotKineCal

    return SerialRobotKineCal.from_screw_axes(
        np.array([[0, 0, 0, 0, 0, 1]] if axis is None else [axis], float),
        np.eye(4) if home is None else home,
        joint_names=[name],
    )


def test_factory_cannot_reuse_one_mutable_calibrator_across_rounds(monkeypatch):
    import kinematic_calib.service as service

    calibrator = _one_joint_calibrator()
    monkeypatch.setattr(service, "fit_dataset", _stub_inner_fit)
    with pytest.raises(ValueError, match="reused an existing object"):
        fit_dataset(
            _dataset(),
            options=FitOptions(robust=HuberOptions(0.001, 0.002)),
            calibrator_factory=lambda: calibrator,
        )


@pytest.mark.parametrize("change", ["axis", "home", "names"])
def test_factory_cannot_change_initial_nominal_geometry_or_joint_names(
    monkeypatch, change
):
    import kinematic_calib.service as service

    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _one_joint_calibrator()
        if change == "axis":
            return _one_joint_calibrator(axis=[0, 0, 0, 0, 1, 0])
        if change == "names":
            return _one_joint_calibrator(name="different_joint")
        home = np.eye(4)
        home[0, 3] = 0.001
        return _one_joint_calibrator(home=home)

    monkeypatch.setattr(service, "fit_dataset", _stub_inner_fit)
    with pytest.raises(ValueError, match="changed the initial nominal"):
        fit_dataset(
            _dataset(),
            options=FitOptions(robust=HuberOptions(0.001, 0.002)),
            calibrator_factory=factory,
        )


def test_diagnostics_identify_every_retained_row_and_actual_applied_weights(
    monkeypatch,
):
    import kinematic_calib.service as service

    dataset = _dataset()
    original_joints, original_poses = (
        dataset.joint_positions.copy(),
        dataset.observed_poses.copy(),
    )
    monkeypatch.setattr(service, "fit_dataset", _stub_inner_fit)
    fit = fit_dataset(
        dataset,
        options=FitOptions(robust=HuberOptions(0.001, 0.002)),
        calibrator_factory=_one_joint_calibrator,
    )
    diagnostics = fit.robust_diagnostics
    assert diagnostics["outer_converged"]
    assert len(diagnostics["iterations"]) == 2
    assert diagnostics["samples_retained"] == 3
    assert diagnostics["samples_removed"] == 0
    assert [
        row["sample_id"] for row in diagnostics["samples"]
    ] == dataset.sample_indices.tolist()
    applied = diagnostics["iterations"][-1]["sample_residual_multipliers"]
    assert [row["residual_multiplier"] for row in diagnostics["samples"]] == applied
    assert all(row["residual_multiplier"] > 0 for row in diagnostics["samples"])
    assert all(
        row["precision_weight"] == pytest.approx(row["residual_multiplier"] ** 2)
        for row in diagnostics["samples"]
    )
    np.testing.assert_array_equal(dataset.joint_positions, original_joints)
    np.testing.assert_array_equal(dataset.observed_poses, original_poses)


def test_max_outer_limit_does_not_claim_stable_huber_weights(monkeypatch):
    import kinematic_calib.service as service

    monkeypatch.setattr(service, "fit_dataset", _stub_inner_fit)
    fit = fit_dataset(
        _dataset(),
        options=FitOptions(robust=HuberOptions(0.001, 0.002, max_outer_iterations=1)),
        calibrator_factory=_one_joint_calibrator,
    )
    diagnostics = fit.robust_diagnostics
    assert not diagnostics["outer_converged"]
    assert not diagnostics["converged"]
    assert diagnostics["outer_termination_reason"] == "max_outer_iterations"
    assert all(row["residual_multiplier"] == 1 for row in diagnostics["samples"])


def _contaminated_problem(seed, fraction):
    from scipy.spatial.transform import Rotation

    from kinematic_calibration.backends import NumpyBackend
    from kinematic_calibration.experiment import example_urdf
    from kinematic_calibration.simulation import MeasurementNoise, sample_measurements
    from robotkinecal import SerialRobotKineCal

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=seed + 1)
    training = sample_measurements(truth, count=100, seed=seed)
    validation = sample_measurements(
        truth, count=40, seed=seed + 2, noise=MeasurementNoise(0, 0)
    )
    observed = training.observed_poses.copy()
    random = np.random.default_rng(seed + 3)
    count = int(fraction * len(observed))
    corrupted = random.choice(len(observed), count, replace=False)
    observed[corrupted, :3, 3] += random.normal(0, 0.03, (count, 3))
    observed[corrupted, :3, :3] = (
        observed[corrupted, :3, :3]
        @ Rotation.from_rotvec(random.normal(0, 0.05, (count, 3))).as_matrix()
    )
    dataset = KinematicDataset(
        training.joint_positions,
        observed,
        training.observation_frame,
        training.sample_indices + 1000,
        training.joint_names,
    )
    axes, home = nominal.poe_parameters()

    def factory():
        return SerialRobotKineCal.from_screw_axes(
            axes, home, joint_names=nominal.joint_names
        )

    return dataset, validation, corrupted, factory


@pytest.mark.parametrize("seed", [7, 42, 123])
@pytest.mark.parametrize("fraction", [0, 0.01, 0.05, 0.1])
def test_huber_preserves_clean_accuracy_and_reduces_corrupted_training_sensitivity(
    seed, fraction
):
    from kinematic_calibration.experiment import pose_metrics, predict_poe

    dataset, validation, corrupted, factory = _contaminated_problem(seed, fraction)
    source_joints, source_poses = (
        dataset.joint_positions.copy(),
        dataset.observed_poses.copy(),
    )
    ordinary = fit_dataset(dataset, factory())
    robust = fit_dataset(
        dataset,
        calibrator_factory=factory,
        options=FitOptions(robust=HuberOptions(0.0001, 0.0002)),
    )
    ordinary_metrics = pose_metrics(
        predict_poe(
            ordinary.screw_axes, ordinary.zero_pose.A, validation.joint_positions
        ),
        validation.observed_poses,
    )
    robust_metrics = pose_metrics(
        predict_poe(robust.screw_axes, robust.zero_pose.A, validation.joint_positions),
        validation.observed_poses,
    )
    for name in ("position_rmse_m", "orientation_rmse_rad"):
        if fraction:
            assert robust_metrics[name] < 0.35 * ordinary_metrics[name]
        else:
            assert robust_metrics[name] <= 1.15 * ordinary_metrics[name] + 1e-6
    diagnostics = robust.robust_diagnostics
    assert diagnostics["samples_removed"] == 0
    assert diagnostics["samples_retained"] == len(dataset.joint_positions)
    assert (
        diagnostics["weighted_regressor_rank"] == diagnostics["parameter_count"] == 30
    )
    assert diagnostics["weighted_condition_number"] < 100
    assert [
        item["sample_id"] for item in diagnostics["samples"]
    ] == dataset.sample_indices.tolist()
    assert diagnostics["inner_converged"] == robust.result.has_converged
    assert diagnostics["converged"] == (
        diagnostics["outer_converged"] and robust.result.has_converged
    )
    if len(corrupted):
        assert (
            max(
                diagnostics["samples"][index]["residual_multiplier"]
                for index in corrupted
            )
            < 0.25
        )
    np.testing.assert_array_equal(dataset.joint_positions, source_joints)
    np.testing.assert_array_equal(dataset.observed_poses, source_poses)


def test_noiseless_inliers_keep_unit_weights_and_the_ordinary_solution():
    from kinematic_calibration.backends import NumpyBackend
    from kinematic_calibration.experiment import example_urdf
    from kinematic_calibration.simulation import MeasurementNoise, sample_measurements
    from robotkinecal import SerialRobotKineCal

    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=21)
    data = sample_measurements(truth, count=40, seed=22, noise=MeasurementNoise(0, 0))
    axes, home = nominal.poe_parameters()

    def factory():
        return SerialRobotKineCal.from_screw_axes(
            axes, home, joint_names=nominal.joint_names
        )

    ordinary = fit_dataset(data, factory())
    robust = fit_dataset(
        data,
        calibrator_factory=factory,
        options=FitOptions(robust=HuberOptions(0.0001, 0.0002)),
    )
    assert robust.robust_diagnostics["outer_converged"]
    assert len(robust.robust_diagnostics["iterations"]) == 1
    assert all(
        row["residual_multiplier"] == 1 for row in robust.robust_diagnostics["samples"]
    )
    np.testing.assert_allclose(
        ordinary.screw_axes, robust.screw_axes, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        ordinary.zero_pose.A, robust.zero_pose.A, atol=1e-12, rtol=0
    )


def test_known_camera_extrinsics_preserve_body_fit_and_sample_identity():
    from scipy.spatial.transform import Rotation

    dataset, _, _, factory = _contaminated_problem(42, 0.05)
    base_t_camera = np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    base_t_camera[:3, 3] = [0.4, -0.3, 1.0]
    ee_t_target = np.eye(4)
    ee_t_target[:3, 3] = [0.03, -0.02, 0.1]
    camera_poses = np.asarray(
        [
            np.linalg.inv(base_t_camera) @ pose @ ee_t_target
            for pose in dataset.observed_poses
        ]
    )
    camera = KinematicDataset(
        dataset.joint_positions,
        camera_poses,
        "camera_T_target",
        dataset.sample_indices,
        dataset.joint_names,
    )
    options = FitOptions(robust=HuberOptions(0.0001, 0.0002))
    body_fit = fit_dataset(dataset, options=options, calibrator_factory=factory)
    camera_fit = fit_dataset(
        camera,
        options=options,
        calibrator_factory=factory,
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
    )
    assert (
        camera_fit.robust_diagnostics["source_observation_frame"] == "camera_T_target"
    )
    assert camera_fit.robust_diagnostics["fit_observation_frame"] == "base_T_ee"
    assert camera_fit.dataset.joint_names == dataset.joint_names
    assert camera_fit.dataset.sample_indices.tolist() == dataset.sample_indices.tolist()
    np.testing.assert_allclose(
        camera_fit.screw_axes, body_fit.screw_axes, atol=1e-7, rtol=0
    )
    np.testing.assert_allclose(
        camera_fit.zero_pose.A, body_fit.zero_pose.A, atol=1e-7, rtol=0
    )
