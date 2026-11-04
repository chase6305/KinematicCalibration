"""Saved-fit reading from actual solve artifacts, independent of CLI startup."""

import hashlib
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib.cli import main
from kinematic_calib.dataset import KinematicDataset, sample_sha256
from kinematic_calib.saved_model import FittedModel, load_fitted_model
from kinematic_calibration.backends.numpy import NumpyBackend
from kinematic_calibration.experiment import example_urdf
from robotkinecal import SerialRobotKineCal


@pytest.fixture(scope="module", params=["base", "fixed-camera", "self-calibrating"])
def fitted(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"saved-fit-{request.param}")
    nominal = NumpyBackend.from_urdf(example_urdf(), end_link="ee_link")
    truth = nominal.perturbed(seed=905)
    positions = np.random.default_rng(902).uniform(-1.5, 1.5, (32, nominal.dof))
    expected = np.asarray([truth.forward(row) for row in positions])
    base_t_camera, ee_t_target = np.eye(4), np.eye(4)
    base_t_camera[:3, :3] = Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix()
    base_t_camera[:3, 3] = [0.4, 0.5, 0.8]
    ee_t_target[:3, :3] = Rotation.from_rotvec([0.1, -0.1, 0.2]).as_matrix()
    ee_t_target[:3, 3] = [0.1, 0.02, -0.03]
    observed = expected.copy()
    frame = "base_T_ee"
    if request.param != "base":
        observed = np.linalg.inv(base_t_camera) @ expected @ ee_t_target
        frame = "camera_T_target"
    dataset = KinematicDataset(
        positions, observed, frame, np.arange(100, 132), nominal.joint_names
    )
    dataset_path, parameters = directory / "train.npz", directory / "fit.npz"
    dataset.save(dataset_path)
    arguments = [
        "solve",
        "--backend",
        "numpy",
        "--dataset",
        str(dataset_path),
        "--urdf",
        str(example_urdf()),
        "--ee-name",
        "ee_link",
        "--model-mode",
        "self-calibrating" if request.param == "self-calibrating" else "robot-body",
        "--output",
        str(parameters),
        "--quiet",
    ]
    if request.param == "fixed-camera":
        np.save(directory / "base_T_camera.npy", base_t_camera)
        np.save(directory / "ee_T_target.npy", ee_t_target)
        arguments.extend(
            [
                "--base-t-camera",
                str(directory / "base_T_camera.npy"),
                "--ee-t-target",
                str(directory / "ee_T_target.npy"),
            ]
        )
    assert main(arguments) == 0
    return (
        parameters,
        dataset,
        observed if request.param == "self-calibrating" else expected,
    )


def _edited_pair(fitted, tmp_path, mutate):
    original, _, _ = fitted
    with np.load(original, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    report = json.loads(original.with_suffix(".json").read_text())
    mutate(arrays, report)
    parameters = tmp_path / "edited.npz"
    np.savez_compressed(parameters, **arrays)
    report["parameters_npz_sha256"] = hashlib.sha256(
        parameters.read_bytes()
    ).hexdigest()
    parameters.with_suffix(".json").write_text(json.dumps(report))
    return parameters


def test_actual_solve_model_and_source_fingerprints_survive_loading(
    fitted, monkeypatch
):
    parameters, source, expected = fitted
    for module in (
        "roboticstoolbox",
        "cv2",
        "viser",
        "handeye_calibration",
        "dexsim",
        "kinematic_calib.cli",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    loaded = load_fitted_model(parameters, example_urdf())
    with np.load(parameters, allow_pickle=False) as archive:
        end_link = archive["ee_name"]
        assert end_link.shape == () and end_link.dtype.kind == "U"
        assert end_link.item() == loaded.report["ee_name"] == "ee_link"
    assert loaded.end_link_identity_verified is True
    np.testing.assert_array_equal(loaded.source_sample_indices, source.sample_indices)
    np.testing.assert_array_equal(loaded.source_sample_sha256, sample_sha256(source))
    assert loaded.source_sample_indices.dtype == np.int64
    assert loaded.source_sample_sha256.dtype.kind == "U"
    assert isinstance(loaded.zero_pose, np.ndarray)
    calibrator = SerialRobotKineCal.from_screw_axes(loaded.screw_axes, loaded.zero_pose)
    np.testing.assert_allclose(
        calibrator.forward_kinematics_batch(source.joint_positions), expected, atol=2e-7
    )
    if loaded.report["model_mode"] == "self-calibrating":
        np.testing.assert_array_equal(
            loaded.camera_t_base_initialization,
            loaded.report["estimated_camera_T_base"],
        )
        assert loaded.ee_t_target_initialization.shape == (4, 4)
    else:
        assert loaded.camera_t_base_initialization is None
        assert loaded.ee_t_target_initialization is None


def test_rejected_but_consistent_fit_remains_available_for_diagnosis(fitted, tmp_path):
    parameters = _edited_pair(
        fitted, tmp_path, lambda arrays, report: report.update(accepted=False)
    )
    assert load_fitted_model(parameters, example_urdf()).report["accepted"] is False


@pytest.mark.parametrize("consumer", ["load", "verify-result", "validate-result"])
@pytest.mark.parametrize(
    "archived_end_link,reported_end_link",
    [
        pytest.param("ee_link", "Link6", id="different-fixed-tool"),
        pytest.param(["ee_link"], "ee_link", id="terminal-array"),
        pytest.param(123, 123, id="terminal-nonstring"),
        pytest.param("", "", id="terminal-empty"),
    ],
)
def test_terminal_identity_is_rejected_before_loading_the_chain(
    fitted,
    tmp_path,
    monkeypatch,
    caplog,
    consumer,
    archived_end_link,
    reported_end_link,
):
    def change_report_endpoint(arrays, report):
        arrays["ee_name"] = np.asarray(archived_end_link)
        # ee_link and Link6 share active joints but differ by a fixed tool joint.
        report["ee_name"] = reported_end_link

    parameters = _edited_pair(fitted, tmp_path, change_report_endpoint)

    def unexpected_chain(*args, **kwargs):
        pytest.fail("mismatched terminal identity reached nominal chain construction")

    monkeypatch.setattr(NumpyBackend, "from_urdf", unexpected_chain)
    if consumer == "load":
        with pytest.raises(ValueError, match="ee_name|end.link|terminal"):
            load_fitted_model(parameters, example_urdf())
    else:
        arguments = [
            consumer,
            "--parameters",
            str(parameters),
            "--result",
            str(parameters.with_suffix(".json")),
            "--dataset",
            str(fitted[0].parent / "train.npz"),
        ]
        if consumer == "verify-result":
            arguments += ["--urdf", str(example_urdf())]
            report = json.loads(parameters.with_suffix(".json").read_text())
            if "base_T_camera_sha256" in report:
                arguments += [
                    "--base-t-camera",
                    str(fitted[0].parent / "base_T_camera.npy"),
                    "--ee-t-target",
                    str(fitted[0].parent / "ee_T_target.npy"),
                ]
        assert main(arguments) == 2
        assert "ee_name" in caplog.text


def test_legacy_terminal_identity_remains_unverified(fitted, tmp_path):
    parameters = _edited_pair(
        fitted, tmp_path, lambda arrays, report: arrays.pop("ee_name", None)
    )
    loaded = load_fitted_model(parameters, example_urdf())
    assert loaded.report["ee_name"] == "ee_link"
    assert loaded.end_link_identity_verified is False
    positional = FittedModel(
        loaded.report,
        loaded.screw_axes,
        loaded.zero_pose,
        loaded.source_sample_indices,
        loaded.source_sample_sha256,
        loaded.camera_t_base_initialization,
        loaded.ee_t_target_initialization,
    )
    assert positional.end_link_identity_verified is False

    from kinematic_calibration.backends.measured import MeasuredPlayback

    options = {}
    if "base_T_camera_sha256" in loaded.report:
        options = {
            "base_t_camera": fitted[0].parent / "base_T_camera.npy",
            "ee_t_target": fitted[0].parent / "ee_T_target.npy",
        }
    playback = MeasuredPlayback(
        parameters, fitted[0].parent / "train.npz", example_urdf(), **options
    )
    assert playback.report["replay_end_link_identity_verified"] is False
    assert "Legacy fit" in playback.report["display_note"]
    if loaded.report["model_mode"] == "self-calibrating":
        assert "initialization" in playback.report["display_note"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a, r: r.update(accepted="true"),
        lambda a, r: r.update(
            model_mode="robot-body"
            if r["model_mode"] == "self-calibrating"
            else "self-calibrating"
        ),
        lambda a, r: r.update(joint_names=None),
        lambda a, r: r.update(dof=True),
        lambda a, r: r.update(samples=31),
        lambda a, r: a.update(schema_version=np.asarray(True)),
        lambda a, r: a.update(parameter_type=np.asarray("joint-zero-offset")),
        lambda a, r: a.update(screw_axes=np.zeros((6, 6))),
        lambda a, r: a.update(zero_configuration_pose=np.zeros((4, 4))),
        lambda a, r: a.update(source_sample_indices=np.zeros(32, dtype=np.int64)),
        lambda a, r: a.update(source_sample_indices=np.arange(32, dtype=float) + 0.5),
        lambda a, r: a.update(source_sample_sha256=np.asarray(["bad"] * 32)),
        lambda a, r: a.update(source_sample_sha256=np.asarray(["0" * 64] * 31)),
        lambda a, r: a.update(source_urdf_sha256=np.asarray("0" * 64)),
        lambda a, r: r["provenance"].update(implementation_sha256="0" * 64),
    ],
)
def test_inconsistent_metadata_is_rejected_even_after_parameter_rehash(
    fitted, tmp_path, mutate
):
    parameters = _edited_pair(fitted, tmp_path, mutate)
    with pytest.raises(ValueError):
        load_fitted_model(parameters, example_urdf())


def test_fitted_chain_identity_is_checked_against_urdf(fitted, tmp_path):
    def swap_names(arrays, report):
        arrays["joint_names"] = arrays["joint_names"][::-1]
        report["joint_names"] = report["joint_names"][::-1]

    parameters = _edited_pair(fitted, tmp_path, swap_names)
    with pytest.raises(ValueError, match="URDF chain"):
        load_fitted_model(parameters, example_urdf())


def test_mode_specific_extrinsic_metadata_cannot_be_mixed(fitted, tmp_path):
    def edit_extrinsics(arrays, report):
        if report["model_mode"] == "self-calibrating":
            arrays["estimated_camera_T_base"][0, 3] += 0.01
        else:
            arrays["source_base_T_camera_sha256"] = np.asarray("0" * 64)

    parameters = _edited_pair(fitted, tmp_path, edit_extrinsics)
    with pytest.raises(ValueError, match="extrinsic|estimated_camera_T_base"):
        load_fitted_model(parameters, example_urdf())


def test_custom_result_path_and_actual_file_hashes_are_checked(fitted, tmp_path):
    original, _, _ = fitted
    report = json.loads(original.with_suffix(".json").read_text())
    companion = tmp_path / "custom-result.json"
    companion.write_text(json.dumps(report))
    assert (
        load_fitted_model(original, example_urdf(), result_path=companion).report
        == report
    )
    report["parameters_npz_sha256"] = "0" * 64
    companion.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="NPZ SHA-256"):
        load_fitted_model(original, example_urdf(), result_path=companion)
    altered_urdf = tmp_path / "altered.urdf"
    altered_urdf.write_bytes(example_urdf().read_bytes() + b"\n")
    with pytest.raises(ValueError, match="URDF SHA-256"):
        load_fitted_model(original, altered_urdf)


def test_fingerprint_alias_preserves_existing_cli_algorithm():
    from kinematic_calib.cli import _dataset_sample_sha256

    assert _dataset_sample_sha256 is sample_sha256
    joints = np.asarray([[0.2, -0.1]])
    poses = np.eye(4)[None]
    poses[0, :3, 3] = [0.3, -0.5, 0.7]
    dataset = KinematicDataset(joints, poses, "base_T_ee")
    payload = (
        b"base_T_ee" + joints.astype("<f8").tobytes() + poses[0].astype("<f8").tobytes()
    )
    assert sample_sha256(dataset).tolist() == [hashlib.sha256(payload).hexdigest()]
