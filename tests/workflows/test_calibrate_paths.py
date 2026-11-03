"""Unified calibration preflights destinations before publishing any stage."""

import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinematic_calib import KinematicDataset, ObservationFrame
from kinematic_calib.cli import main
from kinematic_calibration.backends import NumpyBackend
from kinematic_calibration.experiment import example_urdf


def _pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    pose[:3, 3] = translation
    return pose


def _handeye_samples(path):
    camera = _pose([0.2, -0.1, 0.3], [0.4, -0.2, 1.1])
    target = _pose([-0.1, 0.15, 0.05], [0.02, -0.03, 0.18])
    robot = np.stack(
        [
            _pose(
                [0.08 * index, -0.05 * index**2, 0.04 * (index + 1)],
                [0.1 * np.sin(index), 0.08 * np.cos(index), 0.35 + 0.03 * index],
            )
            for index in range(1, 10)
        ]
    )
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        archive_type=np.asarray("eye-to-hand-observations"),
        base_T_ee=robot,
        camera_T_board=np.stack(
            [np.linalg.inv(camera) @ pose @ target for pose in robot]
        ),
        candidate_indices=np.arange(9),
        translation_candidate_count=np.asarray(6),
        rotation_candidate_count=np.asarray(3),
    )


@pytest.fixture
def inputs(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_bytes(example_urdf().read_bytes())
    nominal = NumpyBackend.from_urdf(urdf, end_link="ee_link")
    truth = nominal.perturbed(
        seed=13, translation_std_m=0.001, rotation_std_rad=0.002
    )
    joints = np.random.default_rng(72).uniform(-1.5, 1.5, (20, nominal.dof))
    dataset = tmp_path / "train.npz"
    KinematicDataset(
        joints,
        np.stack([truth.forward(row) for row in joints]),
        ObservationFrame.BASE_T_EE,
        joint_names=nominal.joint_names,
    ).save(dataset)
    return dataset, urdf


def _arguments(inputs, output):
    dataset, urdf = inputs
    return [
        "calibrate",
        "--backend",
        "numpy",
        "--calibration-type",
        "geometry",
        "--dataset",
        str(dataset),
        "--urdf",
        str(urdf),
        "--output-dir",
        str(output),
    ]


@pytest.mark.parametrize("filename", ["calibration_manifest.json", "final_result.json"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_summary_destinations_preserve_dataset_aliases(
    tmp_path, inputs, caplog, filename, alias
):
    dataset, _ = inputs
    original = dataset.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    destination = output / filename
    if alias == "symlink":
        dataset.unlink()
        destination.write_bytes(original)
        dataset.symlink_to(destination)
    else:
        os.link(dataset, destination)

    assert main([*_arguments(inputs, output), "--skip-handeye", "--force"]) == 2

    assert "output conflicts with input" in caplog.text
    assert dataset.read_bytes() == destination.read_bytes() == original
    assert not (output / "geometry_parameters.npz").exists()


@pytest.mark.parametrize("automatic", [False, True])
def test_geometry_destination_preserves_explicit_or_automatic_handeye_samples(
    tmp_path, inputs, caplog, automatic
):
    output = tmp_path / "output"
    output.mkdir()
    samples = output / "geometry_parameters.npz"
    _handeye_samples(samples)
    original = samples.read_bytes()
    arguments = [*_arguments(inputs, output), "--force"]
    if automatic:
        (tmp_path / "handeye_observations.npz").symlink_to(samples)
    else:
        arguments.extend(["--handeye-samples", str(samples)])

    assert main(arguments) == 2

    assert "output conflicts with input" in caplog.text
    assert samples.read_bytes() == original
    assert not (output / "handeye_result.json").exists()
    assert not (output / "calibration_manifest.json").exists()


@pytest.mark.parametrize(
    "filename,calibration_type",
    [
        ("calibration_manifest.json", "geometry"),
        ("final_result.json", "geometry"),
        ("geometry_parameters.json", "geometry"),
        ("zero_offsets.json", "zero-offset"),
    ],
)
def test_custom_handeye_result_cannot_be_another_active_destination(
    tmp_path, inputs, caplog, filename, calibration_type
):
    output = tmp_path / "new-output"
    samples = tmp_path / "handeye.npz"
    _handeye_samples(samples)
    original = samples.read_bytes()
    arguments = _arguments(inputs, output)
    arguments[arguments.index("numpy")] = "rtb"
    arguments[arguments.index("geometry")] = calibration_type

    assert (
        main(
            [
                *arguments,
                "--handeye-samples",
                str(samples),
                "--handeye-result",
                str(output / filename),
                "--force",
            ]
        )
        == 2
    )

    assert "output conflicts with input" in caplog.text
    assert samples.read_bytes() == original
    assert not output.exists()


@pytest.mark.parametrize("source_index", [0, 1])
def test_recomputed_handeye_result_preserves_dataset_or_urdf_alias(
    tmp_path, inputs, caplog, source_index
):
    source = inputs[source_index]
    original = source.read_bytes()
    result = tmp_path / "custom-handeye.json"
    source.unlink()
    result.write_bytes(original)
    source.symlink_to(result)
    samples = tmp_path / "handeye.npz"
    _handeye_samples(samples)
    output = tmp_path / "new-output"

    assert (
        main(
            [
                *_arguments(inputs, output),
                "--handeye-samples",
                str(samples),
                "--handeye-result",
                str(result),
                "--force",
            ]
        )
        == 2
    )

    assert "output conflicts with input" in caplog.text
    assert source.read_bytes() == result.read_bytes() == original
    assert not output.exists()


def test_accepted_handeye_extrinsic_destination_preserves_urdf_before_solving(
    tmp_path, inputs, caplog
):
    pytest.importorskip("cv2")
    from kinematic_calibration.handeye_cli import main as handeye_main

    dataset_path, urdf = inputs
    dataset = KinematicDataset.load(dataset_path)
    KinematicDataset(
        dataset.joint_positions,
        dataset.observed_poses,
        ObservationFrame.CAMERA_T_TARGET,
        joint_names=dataset.joint_names,
    ).save(dataset_path)
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "base_T_camera.npy"
    original = urdf.read_bytes()
    urdf.unlink()
    destination.write_bytes(original)
    urdf.symlink_to(destination)
    samples, result = tmp_path / "handeye.npz", tmp_path / "handeye.json"
    _handeye_samples(samples)
    assert (
        handeye_main(
            ["solve", "--samples", str(samples), "--output", str(result), "--method", "li"]
        )
        == 0
    )
    result_bytes = result.read_bytes()

    assert (
        main(
            [
                *_arguments(inputs, output),
                "--handeye-samples",
                str(samples),
                "--handeye-result",
                str(result),
                "--force",
            ]
        )
        == 2
    )

    assert "output conflicts with input" in caplog.text
    assert urdf.is_symlink()
    assert urdf.read_bytes() == destination.read_bytes() == original
    assert result.read_bytes() == result_bytes
    assert not (output / "ee_T_target.npy").exists()
    assert not (output / "geometry_parameters.npz").exists()


@pytest.mark.parametrize(
    "camera,handeye,suffix",
    [
        (False, False, "calibrated"),
        (True, False, "self_calibrated"),
        (True, True, "calibrated"),
        (True, True, "self_calibrated"),
    ],
)
def test_rtb_export_destinations_preserve_source_urdf_aliases(
    tmp_path, inputs, caplog, camera, handeye, suffix
):
    dataset_path, urdf = inputs
    if camera:
        dataset = KinematicDataset.load(dataset_path)
        KinematicDataset(
            dataset.joint_positions,
            dataset.observed_poses,
            ObservationFrame.CAMERA_T_TARGET,
        ).save(dataset_path)
    output = tmp_path / "output"
    output.mkdir()
    destination = output / f"robot_{suffix}.urdf"
    original = urdf.read_bytes()
    destination.write_bytes(original)
    urdf.unlink()
    urdf.symlink_to(destination)
    arguments = _arguments(inputs, output)
    arguments[arguments.index("numpy")] = "rtb"
    if handeye:
        samples = tmp_path / "handeye.npz"
        _handeye_samples(samples)
        arguments.extend(["--handeye-samples", str(samples)])
    else:
        arguments.append("--skip-handeye")

    assert main([*arguments, "--force"]) == 2

    assert "output conflicts with input" in caplog.text
    assert urdf.read_bytes() == original
    assert not (output / "geometry_parameters.npz").exists()


def test_numpy_default_success_preserves_nominal_urdf_inside_output(tmp_path, inputs):
    dataset, urdf = inputs
    output = tmp_path / "output"
    output.mkdir()
    archived_urdf = output / "nominal.urdf"
    archived_urdf.write_bytes(urdf.read_bytes())
    original = archived_urdf.read_bytes()
    stale_zero = output / "zero_offsets.json"
    stale_zero.write_text("unused zero-offset artifact")

    assert (
        main([*_arguments((dataset, archived_urdf), output), "--skip-handeye"])
        == 0
    )

    assert archived_urdf.read_bytes() == original
    assert stale_zero.read_text() == "unused zero-offset artifact"
    report = json.loads((output / "final_result.json").read_text())
    assert report["accepted"]
    assert report["kinematic_calibration"]["parameters"] == str(
        output / "geometry_parameters.npz"
    )


@pytest.mark.parametrize("mode", ["generate", "reuse", "recompute"])
def test_independent_custom_handeye_result_remains_a_legal_destination(
    tmp_path, inputs, monkeypatch, mode
):
    pytest.importorskip("cv2")
    import kinematic_calibration.handeye_cli as handeye_cli

    output = tmp_path / "output"
    samples = tmp_path / "handeye.npz"
    _handeye_samples(samples)
    # This name is unused by the selected geometry-only calibration stage.
    result = output / "zero_offsets.json"
    if mode != "generate":
        assert (
            handeye_cli.main(
                [
                    "solve",
                    "--samples",
                    str(samples),
                    "--output",
                    str(result),
                    "--method",
                    "li",
                ]
            )
            == 0
        )
    original_solve = handeye_cli._solve
    calls = []

    def tracked_solve(args):
        calls.append(args.output)
        return original_solve(args)

    monkeypatch.setattr(handeye_cli, "_solve", tracked_solve)
    arguments = [
        *_arguments(inputs, output),
        "--handeye-samples",
        str(samples),
        "--handeye-result",
        str(result),
        "--opencv-method",
        "li",
    ]
    if mode == "recompute":
        arguments.append("--force")

    assert main(arguments) == 0

    assert calls == ([] if mode == "reuse" else [result])
    assert json.loads(result.read_text())["calibration_type"] == "eye-to-hand"
    assert json.loads((output / "final_result.json").read_text())["accepted"]
