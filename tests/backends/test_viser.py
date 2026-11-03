import json
import urllib.request

import numpy as np
import pytest

from kinematic_calib.dataset import KinematicDataset
from kinematic_calibration.backends.viser import create_viewer
from kinematic_calibration.experiment import pose_metrics, predict_poe, run_experiment

viser = pytest.importorskip("viser")


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    path = tmp_path_factory.mktemp("viewer")
    run_experiment(path, samples=12, validation_samples=8)
    return path


@pytest.fixture
def controls(monkeypatch):
    handles = {"sliders": {}, "checkboxes": {}, "images": {}, "markdown": []}
    add_slider = viser.GuiApi.add_slider
    add_checkbox = viser.GuiApi.add_checkbox
    add_markdown = viser.GuiApi.add_markdown
    add_image = viser.GuiApi.add_image

    def record_slider(gui, label, *args, **kwargs):
        handle = add_slider(gui, label, *args, **kwargs)
        handles["sliders"][label] = handle
        return handle

    def record_checkbox(gui, label, *args, **kwargs):
        handle = add_checkbox(gui, label, *args, **kwargs)
        handles["checkboxes"][label] = handle
        return handle

    def record_markdown(gui, *args, **kwargs):
        handle = add_markdown(gui, *args, **kwargs)
        handles["markdown"].append(handle)
        return handle

    def record_image(gui, *args, **kwargs):
        handle = add_image(gui, *args, **kwargs)
        handles["images"][kwargs["label"]] = handle
        return handle

    monkeypatch.setattr(viser.GuiApi, "add_slider", record_slider)
    monkeypatch.setattr(viser.GuiApi, "add_checkbox", record_checkbox)
    monkeypatch.setattr(viser.GuiApi, "add_markdown", record_markdown)
    monkeypatch.setattr(viser.GuiApi, "add_image", record_image)
    return handles


@pytest.mark.parametrize("split", ["validation", "training"])
def test_viewer_serves_selected_poses_and_target_mesh(experiment, controls, split):
    filename = "train.npz" if split == "training" else "validation.npz"
    data = KinematicDataset.load(experiment / filename)
    with np.load(experiment / "model.npz", allow_pickle=False) as archive:
        predictions = predict_poe(
            archive["screw_axes"], archive["home"], data.joint_positions
        )
        nominal = predict_poe(
            archive["nominal_axes"], archive["nominal_home"], data.joint_positions
        )
    # Exercise the public default as well as explicit training selection.
    kwargs = {"split": split} if split == "training" else {}
    server = create_viewer(experiment, port=0, **kwargs)
    try:
        url = f"http://127.0.0.1:{server.get_port()}"
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.status == 200
            assert b"<html" in response.read().lower()
        assert server.scene.get_handle_by_name("/target") is not None
        for name, expected in (
            ("measured", data.observed_poses[:, :3, 3]),
            ("calibrated", predictions[:, :3, 3]),
        ):
            points = server.scene.get_handle_by_name(f"/poses/{name}").points
            # Viser may compress point-cloud coordinates for transport.
            np.testing.assert_array_equal(points, expected.astype(points.dtype))
        label = "Training pose" if split == "training" else "Validation pose"
        slider = controls["sliders"][label]
        assert slider.max == len(data.joint_positions) - 1
        slider.value = 3
        np.testing.assert_allclose(
            server.scene.get_handle_by_name("/current/measured").position,
            data.observed_poses[3, :3, 3],
        )
        np.testing.assert_allclose(
            server.scene.get_handle_by_name("/current/calibrated").position,
            predictions[3, :3, 3],
            atol=1e-12,
        )
        reference = (
            "training measurement" if split == "training" else "held-out measurement"
        )
        assert reference in controls["markdown"][-1].content
        nominal_rmse = pose_metrics(nominal, data.observed_poses)["position_rmse_m"]
        fitted_rmse = pose_metrics(predictions, data.observed_poses)["position_rmse_m"]
        role = "Training" if split == "training" else "Held-out"
        assert (
            f"{role} position RMSE: {nominal_rmse * 1000:.3f} → {fitted_rmse * 1000:.3f} mm"
            in controls["markdown"][0].content
        )
        controls["checkboxes"]["Explore joint angles"].value = True
        assert "synthetic truth" in controls["markdown"][-1].content
        assert not server.scene.get_handle_by_name("/current/measured").visible
    finally:
        server.stop()


@pytest.fixture(scope="module")
def image_experiment(tmp_path_factory):
    pytest.importorskip("cv2")
    from kinematic_calibration.image_simulation import ImageNoise, run_image_experiment

    path = tmp_path_factory.mktemp("image-viewer") / "experiment"
    run_image_experiment(
        path, samples=12, validation_samples=5, noise=ImageNoise(0, 0, 1)
    )
    return path


@pytest.mark.parametrize("split", ["validation", "training"])
def test_image_preview_tracks_archived_sample_identity(
    image_experiment, controls, split
):
    from PIL import Image

    filename = "train.npz" if split == "training" else "validation.npz"
    data = KinematicDataset.load(image_experiment / filename)
    role = "train" if split == "training" else "validation"
    capture = image_experiment / "captures" / role
    records = json.loads((capture / "manifest.json").read_text())["samples"]
    frames = {row["sample_id"]: row["image"] for row in records}
    label = "Training" if split == "training" else "Validation"
    server = create_viewer(image_experiment, split=split, port=0)
    try:
        preview = controls["images"][f"{label} image"]
        slider = controls["sliders"][f"{label} pose"]
        for index in (0, 2, len(data.joint_positions) - 1):
            slider.value = index
            sample_id = int(data.sample_indices[index])
            with Image.open(capture / frames[sample_id]) as image:
                np.testing.assert_array_equal(
                    preview.image, np.asarray(image.convert("RGB"))
                )
            np.testing.assert_allclose(
                server.scene.get_handle_by_name("/current/measured").position,
                data.observed_poses[index, :3, 3],
            )
        controls["checkboxes"]["Explore joint angles"].value = True
        assert not preview.visible
        assert "synthetic truth" in controls["markdown"][-1].content
        controls["checkboxes"]["Explore joint angles"].value = False
        assert preview.visible
    finally:
        server.stop()


def test_supplied_plan_displays_original_source_sample_id(tmp_path, controls):
    from kinematic_calibration.planning import load_pose_plan, save_pose_plan

    source = tmp_path / "chosen.npz"
    sample_ids = np.arange(12) * 7 + 91
    np.savez_compressed(
        source,
        joint_positions=np.random.default_rng(42).uniform(-1, 1, (12, 6)),
        sample_indices=sample_ids,
        joint_names=np.asarray([f"Joint{i}" for i in range(1, 7)]),
    )
    plan_path = tmp_path / "plan.npz"
    save_pose_plan(plan_path, input_joints=source)
    plan = load_pose_plan(plan_path)
    experiment = tmp_path / "experiment"
    run_experiment(experiment, plan=plan_path, validation_samples=5)
    server = create_viewer(experiment, split="training", port=0)
    try:
        slider = controls["sliders"]["Training pose"]
        for row in (0, 11):
            slider.value = row
            candidate = int(plan.candidate_indices[row])
            assert (
                f"Training row: {row} · Candidate: {candidate} · Source sample: {sample_ids[candidate]}"
                in controls["markdown"][-1].content
            )
        controls["checkboxes"]["Explore joint angles"].value = True
        assert "Source sample:" not in controls["markdown"][-1].content
    finally:
        server.stop()
