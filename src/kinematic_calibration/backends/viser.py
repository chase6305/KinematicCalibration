"""Optional browser replay of experiments and recorded calibration data."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..targets import BoardConfig, get_board_config
from .playback import ExperimentPlayback


def _frame(scene, name: str, pose: np.ndarray, length: float = 0.08):
    xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return scene.add_frame(
        name,
        position=pose[:3, 3],
        wxyz=xyzw[[3, 0, 1, 2]],
        axes_length=length,
        axes_radius=0.003,
    )


def create_viewer(
    experiment: Path | str,
    *,
    split: str = "validation",
    host: str = "127.0.0.1",
    port: int = 8080,
):
    """Create a Viser server and return it; callers own its lifecycle.

    Robot skeletons come from URDF geometry. Calibrated endpoint predictions
    come from the fitted POE, so the overlay displays numerical model outputs.
    """
    return _create_viewer(
        ExperimentPlayback(experiment, split=split), host=host, port=port
    )


def create_fitted_viewer(
    parameters: Path | str,
    dataset: Path | str,
    urdf: Path | str,
    *,
    result: Path | str | None = None,
    base_t_camera: Path | str | None = None,
    ee_t_target: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
):
    """Replay fitted parameters against recorded data without inventing truth."""
    from .measured import MeasuredPlayback

    playback = MeasuredPlayback(
        parameters,
        dataset,
        urdf,
        result=result,
        base_t_camera=base_t_camera,
        ee_t_target=ee_t_target,
    )
    return _create_viewer(playback, host=host, port=port)


def _create_viewer(playback, *, host: str, port: int):
    """Render either playback source; measured data has no truth or target mesh."""
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "Viser is optional; install 'kinematic-calibration[visualization]'"
        ) from exc

    report, dataset, model = playback.report, playback.dataset, playback.model
    nominal = playback.nominal
    synthetic = "truth_predictions" in model
    training = synthetic and playback.split == "training"
    image_experiment = (
        synthetic and report.get("experiment_type") == "image-kinematic-calibration"
    )
    board_mesh = None
    ee_t_board = None
    if image_experiment:
        import trimesh
        from PIL import Image

        board = BoardConfig(**report["board_geometry"])
        with Image.open(playback.path / "board.png") as texture:
            board_mesh = trimesh.Trimesh(
                vertices=board.corners,
                faces=[[0, 2, 1], [0, 3, 2]],
                visual=trimesh.visual.TextureVisuals(
                    uv=[[0, 1], [1, 1], [1, 0], [0, 0]],
                    image=texture.copy(),
                ),
                process=False,
            )
        ee_t_board = model["ee_T_board"]
    elif synthetic:
        board = get_board_config()
        board_mesh = board.mesh()
        ee_t_board = np.eye(4)
    coordinate_frame = report.get("coordinate_frame", "base")
    camera_coordinates = coordinate_frame == "camera"
    server = viser.ViserServer(host=host, port=port, label="Kinematic Calibration")
    # OpenCV optical coordinates have Y down; robot-base coordinates have Z up.
    server.scene.set_up_direction("-y" if camera_coordinates else "+z")
    server.scene.add_grid(
        "/grid", width=3, height=3, plane="xz" if camera_coordinates else "xy"
    )
    _frame(server.scene, f"/{coordinate_frame}", np.eye(4), 0.2)
    if image_experiment:
        _frame(server.scene, "/camera", model["base_T_camera"], 0.15)
    elif "camera_T_base_initialization" in model:
        _frame(
            server.scene,
            "/nominal_base_initialization",
            model["camera_T_base_initialization"],
            0.15,
        )
    palette = {
        "nominal": (230, 155, 60),
        "truth": (60, 190, 130),
        "calibrated": (70, 140, 255),
        "measured": (235, 90, 120),
    }
    prediction_names = tuple(
        name
        for name in ("nominal", "truth", "calibrated")
        if f"{name}_predictions" in model
    )
    for name in prediction_names:
        server.scene.add_point_cloud(
            f"/poses/{name}",
            points=model[f"{name}_predictions"][:, :3, 3],
            colors=palette[name],
            point_size=0.008,
        )
    server.scene.add_point_cloud(
        "/poses/measured",
        points=dataset.observed_poses[:, :3, 3],
        colors=palette["measured"],
        point_size=0.004,
    )
    display_points = np.vstack(
        [
            *(model[f"{name}_predictions"][:, :3, 3] for name in prediction_names),
            dataset.observed_poses[:, :3, 3],
            np.zeros((1, 3)),
        ]
    )
    center = (display_points.min(axis=0) + display_points.max(axis=0)) / 2
    radius = max(
        float(np.linalg.norm(display_points.max(axis=0) - display_points.min(axis=0)))
        / 2,
        0.45,
    )

    @server.on_client_connect
    def initialize_camera(client):
        offset = [1.7, -1.5, -2.2] if camera_coordinates else [1.7, -2.2, 1.5]
        client.camera.position = center + np.array(offset) * radius
        client.camera.look_at = center
        client.camera.up_direction = (
            (0.0, -1.0, 0.0) if camera_coordinates else (0.0, 0.0, 1.0)
        )
        client.camera.fov = np.deg2rad(45.0)

    legend = "Orange: nominal · "
    if synthetic:
        legend += "Green: truth · "
    legend += "Blue: calibrated · Pink: measured"
    role = report.get("data_role", "held-out")
    summary = (
        "**Kinematic calibration**\n\n"
        f"{legend}\n\n"
        f"{role.capitalize()} position RMSE: {report['nominal_vs_measurements']['position_rmse_m'] * 1000:.3f} "
        f"→ {report['calibrated_vs_measurements']['position_rmse_m'] * 1000:.3f} mm\n\n"
        f"Termination: `{report['termination_reason']}`"
    )
    if not synthetic:
        summary += (
            f"\n\nCoordinates: {coordinate_frame} · "
            f"Stored fit accepted: {report['accepted']}"
        )
        if report.get("display_note"):
            summary += f"\n\n{report['display_note']}"
    server.gui.add_markdown(summary)
    slider = server.gui.add_slider(
        ("Training pose" if training else "Validation pose")
        if synthetic
        else "Recorded pose",
        min=0,
        max=len(dataset.joint_positions) - 1,
        step=1,
        initial_value=0,
    )
    manual_mode = server.gui.add_checkbox("Explore joint angles", initial_value=False)
    joint_sliders = []
    with server.gui.add_folder("Joint angles (rad)"):
        for index, name in enumerate(nominal.joint_names):
            lower, upper = nominal.joint_limits[index]
            joint_sliders.append(
                server.gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper),
                    step=0.001,
                    initial_value=float(dataset.joint_positions[0, index]),
                    disabled=True,
                )
            )
    error = server.gui.add_markdown("")
    show_board = None
    board_handle = None
    if board_mesh is not None:
        show_board = server.gui.add_checkbox(
            "Show calibration target", initial_value=True
        )
        board_handle = server.scene.add_mesh_trimesh("/target", board_mesh)
    preview = None
    capture_frames = {}
    if image_experiment and report["measurement_model"] == "rendered-image-detection":
        capture = playback.path / "captures" / ("train" if training else "validation")
        capture_frames = {
            row["sample_id"]: row
            for row in json.loads((capture / "manifest.json").read_text())["samples"]
        }
        preview = server.gui.add_image(
            np.zeros((32, 32, 3), dtype=np.uint8),
            label="Training image" if training else "Validation image",
        )

    def update(_event=None):
        index = int(slider.value)
        state = (
            playback.manual([joint.value for joint in joint_sliders])
            if manual_mode.value
            else playback.sample(index)
        )
        with server.atomic():
            for name, links in state.links.items():
                positions = np.asarray([pose[:3, 3] for pose in links.values()])
                server.scene.add_line_segments(
                    f"/robot/{name}",
                    points=np.stack([positions[:-1], positions[1:]], axis=1),
                    colors=palette[name],
                    thickness=0.012,
                )
            for name, pose in state.poses.items():
                _frame(
                    server.scene,
                    f"/current/{name}",
                    pose,
                )
            measured_frame = _frame(
                server.scene,
                "/current/measured",
                np.eye(4) if state.measured_pose is None else state.measured_pose,
                0.055,
            )
            measured_frame.visible = state.measured_pose is not None
            if board_handle is not None:
                board_pose = state.poses["truth"] @ ee_t_board
                board_handle.position = board_pose[:3, 3]
                board_handle.wxyz = Rotation.from_matrix(board_pose[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                board_handle.visible = show_board.value
            if preview is not None:
                preview.visible = not manual_mode.value
                if preview.visible:
                    with Image.open(
                        capture / capture_frames[state.sample_index]["image"]
                    ) as frame:
                        preview.image = np.asarray(frame.convert("RGB"))
            if state.position_error_m is None or state.orientation_error_rad is None:
                error.content = "No measurement for this configuration."
            else:
                error.content = (
                    f"Model vs {state.error_reference}: **{state.position_error_m * 1000:.3f} mm**, "
                    f"**{np.rad2deg(state.orientation_error_rad):.4f}°**"
                )
                if training and "training_plan" in report and not manual_mode.value:
                    candidate = report["training_plan"]["candidate_indices"][index]
                    error.content += f"\n\nTraining row: {index} · Candidate: {candidate}"

    def change_mode(_event):
        with server.atomic():
            slider.disabled = manual_mode.value
            for index, joint in enumerate(joint_sliders):
                joint.disabled = not manual_mode.value
                if manual_mode.value:
                    joint.value = float(
                        dataset.joint_positions[int(slider.value), index]
                    )
        update()

    slider.on_update(update)
    if show_board is not None:
        show_board.on_update(update)
    manual_mode.on_update(change_mode)
    for joint in joint_sliders:
        joint.on_update(update)
    update()
    return server
