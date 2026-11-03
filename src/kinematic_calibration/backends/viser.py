"""Optional browser visualization for portable headless experiments."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..experiment import load_experiment
from ..targets import BoardConfig, get_board_config
from .numpy import NumpyBackend


def _frame(scene, name: str, pose: np.ndarray, length: float = 0.08):
    xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return scene.add_frame(
        name,
        position=pose[:3, 3],
        wxyz=xyzw[[3, 0, 1, 2]],
        axes_length=length,
        axes_radius=0.003,
    )


def create_viewer(experiment: Path | str, *, host: str = "127.0.0.1", port: int = 8080):
    """Create a Viser server and return it; callers own its lifecycle.

    Robot skeletons come from URDF geometry. Calibrated endpoint predictions
    come from the fitted POE, so the overlay displays numerical model outputs.
    """
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "Viser is optional; install 'kinematic-calibration[visualization]'"
        ) from exc

    manifest = json.loads((Path(experiment) / "manifest.json").read_text())
    image_experiment = manifest.get("artifact_type") == "image-kinematic-experiment"
    if image_experiment:
        from ..image_simulation import load_image_experiment

        report, dataset, model = load_image_experiment(experiment)
    else:
        report, dataset, model = load_experiment(experiment)
    nominal = NumpyBackend.from_urdf(
        Path(experiment) / "nominal.urdf",
        end_link=report["end_link"],
        base_link=report["base_link"],
    )
    truth = nominal.perturbed(seed=report["truth_seed"])
    if image_experiment:
        import trimesh
        from PIL import Image

        board = BoardConfig(**report["board_geometry"])
        with Image.open(Path(experiment) / "board.png") as texture:
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
    else:
        board = get_board_config()
        board_mesh = board.mesh()
        ee_t_board = np.eye(4)
    server = viser.ViserServer(host=host, port=port, label="Kinematic Calibration")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=3, height=3)
    _frame(server.scene, "/base", np.eye(4), 0.2)
    if image_experiment:
        _frame(server.scene, "/camera", model["base_T_camera"], 0.15)
    palette = {
        "nominal": (230, 155, 60),
        "truth": (60, 190, 130),
        "calibrated": (70, 140, 255),
        "measured": (235, 90, 120),
    }
    for name in ("nominal", "truth", "calibrated"):
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
            model["truth_predictions"][:, :3, 3],
            model["nominal_predictions"][:, :3, 3],
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
        client.camera.position = center + np.array([1.7, -2.2, 1.5]) * radius
        client.camera.look_at = center
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = np.deg2rad(45.0)

    server.gui.add_markdown(
        "**Kinematic calibration**\n\n"
        "Orange: nominal · Green: truth · Blue: calibrated · Pink: measured\n\n"
        f"Held-out position RMSE: {report['nominal_vs_measurements']['position_rmse_m'] * 1000:.3f} "
        f"→ {report['calibrated_vs_measurements']['position_rmse_m'] * 1000:.3f} mm\n\n"
        f"Termination: `{report['termination_reason']}`"
    )
    slider = server.gui.add_slider(
        "Validation pose",
        min=0,
        max=len(dataset.joint_positions) - 1,
        step=1,
        initial_value=0,
    )
    error = server.gui.add_markdown("")
    show_board = server.gui.add_checkbox("Show calibration target", initial_value=True)
    board_handle = server.scene.add_mesh_trimesh("/target", board_mesh)
    preview = None
    validation_frames = []
    if image_experiment and report["measurement_model"] == "rendered-image-detection":
        capture = Path(experiment) / "captures/validation"
        validation_frames = json.loads((capture / "manifest.json").read_text())[
            "samples"
        ]
        preview = server.gui.add_image(
            np.zeros((32, 32, 3), dtype=np.uint8), label="Validation image"
        )

    def update(_event=None):
        index = int(slider.value)
        q = dataset.joint_positions[index]
        with server.atomic():
            for name, backend in (("nominal", nominal), ("truth", truth)):
                positions = np.asarray(
                    [pose[:3, 3] for pose in backend.link_transforms(q).values()]
                )
                server.scene.add_line_segments(
                    f"/robot/{name}",
                    points=np.stack([positions[:-1], positions[1:]], axis=1),
                    colors=palette[name],
                    thickness=0.012,
                )
            for name in ("nominal", "truth", "calibrated"):
                _frame(
                    server.scene,
                    f"/current/{name}",
                    model[f"{name}_predictions"][index],
                )
            _frame(
                server.scene, "/current/measured", dataset.observed_poses[index], 0.055
            )
            board_pose = model["truth_predictions"][index] @ ee_t_board
            board_handle.position = board_pose[:3, 3]
            board_handle.wxyz = Rotation.from_matrix(board_pose[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ]
            board_handle.visible = show_board.value
            if preview is not None:
                with Image.open(capture / validation_frames[index]["image"]) as frame:
                    preview.image = np.asarray(frame.convert("RGB"))
            measured = dataset.observed_poses[index, :3, 3]
            fitted = model["calibrated_predictions"][index, :3, 3]
            error.content = f"Selected endpoint error: **{np.linalg.norm(fitted - measured) * 1000:.3f} mm**"

    slider.on_update(update)
    show_board.on_update(update)
    update()
    return server
