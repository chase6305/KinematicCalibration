"""Optional browser visualization for portable headless experiments."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..experiment import load_experiment
from ..targets import get_board_config
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

    report, dataset, model = load_experiment(experiment)
    nominal = NumpyBackend.from_urdf(
        Path(experiment) / "nominal.urdf",
        end_link=report["end_link"],
        base_link=report["base_link"],
    )
    truth = nominal.perturbed(seed=report["truth_seed"])
    board = get_board_config()
    board_mesh = board.mesh()
    server = viser.ViserServer(host=host, port=port, label="Kinematic Calibration")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=3, height=3)
    _frame(server.scene, "/base", np.eye(4), 0.2)
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
    show_board = server.gui.add_checkbox(
        "Show target at truth EE frame", initial_value=True
    )
    board_handle = server.scene.add_mesh_trimesh("/target", board_mesh)

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
            board_pose = model["truth_predictions"][index]
            board_handle.position = board_pose[:3, 3]
            board_handle.wxyz = Rotation.from_matrix(board_pose[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ]
            board_handle.visible = show_board.value
            measured = dataset.observed_poses[index, :3, 3]
            fitted = model["calibrated_predictions"][index, :3, 3]
            error.content = f"Selected endpoint error: **{np.linalg.norm(fitted - measured) * 1000:.3f} mm**"

    slider.on_update(update)
    show_board.on_update(update)
    update()
    return server
