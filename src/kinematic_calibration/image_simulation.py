"""Reproducible camera-image experiments without a simulator dependency.

This is a pinhole image-formation experiment, not a dynamics/occlusion engine.
In image mode, every calibration observation comes from detecting a rendered
PNG with OpenCV. Robot truth poses are used only for rendering and evaluation.
"""

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from kinematic_calib.dataset import (
    KinematicDataset,
    ObservationFrame,
    invert_transform,
    minimum_full_pose_samples,
    validated_joint_names,
)
from kinematic_calib.io_utils import atomic_savez_compressed, atomic_write_json
from kinematic_calib.service import FitOptions, fit_dataset
from kinematic_calib.validation import validate_saved_poe_parameters

from .backends.numpy import NumpyBackend
from .experiment import (
    _check_metrics,
    _diagnostics,
    example_urdf,
    pose_metrics,
    predict_poe,
)
from .handeye import _pose_values
from .targets import BoardConfig, BoardType, get_board_config
from .targets.detection import (
    detect_opencv_circles,
    detect_toolkit_board,
    pose_from_correspondences,
)


@dataclass(frozen=True)
class ImageCamera:
    """Zero-distortion pinhole camera, in optical (+X right,+Y down,+Z forward)."""

    width: int = 1280
    height: int = 960
    focal_length_px: float = 1100.0

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or value < 64:
                raise ValueError(f"{name} must be an integer >= 64")
        if not np.isfinite(self.focal_length_px) or self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive and finite")

    @property
    def intrinsic(self):
        return np.array(
            [
                [self.focal_length_px, 0, self.width / 2],
                [0, self.focal_length_px, self.height / 2],
                [0, 0, 1.0],
            ]
        )


@dataclass(frozen=True)
class ImageNoise:
    """Image-space noise applied before board detection; no pose noise is added."""

    rgb_std: float = 0.5
    blur_sigma_px: float = 0.3
    supersample: int = 2

    def __post_init__(self):
        if (
            not np.isfinite([self.rgb_std, self.blur_sigma_px]).all()
            or min(self.rgb_std, self.blur_sigma_px) < 0
        ):
            raise ValueError("image noise scales must be finite and nonnegative")
        if type(self.supersample) is not int or not 1 <= self.supersample <= 4:
            raise ValueError("supersample must be an integer in [1,4]")


def default_camera_pose():
    """A fixed Base_T_Camera overlooking a useful part of the UR10 workspace."""
    position, target = np.array([0.8, -1.3, 0.9]), np.array([-0.2, -0.2, 0.45])
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    value = np.eye(4)
    value[:3, :3] = np.column_stack([right, down, forward])
    value[:3, 3] = position
    return value


def default_board_mount(board: BoardConfig):
    """Center the plate over the terminal axis with an 80 mm standoff."""
    value = np.eye(4)
    value[:3, 3] = [
        -board.width_m / 2 + board.origin_from_top_left_m[0],
        -board.height_m / 2 + board.origin_from_top_left_m[1],
        0.08,
    ]
    return value


def _svg_texture_controls(board, width_px, height_px):
    root = ET.parse(board.asset("svg")).getroot()
    view_box = root.get("viewBox")
    if view_box:
        vx, vy, vw, vh = map(float, view_box.split())
    else:
        vx, vy = 0.0, 0.0
        vw, vh = (
            float(root.get("width").removesuffix("mm")),
            float(root.get("height").removesuffix("mm")),
        )
    centers = {
        (
            float(node.get("cx").removesuffix("mm")),
            float(node.get("cy").removesuffix("mm")),
        )
        for node in root.findall(".//{http://www.w3.org/2000/svg}circle")
        if node.get("fill") == "black"
    }
    if len(centers) != len(board.points):
        raise ValueError("printable SVG marker count does not match configured board")
    centers = np.asarray(sorted(centers, key=lambda p: (p[1], p[0])))
    return (centers - [vx, vy]) / [vw, vh] * [width_px, height_px] - 0.5


class PlanarBoardRenderer:
    """Project retained board artwork into a pinhole image using metric geometry.

    Circle artwork is registered once against its known control points to
    account for the source PNG's integer canvas dimensions. This establishes
    the static texture's physical scale. Per-frame measurements always run a
    fresh image detector after projection, antialiasing, blur and pixel noise.
    """

    def __init__(
        self,
        board: BoardConfig,
        camera: ImageCamera | None = None,
        noise: ImageNoise | None = None,
        *,
        texture_path: Path | str | None = None,
        registration_backend: str = "opencv",
    ):
        import cv2 as cv

        self.board, self.camera, self.noise = (
            board,
            camera or ImageCamera(),
            noise or ImageNoise(),
        )
        self.texture_path = (
            Path(texture_path) if texture_path is not None else board.asset("png")
        )
        self.texture_bytes = self.texture_path.read_bytes()
        texture = cv.imdecode(
            np.frombuffer(self.texture_bytes, np.uint8), cv.IMREAD_COLOR
        )
        if texture is None:
            raise ValueError("board texture PNG could not be decoded")
        self.texture = cv.cvtColor(texture, cv.COLOR_BGR2RGB)
        texture_height, texture_width = texture.shape[:2]
        if board.board_type == BoardType.OPENCV_ACIRCLE:
            gray = cv.cvtColor(self.texture, cv.COLOR_RGB2GRAY)
            found, points = cv.findCirclesGrid(
                cv.bitwise_not(gray),
                board.toolkit_size,
                flags=cv.CALIB_CB_ASYMMETRIC_GRID | cv.CALIB_CB_CLUSTERING,
            )
            if not found:
                raise ValueError(
                    "source PNG does not contain the configured circle grid"
                )
            controls = points[:, 0]
            registration_method = "static-circle-control-point-registration"
        elif (
            board.board_type == BoardType.CCTAG_MULTI_V3
            and registration_backend == "toolkit"
        ):
            # Actual native ID decoding establishes the texture's physical
            # control points. No projected ideal center is a measurement.
            virtual_camera = np.array(
                [
                    [texture_width, 0, texture_width / 2],
                    [0, texture_width, texture_height / 2],
                    [0, 0, 1.0],
                ]
            )
            detected = detect_toolkit_board(
                self.texture,
                virtual_camera,
                board=board,
                max_board_tilt_rad=np.pi,
            )
            if detected is None or detected.image_points is None:
                raise ValueError(
                    "native detector could not decode the complete v3 source PNG"
                )
            controls = detected.image_points
            registration_method = "static-native-cctag-control-point-registration"
        elif board.is_cctag_multi:
            # The external-centers mode supports the retained legacy artwork
            # without claiming native decoding. Verify the exact known PNG,
            # then read marker locations from its paired printable SVG.
            if self.texture_bytes != board.asset("png").read_bytes():
                raise ValueError(
                    "unrecognized CCTag texture requires actual native registration"
                )
            controls = _svg_texture_controls(board, texture_width, texture_height)
            registration_method = "static-svg-control-point-registration"
        else:
            raise ValueError("texture rendering requires a full grid board")
        self.board_to_texture, _ = cv.findHomography(board.points[:, :2], controls, 0)
        if self.board_to_texture is None:
            raise ValueError("board texture controls are degenerate")
        predicted = cv.perspectiveTransform(
            board.points[:, :2][None], self.board_to_texture
        )[0]
        residual = float(np.sqrt(np.mean(np.sum((predicted - controls) ** 2, axis=1))))
        if residual > 0.25:
            raise ValueError("source PNG grid disagrees with metric board geometry")
        self.registration = {"method": registration_method, "rms_source_px": residual}
        if self.board_to_texture is None or not np.all(
            np.isfinite(self.board_to_texture)
        ):
            raise ValueError("board texture registration is singular or nonfinite")
        self.texture_to_board = np.linalg.inv(self.board_to_texture)
        texture_corners = np.array(
            [
                [-0.5, -0.5],
                [texture_width - 0.5, -0.5],
                [texture_width - 0.5, texture_height - 0.5],
                [-0.5, texture_height - 0.5],
            ],
            np.float64,
        )
        registered_corners = cv.perspectiveTransform(
            texture_corners[None], self.texture_to_board
        )[0]
        if (
            np.max(np.linalg.norm(registered_corners - board.corners[:, :2], axis=1))
            > 0.001
        ):
            raise ValueError(
                "PNG canvas and declared board dimensions/origin disagree by more than 1 mm"
            )

    def project(self, points, camera_t_board):
        camera_points = (
            np.asarray(points) @ camera_t_board[:3, :3].T + camera_t_board[:3, 3]
        )
        if np.any(camera_points[:, 2] <= 0):
            raise ValueError("board has nonpositive optical depth")
        return camera_points[:, :2] / camera_points[
            :, 2:
        ] * self.camera.focal_length_px + [
            self.camera.width / 2,
            self.camera.height / 2,
        ]

    def visibility(self, camera_t_board):
        """Check the entire plate, printed side, perspective and marker spacing."""
        camera_t_board = _pose_values(camera_t_board, "Camera_T_Board")
        camera_corners = (
            self.board.corners @ camera_t_board[:3, :3].T + camera_t_board[:3, 3]
        )
        if np.any(camera_corners[:, 2] < 0.35):
            return False, "depth"
        center = (
            camera_t_board[:3, :3] @ self.board.points.mean(axis=0)
            + camera_t_board[:3, 3]
        )
        cosine = float(camera_t_board[:3, 2] @ center / np.linalg.norm(center))
        # +Z points into the printed face. Exclude backside and views close to
        # grazing or fronto-parallel, which have weak planar orientation evidence.
        if not 0.45 < cosine < 0.98:
            return False, "printed-side-or-tilt"
        pixels = self.project(self.board.corners, camera_t_board)
        if np.any(pixels < [20, 20]) or np.any(
            pixels > [self.camera.width - 20, self.camera.height - 20]
        ):
            return False, "field-of-view"
        centers = self.project(self.board.points, camera_t_board)
        distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        distances[np.diag_indices_from(distances)] = np.inf
        if np.min(distances) < 12:
            return False, "marker-resolution"
        return True, None

    def render(self, camera_t_board, generator: np.random.Generator | None = None):
        """Return an RGB uint8 image; requires an entirely visible front face."""
        import cv2 as cv

        camera_t_board = _pose_values(camera_t_board, "Camera_T_Board")
        visible, reason = self.visibility(camera_t_board)
        if not visible:
            raise ValueError(f"board is not renderable: {reason}")
        generator = generator or np.random.default_rng(0)
        scale = self.noise.supersample
        intrinsic = self.camera.intrinsic.copy()
        intrinsic[:2] *= scale
        # INTER_AREA's output pixel j averages high-resolution centers at
        # scale*j + (scale-1)/2. Preserve the original camera pixel centers.
        intrinsic[0, 2] += (scale - 1) / 2
        intrinsic[1, 2] += (scale - 1) / 2
        plane_to_camera = np.column_stack(
            [camera_t_board[:3, 0], camera_t_board[:3, 1], camera_t_board[:3, 3]]
        )
        homography = intrinsic @ plane_to_camera @ self.texture_to_board
        image = cv.warpPerspective(
            self.texture,
            homography,
            (self.camera.width * scale, self.camera.height * scale),
            flags=cv.INTER_LINEAR,
            borderValue=(150, 150, 150),
        )
        if scale != 1:
            image = cv.resize(
                image,
                (self.camera.width, self.camera.height),
                interpolation=cv.INTER_AREA,
            )
        if self.noise.blur_sigma_px > 0:
            image = cv.GaussianBlur(image, (0, 0), self.noise.blur_sigma_px)
        if self.noise.rgb_std > 0:
            image = np.clip(
                np.rint(
                    image.astype(float)
                    + generator.normal(0, self.noise.rgb_std, image.shape)
                ),
                0,
                255,
            ).astype(np.uint8)
        return image


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _collect(
    truth,
    renderer,
    base_t_camera,
    ee_t_board,
    count,
    seed,
    role,
    stage,
    measurement_mode,
    detector_backend,
    max_candidates,
    sample_id_offset,
):
    import cv2 as cv

    streams = np.random.SeedSequence(seed).spawn(2)
    joints_rng, pixels_rng = (np.random.default_rng(stream) for stream in streams)
    camera_t_base = invert_transform(base_t_camera)
    directory = stage / "captures" / role
    (directory / "frames").mkdir(parents=True)
    joints, observed, truth_poses, sample_ids, samples, attempts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rejected = Counter()
    lower, upper = (
        np.maximum(truth.joint_limits[:, 0], -np.pi),
        np.minimum(truth.joint_limits[:, 1], np.pi),
    )
    empty = lower >= upper
    lower[empty], upper[empty] = (
        truth.joint_limits[empty, 0],
        truth.joint_limits[empty, 1],
    )
    for candidate in range(max_candidates):
        q = joints_rng.uniform(lower, upper)
        base_t_ee = truth.forward(q)
        camera_t_board = camera_t_base @ base_t_ee @ ee_t_board
        visible, reason = renderer.visibility(camera_t_board)
        if not visible:
            rejected[reason] += 1
            continue
        sample_id = sample_id_offset + candidate
        image = renderer.render(camera_t_board, pixels_rng)
        filename = f"frames/{sample_id:08d}.png"
        encoded, buffer = cv.imencode(".png", cv.cvtColor(image, cv.COLOR_RGB2BGR))
        if not encoded:
            raise RuntimeError("PNG encoding failed")
        (directory / filename).write_bytes(buffer.tobytes())
        record = {
            "sample_id": sample_id,
            "image": filename,
            "image_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
            "joint_positions": q.tolist(),
            "joint_timestamp_s": candidate * 0.1,
            "image_timestamp_s": candidate * 0.1,
            "accepted": False,
        }
        try:
            if measurement_mode == "image":
                if detector_backend == "opencv":
                    detection = detect_opencv_circles(
                        image,
                        renderer.camera.intrinsic,
                        np.zeros(5),
                        board=renderer.board,
                        max_reprojection_rms_px=0.5,
                    )
                else:
                    # Match the detector defaults used by import_captures so
                    # accepted PNGs remain valid when the manifest is reimported.
                    detection = detect_toolkit_board(
                        image,
                        renderer.camera.intrinsic,
                        np.zeros(5),
                        board=renderer.board,
                    )
                    if detection is not None and (
                        detection.reprojection_rms_px is None
                        or detection.reprojection_rms_px > 0.5
                    ):
                        raise ValueError(
                            "native image reprojection RMS exceeds 0.5 px or is unavailable"
                        )
            else:
                centers = renderer.project(renderer.board.points, camera_t_board)
                centers += pixels_rng.normal(0, 0.2, centers.shape)
                detection = pose_from_correspondences(
                    renderer.board,
                    centers,
                    renderer.camera.intrinsic,
                    np.zeros(5),
                    max_reprojection_rms_px=0.5,
                )
                record["image_points"] = centers.tolist()
                # This mode does not claim the CCTag image was detected.
                record["center_noise_std_px"] = 0.2
            if detection is None:
                raise ValueError("complete grid was not detected")
            record["reprojection_rms_px"] = detection.reprojection_rms_px
            record["pose_diagnostics"] = detection.pose_diagnostics
            if detection.pose_diagnostics and detection.pose_diagnostics.get(
                "ambiguous"
            ):
                raise ValueError("ambiguous planar pose")
        except ValueError as exc:
            reason = str(exc)
            record["rejection_reason"] = reason
            rejected[reason] += 1
            attempts.append(record)
            continue
        record["accepted"] = True
        record["camera_T_board"] = detection.camera_T_board.tolist()
        attempts.append(record)
        joints.append(q)
        observed.append(detection.camera_T_board)
        truth_poses.append(base_t_ee)
        sample_ids.append(sample_id)
        sample = {
            key: record[key]
            for key in (
                "sample_id",
                "joint_positions",
                "joint_timestamp_s",
                "image_timestamp_s",
            )
        }
        if measurement_mode == "image":
            sample["image"] = filename
        else:
            sample["image_points"] = record["image_points"]
        samples.append(sample)
        if len(joints) == count:
            break
    if len(joints) != count:
        raise RuntimeError(
            f"{role}: collected {len(joints)}/{count} valid measurements after {max_candidates} candidates; rejection counts: {dict(rejected)}"
        )
    manifest = {
        "schema_version": 1,
        "observation_frame": "camera_T_target",
        "camera_convention": "opencv-optical",
        "joint_unit": "rad",
        "translation_unit": "m",
        "board_type": renderer.board.board_type.value,
        "joint_names": list(truth.joint_names),
        "camera_matrix": renderer.camera.intrinsic.tolist(),
        "distortion": [0.0] * 5,
        "detector_backend": detector_backend,
        "measurement_model": "rendered-image-detection"
        if measurement_mode == "image"
        else "simulated-external-centers",
        "samples": samples,
    }
    atomic_write_json(directory / "manifest.json", manifest)
    atomic_write_json(
        directory / "collection.json",
        {
            "candidates": candidate + 1,
            "rejection_counts": dict(rejected),
            "rendered_attempts": attempts,
        },
    )
    dataset = KinematicDataset(
        np.asarray(joints),
        np.asarray(observed),
        ObservationFrame.CAMERA_T_TARGET,
        np.asarray(sample_ids),
        joint_names=truth.joint_names,
    )
    return (
        dataset,
        np.asarray(truth_poses),
        {
            "candidates": candidate + 1,
            "accepted": len(joints),
            "rendered_images": len(attempts),
            "rejection_counts": dict(rejected),
            "mean_reprojection_rms_px": float(
                np.mean(
                    [row["reprojection_rms_px"] for row in attempts if row["accepted"]]
                )
            ),
        },
    )


def run_image_experiment(
    output: Path | str,
    *,
    urdf: Path | str | None = None,
    end_link: str = "ee_link",
    base_link: str | None = None,
    samples: int = 50,
    validation_samples: int = 20,
    seed: int = 42,
    board: BoardConfig | None = None,
    measurement_mode: str = "image",
    detector_backend: str = "opencv",
    camera: ImageCamera | None = None,
    noise: ImageNoise | None = None,
    base_t_camera=None,
    ee_t_board=None,
    max_candidates: int = 10000,
    max_iterations: int = 30,
):
    """Render, detect, fit, and evaluate a robot using independent image sets.

    ``image`` detects actual RGB pixels with OpenCV circles or the explicitly
    selected native toolkit and its matching CCTag v3 artwork.
    ``external-centers`` simulates noisy center coordinates for an external
    detector; rendered images are evidence rather than detector input in that
    mode. Extrinsics are known independently in both modes.

    Output must be a new or empty directory. The complete artifact directory is
    staged and published only after the numerical experiment succeeds.
    """
    import cv2 as cv
    import scipy

    from robotkinecal import SerialRobotKineCal

    samples = _integer(samples, "samples", 6)
    validation_samples = _integer(validation_samples, "validation_samples", 5)
    seed = _integer(seed, "seed", 0)
    max_candidates = _integer(
        max_candidates, "max_candidates", max(samples, validation_samples)
    )
    max_iterations = _integer(max_iterations, "max_iterations", 1)
    board = board or get_board_config(BoardType.OPENCV_ACIRCLE)
    if board.board_type == BoardType.CCTAG_SINGLE:
        raise ValueError(
            "a single marker cannot provide the full poses required by this experiment"
        )
    if measurement_mode not in {"image", "external-centers"}:
        raise ValueError("measurement_mode must be image or external-centers")
    if detector_backend not in ("opencv", "toolkit"):
        raise ValueError("detector_backend must be opencv or toolkit")
    if measurement_mode == "image":
        if (
            detector_backend == "opencv"
            and board.board_type != BoardType.OPENCV_ACIRCLE
        ):
            raise ValueError(
                "OpenCV full image detection requires opencv-acircle; CCTag needs explicit toolkit backend with v3, or external-centers mode"
            )
        if (
            detector_backend == "toolkit"
            and board.is_cctag_multi
            and board.legacy_artwork
        ):
            raise ValueError(
                "legacy CCTag image detection requires a separately verified detector; use the independent v3 target or external-centers"
            )
    camera, noise = camera or ImageCamera(), noise or ImageNoise()
    base_t_camera = _pose_values(
        default_camera_pose() if base_t_camera is None else base_t_camera,
        "Base_T_Camera",
    )
    ee_t_board = _pose_values(
        default_board_mount(board) if ee_t_board is None else ee_t_board, "EE_T_Board"
    )
    output = Path(output)
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise FileExistsError(
            "image experiment output must be a new or empty directory"
        )
    source = Path(urdf) if urdf is not None else example_urdf()
    source_bytes = source.read_bytes()
    renderer = PlanarBoardRenderer(
        board, camera, noise, registration_backend=detector_backend
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-image-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        (stage / "nominal.urdf").write_bytes(source_bytes)
        (stage / "board.png").write_bytes(renderer.texture_bytes)
        nominal = NumpyBackend.from_urdf(
            stage / "nominal.urdf", end_link=end_link, base_link=base_link
        )
        minimum = minimum_full_pose_samples(nominal.dof)
        if samples < minimum:
            raise ValueError(f"samples must be >= {minimum} for {nominal.dof} joints")
        truth = nominal.perturbed(seed=seed + 1)
        train, train_truth, train_collection = _collect(
            truth,
            renderer,
            base_t_camera,
            ee_t_board,
            samples,
            seed,
            "train",
            stage,
            measurement_mode,
            detector_backend,
            max_candidates,
            0,
        )
        validation, validation_truth, validation_collection = _collect(
            truth,
            renderer,
            base_t_camera,
            ee_t_board,
            validation_samples,
            seed + 2,
            "validation",
            stage,
            measurement_mode,
            detector_backend,
            max_candidates,
            max_candidates,
        )
        train.save(stage / "train-camera.npz")
        validation.save(stage / "validation-camera.npz")
        robot_train, robot_validation = (
            train.as_robot_body(base_t_camera, ee_t_board),
            validation.as_robot_body(base_t_camera, ee_t_board),
        )
        robot_train.save(stage / "train.npz")
        robot_validation.save(stage / "validation.npz")
        axes, home = nominal.poe_parameters()
        calibrator = SerialRobotKineCal.from_screw_axes(
            axes, home, joint_names=nominal.joint_names
        )
        fitted = fit_dataset(
            robot_train, calibrator, options=FitOptions(max_iterations=max_iterations)
        )
        fitted_predictions = predict_poe(
            fitted.screw_axes, fitted.zero_pose.A, robot_validation.joint_positions
        )
        nominal_predictions = np.asarray(
            [nominal.forward(q) for q in robot_validation.joint_positions]
        )
        last = fitted.result.iteration_results[-1]
        condition = float(last.condition_number)
        failures = []
        if not fitted.result.has_converged:
            failures.append("solver did not converge")
        if last.matrix_rank != last.N_PARAMS:
            failures.append("regressor is rank deficient")
        if not np.isfinite(condition) or condition > 1e8:
            failures.append("regressor exceeds condition-number threshold")
        report = {
            "schema_version": 1,
            "experiment_type": "image-kinematic-calibration",
            "measurement_model": "rendered-image-detection"
            if measurement_mode == "image"
            else "simulated-external-centers",
            "backend": "numpy-pinhole",
            "detector_backend": detector_backend,
            "board_type": board.board_type.value,
            "seed": seed,
            "truth_seed": seed + 1,
            "validation_seed": seed + 2,
            "source_urdf_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "board_png_sha256": hashlib.sha256(renderer.texture_bytes).hexdigest(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "opencv": cv.__version__,
            },
            "dof": nominal.dof,
            "max_candidates": max_candidates,
            "iterations": len(fitted.result.iteration_results),
            "max_condition_number": 1e8,
            "training_samples": samples,
            "validation_samples": validation_samples,
            "known_extrinsics": True,
            "Base_T_Camera": base_t_camera.tolist(),
            "EE_T_Board": ee_t_board.tolist(),
            "camera": asdict(camera),
            "image_noise": asdict(noise),
            "texture_registration": renderer.registration,
            "board_geometry": {**asdict(board), "board_type": board.board_type.value},
            "joint_names": list(nominal.joint_names),
            "joint_column_identity": robot_train.check_joint_identity(
                nominal.joint_names
            ),
            "base_link": nominal.base_link,
            "end_link": nominal.end_link,
            "regressor_rank": int(last.matrix_rank),
            "parameter_count": int(last.N_PARAMS),
            "condition_number": condition if np.isfinite(condition) else None,
            "converged": bool(fitted.result.has_converged),
            "termination_reason": fitted.result.termination_reason,
            "accepted": not failures,
            "quality_failures": failures,
            "acceptance_scope": "synthetic_image_solver_diagnostics",
            "train_collection": train_collection,
            "validation_collection": validation_collection,
            "nominal_vs_measurements": pose_metrics(
                nominal_predictions, robot_validation.observed_poses
            ),
            "calibrated_vs_measurements": pose_metrics(
                fitted_predictions, robot_validation.observed_poses
            ),
            "nominal_vs_truth": pose_metrics(nominal_predictions, validation_truth),
            "calibrated_vs_truth": pose_metrics(fitted_predictions, validation_truth),
            "measurement_vs_truth": pose_metrics(
                robot_validation.observed_poses, validation_truth
            ),
            "limitations": [
                "pinhole/no lens distortion",
                "no occlusion, lighting variation, collision checking or dynamics",
                "external transforms known exactly",
                "validation samples share the same synthetic physical robot",
            ],
        }
        atomic_write_json(stage / "metrics.json", report)
        truth_axes, truth_home = truth.poe_parameters()
        atomic_savez_compressed(
            stage / "model.npz",
            schema_version=np.asarray(1),
            model_frame=np.asarray("base_T_ee"),
            joint_names=np.asarray(nominal.joint_names),
            base_link=np.asarray(nominal.base_link),
            end_link=np.asarray(nominal.end_link),
            source_urdf_sha256=np.asarray(hashlib.sha256(source_bytes).hexdigest()),
            regressor_singular_values=np.asarray(last.singular_values),
            screw_axes=fitted.screw_axes,
            home=fitted.zero_pose.A,
            nominal_screw_axes=axes,
            nominal_home=home,
            truth_screw_axes=truth_axes,
            truth_home=truth_home,
            train_truth_poses=train_truth,
            validation_truth_poses=validation_truth,
            base_T_camera=base_t_camera,
            ee_T_board=ee_t_board,
        )
        files = {
            str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        atomic_write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "image-kinematic-experiment",
                "files_sha256": files,
            },
        )
        load_image_experiment(stage)
        if output.exists():
            output.rmdir()
        os.replace(stage, output)
    return report


__all__ = [
    "ImageCamera",
    "ImageNoise",
    "PlanarBoardRenderer",
    "default_camera_pose",
    "default_board_mount",
    "run_image_experiment",
    "load_image_experiment",
]


def _read_json(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _relative_artifact(root, name):
    """Resolve a canonical relative artifact name inside a portable bundle."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or Path(name).is_absolute()
        or Path(name).as_posix() != name
        or any(part in ("..", ".") for part in name.split("/"))
    ):
        raise ValueError(f"artifact path must be a canonical relative path: {name!r}")
    candidate = root / name
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes experiment root: {name}")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"missing regular experiment artifact: {name}")
    return candidate


def _model_scalar(model, key):
    if model[key].shape != ():
        raise ValueError(f"model {key} must be scalar")
    return model[key].item()


def _check_array(value, expected, name, atol=1e-9):
    raw = np.asarray(value)
    expected = np.asarray(expected)
    if (
        raw.dtype.kind not in "iuf"
        or raw.shape != expected.shape
        or not np.all(np.isfinite(raw))
        or not np.allclose(raw, expected, atol=atol, rtol=1e-9)
    ):
        raise ValueError(f"{name} does not match the archived experiment")


def _capture_image(root, role, filename, hashes):
    path = _relative_artifact(root / "captures" / role, filename)
    key = path.relative_to(root).as_posix()
    if key not in hashes:
        raise ValueError(f"capture image is missing from hash manifest: {key}")
    return path, key


def _verify_capture_records(root, role, dataset, report, board, camera, hashes):
    """Bind row order, image evidence and redetected poses to an observation NPZ."""
    import cv2 as cv

    directory = root / "captures" / role
    capture = _read_json(directory / "manifest.json")
    collection = _read_json(directory / "collection.json")
    mode = report["measurement_model"]
    for key, expected in (
        ("schema_version", 1),
        ("observation_frame", "camera_T_target"),
        ("camera_convention", "opencv-optical"),
        ("joint_unit", "rad"),
        ("translation_unit", "m"),
        ("board_type", board.board_type.value),
        ("measurement_model", mode),
        ("joint_names", report["joint_names"]),
    ):
        if capture.get(key) != expected or (
            key == "schema_version" and type(capture.get(key)) is not int
        ):
            raise ValueError(f"{role} capture {key} does not match experiment")
    _check_array(
        capture.get("camera_matrix"), camera.intrinsic, f"{role} camera intrinsics"
    )
    _check_array(capture.get("distortion"), np.zeros(5), f"{role} distortion")
    rows = capture.get("samples")
    if not isinstance(rows, list) or len(rows) != len(dataset.joint_positions):
        raise ValueError(f"{role} capture sample count does not match NPZ")
    attempts = collection.get("rendered_attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"{role} rendered_attempts must be a list")
    accepted = []
    for attempt in attempts:
        if not isinstance(attempt, dict) or type(attempt.get("accepted")) is not bool:
            raise ValueError(f"{role} invalid rendered-attempt record")
        image, key = _capture_image(root, role, attempt.get("image"), hashes)
        if attempt.get("image_sha256") != hashes[key]:
            raise ValueError(f"{role} collection image SHA-256 mismatch")
        if attempt["accepted"]:
            accepted.append(attempt)
    if len(accepted) != len(rows):
        raise ValueError(
            f"{role} accepted collection records do not match capture manifest"
        )
    for row, attempt, q, pose, sample_id in zip(
        rows,
        accepted,
        dataset.joint_positions,
        dataset.observed_poses,
        dataset.sample_indices,
        strict=True,
    ):
        if (
            not isinstance(row, dict)
            or type(row.get("sample_id")) is not int
            or row["sample_id"] != sample_id
            or attempt.get("sample_id") != sample_id
        ):
            raise ValueError(f"{role} capture sample IDs/order do not match NPZ")
        _check_array(row.get("joint_positions"), q, f"{role} capture joint positions")
        _check_array(
            attempt.get("joint_positions"), q, f"{role} collection joint positions"
        )
        for time_key in ("joint_timestamp_s", "image_timestamp_s"):
            value = row.get(time_key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value != attempt.get(time_key)
            ):
                raise ValueError(f"{role} capture timestamps are invalid")
        if row["joint_timestamp_s"] != row["image_timestamp_s"]:
            raise ValueError(f"{role} capture timestamps are not synchronized")
        _check_array(
            attempt.get("camera_T_board"), pose, f"{role} collection board pose"
        )
        if mode == "rendered-image-detection":
            if "image_points" in row:
                raise ValueError(
                    "image-mode capture must provide an image, not simulated centers"
                )
            image, key = _capture_image(root, role, row.get("image"), hashes)
            if row["image"] != attempt["image"]:
                raise ValueError(f"{role} image row does not match collection")
            decoded = cv.imdecode(
                np.frombuffer(image.read_bytes(), np.uint8), cv.IMREAD_COLOR
            )
            if decoded is None or decoded.shape[:2] != (camera.height, camera.width):
                raise ValueError(
                    f"{role} image is invalid or has the wrong camera dimensions"
                )
            rgb = cv.cvtColor(decoded, cv.COLOR_BGR2RGB)
            if report.get("detector_backend", "opencv") == "opencv":
                detection = detect_opencv_circles(
                    rgb,
                    camera.intrinsic,
                    np.zeros(5),
                    board=board,
                    max_reprojection_rms_px=0.5,
                )
            else:
                detection = detect_toolkit_board(
                    rgb,
                    camera.intrinsic,
                    np.zeros(5),
                    board=board,
                    max_board_tilt_rad=1.2,
                )
        else:
            if "image" in row or "image_points" not in row:
                raise ValueError(
                    "external-centers capture must provide explicit simulated centers"
                )
            _check_array(
                row["image_points"],
                attempt.get("image_points"),
                f"{role} simulated centers",
            )
            detection = pose_from_correspondences(
                board,
                row["image_points"],
                camera.intrinsic,
                np.zeros(5),
                max_reprojection_rms_px=0.5,
            )
        if detection is None:
            raise ValueError(f"{role} saved image no longer yields a board detection")
        _check_array(
            detection.camera_T_board, pose, f"{role} image/PnP measurement", atol=1e-7
        )
        rms = attempt.get("reprojection_rms_px")
        if (
            not isinstance(rms, (int, float))
            or isinstance(rms, bool)
            or not np.isfinite(rms)
            or not np.isclose(rms, detection.reprojection_rms_px, atol=1e-7, rtol=1e-7)
        ):
            raise ValueError(f"{role} image reprojection RMS does not match collection")
        if detection.pose_diagnostics and detection.pose_diagnostics.get("ambiguous"):
            raise ValueError(f"{role} saved observation has an ambiguous planar pose")
    summary = report.get(f"{role}_collection")
    candidates = _integer(
        collection.get("candidates"), f"{role} candidate count", len(rows)
    )
    counts = collection.get("rejection_counts")
    if (
        not isinstance(counts, dict)
        or any(
            not isinstance(k, str) or type(v) is not int or v < 0
            for k, v in counts.items()
        )
        or sum(counts.values()) != candidates - len(rows)
    ):
        raise ValueError(f"{role} collection rejection counts are inconsistent")
    if (
        not isinstance(summary, dict)
        or summary.get("candidates") != candidates
        or summary.get("accepted") != len(rows)
        or summary.get("rendered_images") != len(attempts)
        or summary.get("rejection_counts") != counts
    ):
        raise ValueError(f"{role} collection summary does not match report")
    mean_rms = float(np.mean([row["reprojection_rms_px"] for row in accepted]))
    if not np.isclose(
        summary.get("mean_reprojection_rms_px", np.nan), mean_rms, rtol=1e-9, atol=1e-12
    ):
        raise ValueError(
            f"{role} collection mean reprojection RMS does not match report"
        )


def load_image_experiment(path: Path | str) -> tuple[dict, KinematicDataset, dict]:
    """Verify and load an image experiment, redetecting accepted image evidence.

    File hashes detect modification and numerical checks detect inconsistent
    records. The manifest is not a signature and does not authenticate an author.
    Returned predictions are recomputed from saved POE models for visualization.
    """
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("image experiment must be a regular directory")
    manifest = _read_json(_relative_artifact(path, "manifest.json"))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("artifact_type") != "image-kinematic-experiment"
    ):
        raise ValueError("unsupported image experiment manifest")
    hashes = manifest.get("files_sha256")
    required = {
        "nominal.urdf",
        "board.png",
        "train.npz",
        "validation.npz",
        "train-camera.npz",
        "validation-camera.npz",
        "model.npz",
        "metrics.json",
        "captures/train/manifest.json",
        "captures/train/collection.json",
        "captures/validation/manifest.json",
        "captures/validation/collection.json",
    }
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("hash manifest is missing required image experiment artifacts")
    for name, digest in hashes.items():
        artifact = _relative_artifact(path, name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise ValueError(f"image experiment SHA-256 mismatch: {name}")
    actual_files = set()
    for artifact in path.rglob("*"):
        if artifact.is_symlink():
            raise ValueError("image experiment must not contain symlinks")
        if artifact.is_file() and artifact != path / "manifest.json":
            actual_files.add(artifact.relative_to(path).as_posix())
    if actual_files != set(hashes):
        raise ValueError("hash manifest must identify every image experiment file")
    report = _read_json(path / "metrics.json")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("experiment_type") != "image-kinematic-calibration"
        or report.get("backend") != "numpy-pinhole"
        or report.get("known_extrinsics") is not True
        or report.get("measurement_model")
        not in ("rendered-image-detection", "simulated-external-centers")
    ):
        raise ValueError("unsupported image experiment report")
    dof = _integer(report.get("dof"), "dof", 1)
    train_count = _integer(report.get("training_samples"), "training_samples", 6)
    validation_count = _integer(
        report.get("validation_samples"), "validation_samples", 5
    )
    seed = _integer(report.get("seed"), "seed", 0)
    if (
        report.get("truth_seed") != seed + 1
        or type(report.get("truth_seed")) is not int
        or report.get("validation_seed") != seed + 2
        or type(report.get("validation_seed")) is not int
    ):
        raise ValueError("image experiment seed identities do not match")
    _integer(report.get("iterations"), "iterations", 1)
    max_candidates = _integer(
        report.get("max_candidates"),
        "max_candidates",
        max(train_count, validation_count),
    )
    names = validated_joint_names(report.get("joint_names"), dof)
    try:
        board = BoardConfig(**report["board_geometry"])
        camera = ImageCamera(**report["camera"])
        ImageNoise(**report["image_noise"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid image camera, noise or board configuration") from exc
    if (
        board.board_type.value != report.get("board_type")
        or board.board_type == BoardType.CCTAG_SINGLE
    ):
        raise ValueError("image board identity does not match full-pose experiment")
    detector_backend = report.get("detector_backend", "opencv")
    if detector_backend not in ("opencv", "toolkit"):
        raise ValueError("unsupported image detector backend")
    if report["measurement_model"] == "rendered-image-detection":
        if (
            detector_backend == "opencv"
            and board.board_type != BoardType.OPENCV_ACIRCLE
        ):
            raise ValueError(
                "OpenCV image detection requires the asymmetric-circle board"
            )
        if (
            detector_backend == "toolkit"
            and board.is_cctag_multi
            and board.legacy_artwork
        ):
            raise ValueError(
                "legacy CCTag archive does not establish detector compatibility"
            )
    # Bind canvas dimensions/origin to actual archived artwork. PnP control
    # points alone cannot observe the board's outer dimensions or margins.
    texture = PlanarBoardRenderer(
        board,
        camera,
        ImageNoise(**report["image_noise"]),
        texture_path=path / "board.png",
        registration_backend=detector_backend,
    )
    if "detector_backend" in report:
        declared = report.get("texture_registration")
        if (
            not isinstance(declared, dict)
            or declared.get("method") != texture.registration["method"]
            or type(declared.get("rms_source_px")) not in (int, float)
            or not np.isclose(
                declared["rms_source_px"],
                texture.registration["rms_source_px"],
                atol=1e-7,
                rtol=1e-7,
            )
        ):
            raise ValueError(
                "archived texture registration does not match board geometry"
            )
    if (
        report.get("source_urdf_sha256") != hashes["nominal.urdf"]
        or report.get("board_png_sha256") != hashes["board.png"]
    ):
        raise ValueError("image experiment source model or texture identity mismatch")
    base_t_camera = _pose_values(report.get("Base_T_Camera"), "Base_T_Camera")
    ee_t_board = _pose_values(report.get("EE_T_Board"), "EE_T_Board")
    data = {}
    for role, count in (("train", train_count), ("validation", validation_count)):
        robot = KinematicDataset.load(path / f"{role}.npz")
        camera_data = KinematicDataset.load(path / f"{role}-camera.npz")
        for name, value, frame in (
            (role, robot, ObservationFrame.BASE_T_EE),
            (f"{role}-camera", camera_data, ObservationFrame.CAMERA_T_TARGET),
        ):
            if (
                value.joint_positions.shape != (count, dof)
                or value.observation_frame != frame
            ):
                raise ValueError(f"{name} dimensions or observation frame mismatch")
            if not value.check_joint_identity(names)["verified"]:
                raise ValueError(f"{name} requires verified joint names")
        _check_array(
            camera_data.joint_positions,
            robot.joint_positions,
            f"{role} camera/robot joint states",
        )
        if not np.array_equal(camera_data.sample_indices, robot.sample_indices):
            raise ValueError(f"{role} camera/robot sample identities mismatch")
        converted = camera_data.as_robot_body(base_t_camera, ee_t_board)
        _check_array(
            converted.observed_poses,
            robot.observed_poses,
            f"{role} camera-to-base pose conversion",
        )
        offset = 0 if role == "train" else max_candidates
        if np.any(robot.sample_indices < offset) or np.any(
            robot.sample_indices >= offset + max_candidates
        ):
            raise ValueError(
                f"{role} sample IDs are outside their declared acquisition namespace"
            )
        data[role] = robot
        _verify_capture_records(path, role, camera_data, report, board, camera, hashes)
    train, dataset = data["train"], data["validation"]
    train.validate_solver_sample_count()
    if np.intersect1d(train.sample_indices, dataset.sample_indices).size or set(
        map(tuple, train.joint_positions)
    ) & set(map(tuple, dataset.joint_positions)):
        raise ValueError("training and held-out sample identities overlap")
    identity = train.check_joint_identity(names)
    if report.get("joint_column_identity") != identity:
        raise ValueError("report joint-column identity does not match datasets")
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    fields = {
        "schema_version",
        "model_frame",
        "joint_names",
        "base_link",
        "end_link",
        "source_urdf_sha256",
        "regressor_singular_values",
        "screw_axes",
        "home",
        "nominal_screw_axes",
        "nominal_home",
        "truth_screw_axes",
        "truth_home",
        "train_truth_poses",
        "validation_truth_poses",
        "base_T_camera",
        "ee_T_board",
    }
    if not fields.issubset(model):
        raise ValueError(
            f"model archive is missing fields: {sorted(fields.difference(model))}"
        )
    if (
        type(_model_scalar(model, "schema_version")) is not int
        or _model_scalar(model, "schema_version") != 1
        or _model_scalar(model, "model_frame") != "base_T_ee"
    ):
        raise ValueError("unsupported image POE model schema/frame")
    if model["joint_names"].shape != (dof,) or model["joint_names"].tolist() != list(
        names
    ):
        raise ValueError("POE model joint identities mismatch")
    for name in ("base_link", "end_link"):
        if (
            not isinstance(report.get(name), str)
            or not report[name]
            or _model_scalar(model, name) != report[name]
        ):
            raise ValueError(f"POE {name} identity mismatch")
    if _model_scalar(model, "source_urdf_sha256") != hashes["nominal.urdf"]:
        raise ValueError("POE source URDF identity mismatch")
    _check_array(model["base_T_camera"], base_t_camera, "POE Base_T_Camera")
    _check_array(model["ee_T_board"], ee_t_board, "POE EE_T_Board")
    nominal = NumpyBackend.from_urdf(
        path / "nominal.urdf",
        base_link=report["base_link"],
        end_link=report["end_link"],
    )
    if nominal.joint_names != names:
        raise ValueError("archived URDF joint names mismatch")
    expected_axes, expected_home = nominal.poe_parameters()
    _check_array(model["nominal_screw_axes"], expected_axes, "nominal POE axes")
    _check_array(model["nominal_home"], expected_home, "nominal POE home")
    expected_axes, expected_home = nominal.perturbed(
        seed=report["truth_seed"]
    ).poe_parameters()
    _check_array(model["truth_screw_axes"], expected_axes, "truth POE axes/seed")
    _check_array(model["truth_home"], expected_home, "truth POE home/seed")
    for role in ("train", "validation"):
        expected = predict_poe(
            model["truth_screw_axes"], model["truth_home"], data[role].joint_positions
        )
        _check_array(
            model[f"{role}_truth_poses"], expected, f"{role} saved truth poses"
        )
        revolute = np.asarray(
            [
                joint.kind == "revolute"
                for joint in nominal.joints
                if joint.kind != "fixed"
            ]
        )
        joints = data[role].joint_positions[:, revolute]
        if np.any(joints < nominal.joint_limits[revolute, 0] - 1e-9) or np.any(
            joints > nominal.joint_limits[revolute, 1] + 1e-9
        ):
            raise ValueError("image experiment joints exceed URDF limits")
    for label, axes, home in (
        ("nominal", "nominal_screw_axes", "nominal_home"),
        ("calibrated", "screw_axes", "home"),
        ("truth", "truth_screw_axes", "truth_home"),
    ):
        checked, _ = validate_saved_poe_parameters(model[axes], model[home])
        if checked.shape != (dof, 6):
            raise ValueError(f"{label} POE axes do not match DOF")
        model[f"{label}_predictions"] = predict_poe(
            model[axes], model[home], dataset.joint_positions
        )
    for key, predicted, observed in (
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
        ("nominal_vs_truth", model["nominal_predictions"], model["truth_predictions"]),
        (
            "calibrated_vs_truth",
            model["calibrated_predictions"],
            model["truth_predictions"],
        ),
        ("measurement_vs_truth", dataset.observed_poses, model["truth_predictions"]),
    ):
        _check_metrics(report, key, pose_metrics(predicted, observed))
    parameters = _integer(report.get("parameter_count"), "parameter_count", 1)
    if parameters != 4 * dof + 6:
        raise ValueError("image model parameter count mismatch")
    singular = model["regressor_singular_values"]
    if singular.dtype.kind not in "iuf":
        raise ValueError("regressor singular values must be real numbers")
    rank, condition = _diagnostics(singular, train_count, parameters)
    condition_matches = (
        report.get("condition_number") is None
        if condition is None
        else type(report.get("condition_number")) in (int, float)
        and np.isclose(report["condition_number"], condition, rtol=1e-9)
    )
    if (
        type(report.get("regressor_rank")) is not int
        or report["regressor_rank"] != rank
        or not condition_matches
    ):
        raise ValueError("image regressor diagnostics mismatch")
    if type(report.get("converged")) is not bool or not isinstance(
        report.get("termination_reason"), str
    ):
        raise ValueError("invalid image solver convergence diagnostics")
    failures = []
    if not report["converged"]:
        failures.append("solver did not converge")
    if rank != parameters:
        failures.append("regressor is rank deficient")
    if condition is None or condition > 1e8:
        failures.append("regressor exceeds condition-number threshold")
    if (
        type(report.get("accepted")) is not bool
        or report["accepted"] != (not failures)
        or report.get("quality_failures") != failures
        or report.get("max_condition_number") != 1e8
        or report.get("acceptance_scope") != "synthetic_image_solver_diagnostics"
    ):
        raise ValueError("image experiment acceptance diagnostics are inconsistent")
    return report, dataset, model
