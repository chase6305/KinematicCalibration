"""Robot-model adapters for RobotKineCal."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CalibrationModel:
    """A numerical calibrator and the model identities needed by the workflow."""

    calibrator: object
    backend: str
    base_link: str
    end_link: str
    joint_names: tuple[str, ...]
    total_joints: int
    link_frames_available: bool


def load_calibration_model(
    path: Path | str,
    end_link: str,
    *,
    backend: str = "rtb",
    base_link: str | None = None,
    verbose: bool = False,
) -> CalibrationModel:
    """Load a URDF through RTB or a dependency-light space-POE adapter.

    The NumPy route preserves joint order and the chosen base/end identities.
    Its optimizer has no calibrated intermediate link frames, so its POE result
    cannot be exported as URDF joint origins. RTB retains the existing export
    behavior. Optional libraries are imported only for the selected route.
    """
    from robotkinecal import SerialRobotKineCal

    path = Path(path)
    if backend not in {"rtb", "numpy"}:
        raise ValueError("backend must be rtb or numpy")
    if backend == "rtb" and base_link is not None:
        raise ValueError("--base-link requires --backend numpy")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML: {path}") from exc
    if backend == "numpy":
        from kinematic_calibration.backends.numpy import NumpyBackend

        robot = NumpyBackend.from_urdf(path, end_link=end_link, base_link=base_link)
        axes, home = robot.poe_parameters()
        calibrator = SerialRobotKineCal.from_screw_axes(
            axes,
            home,
            joint_names=robot.joint_names,
            verbose=verbose,
        )
        return CalibrationModel(
            calibrator,
            backend,
            robot.base_link,
            robot.end_link,
            robot.joint_names,
            sum(
                joint.get("type") in {"revolute", "continuous", "prismatic"}
                and joint.find("mimic") is None
                for joint in root.findall("joint")
            ),
            False,
        )
    try:
        import roboticstoolbox as rtb
    except ImportError as exc:
        raise RuntimeError(
            "Robotics Toolbox is optional; install kinematic-calibration[robotics] "
            "or select --backend numpy for POE fitting without URDF export"
        ) from exc
    robot = rtb.ERobot.URDF(str(path.resolve()))
    calibrator = SerialRobotKineCal(robot, end_link, verbose=verbose)
    initialize_erobot_screw_axes(calibrator)
    by_child = {
        joint.find("child").get("link"): joint.get("name")
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    names = tuple(by_child.get(joint.name, joint.name) for joint in calibrator.joints)
    calibrator.joint_names = names
    return CalibrationModel(
        calibrator,
        backend,
        robot.base_link.name,
        calibrator.ee_name,
        names,
        int(robot.n),
        True,
    )


def initialize_erobot_screw_axes(calibrator) -> None:
    """Initialize POE axes from each ERobot joint's actual rotation axis.

    Retained for compatibility with older RobotKineCal builds. The ported core
    already handles the actual joint ET axis and flipped rotations directly.
    """
    robot = calibrator.robot_model
    # DH and POE models are initialized directly by RobotKineCal. Their links
    # do not expose ERobot's variable elementary transform ``v``.
    if robot is None or not all(hasattr(joint, "v") for joint in calibrator.joints):
        return
    zero_joints = np.zeros(robot.n)
    unit_axes = {
        "Rx": np.array([1.0, 0.0, 0.0]),
        "Ry": np.array([0.0, 1.0, 0.0]),
        "Rz": np.array([0.0, 0.0, 1.0]),
    }
    screw_axes = []
    for joint in calibrator.joints:
        joint_et = joint.v
        axis = unit_axes.get(joint_et.axis)
        if axis is None:
            raise ValueError(
                f"unsupported revolute axis {joint_et.axis!r} for joint {joint.name!r}"
            )
        zero_pose = robot.fkine(zero_joints, end=joint.name)
        direction = zero_pose.R @ axis
        if joint_et.isflip:
            direction = -direction
        point = zero_pose.t
        screw_axes.append(np.block([np.cross(point, direction), direction]))
    calibrator.joint_screw_axis = screw_axes


def _selected_chain_fk(robot, ee_name: str, dof: int):
    """Bind FK once, mapping selected columns to the backend's joint indices.

    RTB keeps global joint indices even for a selected end-effector path.
    Models without link/index metadata retain the existing fkine-only protocol.
    """
    fkine = robot.fkine
    link_dict = getattr(robot, "link_dict", {})
    end_link = link_dict.get(ee_name)
    total_joints = getattr(robot, "n", 0)
    if end_link is None or total_joints <= 0:
        return lambda positions: fkine(positions, end=ee_name)

    selected = []
    visited = set()
    link = end_link
    while link is not None:
        if id(link) in visited:
            raise ValueError("the selected FK chain contains a cycle")
        visited.add(id(link))
        if link.isjoint:
            selected.append(link)
        link = link.parent
    selected.reverse()
    indices = [getattr(joint, "jindex", None) for joint in selected]
    if not indices or any(index is None for index in indices):
        return lambda positions: fkine(positions, end=ee_name)
    if len(indices) != dof:
        raise ValueError(
            f"selected chain has {len(indices)} joints but joint_positions has {dof} columns"
        )
    if (
        any(not isinstance(index, (int, np.integer)) for index in indices)
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= total_joints
    ):
        raise ValueError("the selected FK chain has invalid global joint indices")
    indices = np.asarray(indices, dtype=int)

    def forward(positions):
        full_positions = np.zeros(total_joints)
        full_positions[indices] = positions
        return fkine(full_positions, end=ee_name)

    return forward


def validate_nominal_poe(calibrator, joint_positions, atol: float = 1e-8) -> None:
    """Check that RobotKineCal's nominal POE agrees with the URDF FK."""
    from spatialmath import SE3, Twist3

    q = np.asarray(joint_positions, dtype=float)
    poe_pose = SE3()
    for axis, position in zip(calibrator.joint_screw_axis, q, strict=True):
        poe_pose *= Twist3(axis).exp(position)
    poe_pose *= calibrator.zero_conf_EE_pose
    forward = _selected_chain_fk(
        calibrator.robot_model, calibrator.ee_name, calibrator.N_JOINTS
    )
    fk_pose = forward(q)
    if not np.allclose(poe_pose.A, fk_pose.A, atol=atol):
        position_error = np.linalg.norm(poe_pose.t - fk_pose.t)
        rotation_error = np.linalg.norm((fk_pose.inv() @ poe_pose).log(twist=True)[3:])
        raise ValueError(
            "nominal POE does not match URDF FK: "
            f"position_error_m={position_error:.6g}, "
            f"rotation_error_rad={rotation_error:.6g}"
        )


def estimate_composite_extrinsics(calibrator, joint_positions, observations):
    """Estimate fixed Camera_T_Base and EE_T_Target without external priors."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from spatialmath import SE3, Twist3

    from robotkinecal._batch import pose_log_batch

    from .dataset import _validate_transforms

    joints = np.asarray(joint_positions, dtype=float)
    observed = _validate_transforms("observations", observations)
    if len(joints) != len(observed):
        raise ValueError("joint and observation sample counts must match")

    def forward(q):
        pose = SE3()
        for axis, position in zip(calibrator.joint_screw_axis, q, strict=True):
            pose *= Twist3(axis).exp(position)
        return (pose @ calibrator.zero_conf_EE_pose).A

    nominal = np.asarray([forward(q) for q in joints])

    def unpack(parameters):
        camera_t_base = np.eye(4)
        camera_t_base[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        camera_t_base[:3, 3] = parameters[3:6]
        ee_t_target = np.eye(4)
        ee_t_target[:3, :3] = Rotation.from_rotvec(parameters[6:9]).as_matrix()
        ee_t_target[:3, 3] = parameters[9:12]
        return camera_t_base, ee_t_target

    initial_camera_t_base = observed[0] @ np.linalg.inv(nominal[0])
    initial = np.zeros(12)
    initial[:3] = Rotation.from_matrix(initial_camera_t_base[:3, :3]).as_rotvec()
    initial[3:6] = initial_camera_t_base[:3, 3]

    def residual(parameters):
        camera_t_base, ee_t_target = unpack(parameters)
        predicted = camera_t_base @ nominal @ ee_t_target
        relative = observed @ np.linalg.inv(predicted)
        return pose_log_batch(relative[:, :3, :3], relative[:, :3, 3]).reshape(-1)

    result = least_squares(residual, initial, method="trf")
    if not result.success:
        raise ValueError(f"composite extrinsic initialization failed: {result.message}")
    camera_t_base, ee_t_target = unpack(result.x)
    return (
        camera_t_base,
        ee_t_target,
        {
            "success": bool(result.success),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "evaluations": int(result.nfev),
        },
    )


def initialize_composite_model(calibrator, camera_t_base, ee_t_target) -> None:
    """Express a physical Base/EE POE model in Camera/Target coordinates."""
    from spatialmath import SE3

    camera_t_base = SE3(camera_t_base)
    ee_t_target = SE3(ee_t_target)
    calibrator.joint_screw_axis = [
        camera_t_base.Ad() @ axis for axis in calibrator.joint_screw_axis
    ]
    calibrator.zero_conf_EE_pose = (
        camera_t_base @ calibrator.zero_conf_EE_pose @ ee_t_target
    )
    if calibrator.joint_zero_conf_poses is not None:
        calibrator.joint_zero_conf_poses = [
            camera_t_base @ pose for pose in calibrator.joint_zero_conf_poses
        ]


def get_identifiable_urdf_joint_definitions(
    calibrator, camera_t_base, joint_names_by_child=None
):
    """Return local URDF-style definitions after the unidentifiable first joint."""
    from spatialmath import SE3

    if calibrator.joint_zero_conf_poses is None:
        raise ValueError("joint zero-configuration poses are unavailable")
    if any(
        joint.parent is not previous
        for previous, joint in zip(calibrator.joints, calibrator.joints[1:])
    ):
        raise ValueError(
            "partial URDF definitions require directly connected active joints; "
            "fixed paths between active joints are unsupported"
        )
    base_t_camera = SE3(camera_t_base).inv()
    calibrated_poses = [
        base_t_camera @ pose for pose in calibrator.joint_zero_conf_poses
    ]
    zero_joints = np.zeros(calibrator.robot_model.n)
    nominal_poses = [
        calibrator.robot_model.fkine(zero_joints, end=joint.name)
        for joint in calibrator.joints
    ]
    axis_vectors = {
        "Rx": np.array([1.0, 0.0, 0.0]),
        "Ry": np.array([0.0, 1.0, 0.0]),
        "Rz": np.array([0.0, 0.0, 1.0]),
    }
    definitions = []
    joint_names_by_child = joint_names_by_child or {}
    for index in range(1, calibrator.N_JOINTS):
        joint = calibrator.joints[index]
        nominal_local = nominal_poses[index - 1].inv() @ nominal_poses[index]
        calibrated_local = calibrated_poses[index - 1].inv() @ calibrated_poses[index]
        correction = nominal_local.inv() @ calibrated_local
        axis = axis_vectors[joint.v.axis].copy()
        if joint.v.isflip:
            axis = -axis
        definitions.append(
            {
                "zero_based_index": index,
                "joint_name": joint_names_by_child.get(
                    joint.name, f"{joint.parent.name}-{joint.name}"
                ),
                "parent_link": joint.parent.name,
                "child_link": joint.name,
                "xyz_m": calibrated_local.t.tolist(),
                "rpy_rad": calibrated_local.rpy(unit="rad", order="zyx").tolist(),
                "transform": calibrated_local.A.tolist(),
                "axis": axis.tolist(),
                "delta_xyz_m": correction.t.tolist(),
                "delta_rpy_rad": correction.rpy(unit="rad", order="zyx").tolist(),
                "delta_transform": correction.A.tolist(),
                "rpy_note": (
                    "RPY uses fixed-axis XYZ / order='zyx'; near pitch +/-pi/2 "
                    "use transform and delta_transform to avoid Euler ambiguity"
                ),
                "zero_offset_rad": None,
                "zero_offset_note": (
                    "not estimated separately by the POE axis-geometry model"
                ),
            }
        )
    return definitions


def estimate_joint_zero_offsets(
    robot,
    ee_name,
    joint_positions,
    observations,
    camera_t_base=None,
    ee_t_target=None,
    max_offset_rad=0.5,
):
    """Estimate encoder zero offsets while keeping URDF geometry fixed.

    If fixed transforms are omitted, they are estimated as nuisance variables
    and the first and last offsets are fixed to zero to remove their exact
    gauge ambiguity with Camera_T_Base and EE_T_Target.

    The fit minimizes spatial SE(3) twist residuals in [v, w] order. Reported
    position errors are Euclidean endpoint distances in the observation frame;
    they are distinct from the translation/rotation-coupled linear twist part.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robotkinecal._batch import pose_log_batch

    from .dataset import _validate_transforms

    raw_joints = np.asarray(joint_positions)
    if (
        raw_joints.dtype.kind not in "iuf"
        or raw_joints.ndim != 2
        or min(raw_joints.shape) == 0
        or not np.isfinite(raw_joints).all()
    ):
        raise ValueError("joint_positions must be a finite real non-empty [N, DOF] matrix")
    joints = raw_joints.astype(float, copy=False)
    observed = _validate_transforms("observations", np.asarray(observations))
    if len(joints) != len(observed):
        raise ValueError("joint and observation sample counts must match")
    dof = joints.shape[1]
    self_calibrating = camera_t_base is None and ee_t_target is None
    if (camera_t_base is None) != (ee_t_target is None):
        raise ValueError(
            "camera_t_base and ee_t_target must either both be provided or omitted"
        )
    raw_bound = np.asarray(max_offset_rad)
    if (
        raw_bound.ndim != 0
        or raw_bound.dtype.kind not in "iuf"
        or not np.isfinite(raw_bound)
        or raw_bound <= 0
    ):
        raise ValueError("max_offset_rad must be a positive finite value")
    max_offset_rad = float(raw_bound)
    forward = _selected_chain_fk(robot, ee_name, dof)

    if self_calibrating:
        from robotkinecal import SerialRobotKineCal

        calibrator = SerialRobotKineCal(robot, ee_name, verbose=False)
        initialize_erobot_screw_axes(calibrator)
        camera_t_base, ee_t_target, initialization = estimate_composite_extrinsics(
            calibrator, joints, observed
        )
        estimated_indices = np.arange(1, max(dof - 1, 1), dtype=int)
    else:
        camera_t_base = _validate_transforms(
            "Camera_T_Base", np.asarray(camera_t_base)[None]
        )[0]
        ee_t_target = _validate_transforms(
            "EE_T_Target", np.asarray(ee_t_target)[None]
        )[0]
        initialization = None
        estimated_indices = np.arange(dof, dtype=int)

    def pose_parameters(transform):
        return np.block(
            [
                Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
                transform[:3, 3],
            ]
        )

    def pose_from_parameters(parameters):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        transform[:3, 3] = parameters[3:6]
        return transform

    if self_calibrating:
        initial = np.block(
            [
                pose_parameters(camera_t_base),
                pose_parameters(ee_t_target),
                np.zeros(len(estimated_indices)),
            ]
        )
        lower = np.block(
            [np.full(12, -np.inf), np.full(len(estimated_indices), -max_offset_rad)]
        )
        upper = np.block(
            [np.full(12, np.inf), np.full(len(estimated_indices), max_offset_rad)]
        )
    else:
        initial = np.zeros(len(estimated_indices))
        lower = np.full(len(estimated_indices), -max_offset_rad)
        upper = np.full(len(estimated_indices), max_offset_rad)

    def unpack(parameters):
        offsets = np.zeros(dof)
        if self_calibrating:
            current_camera_t_base = pose_from_parameters(parameters[:6])
            current_ee_t_target = pose_from_parameters(parameters[6:12])
            offsets[estimated_indices] = parameters[12:]
        else:
            current_camera_t_base = camera_t_base
            current_ee_t_target = ee_t_target
            offsets[estimated_indices] = parameters
        return current_camera_t_base, current_ee_t_target, offsets

    def predict_poses(parameters):
        current_camera_t_base, current_ee_t_target, offsets = unpack(parameters)
        base_t_ee = np.asarray(
            [
                forward(configuration + offsets).A
                for configuration in joints
            ]
        )
        return current_camera_t_base @ base_t_ee @ current_ee_t_target

    def residual(parameters):
        predicted = predict_poses(parameters)
        relative_rotations = observed[:, :3, :3] @ predicted[:, :3, :3].transpose(
            0, 2, 1
        )
        relative_translations = observed[:, :3, 3] - np.einsum(
            "nij,nj->ni", relative_rotations, predicted[:, :3, 3]
        )
        return pose_log_batch(relative_rotations, relative_translations).reshape(-1)

    result = least_squares(residual, initial, bounds=(lower, upper), method="trf")
    if not result.success:
        raise ValueError(f"joint zero-offset solve failed: {result.message}")
    final_camera_t_base, final_ee_t_target, offsets = unpack(result.x)
    predicted = predict_poses(result.x)
    position_errors = np.linalg.norm(observed[:, :3, 3] - predicted[:, :3, 3], axis=1)
    relative_rotations = observed[:, :3, :3] @ predicted[:, :3, :3].transpose(0, 2, 1)
    orientation_errors = Rotation.from_matrix(relative_rotations).magnitude()

    _, singular_values, right_singular_vectors = np.linalg.svd(
        result.jac, full_matrices=False
    )
    rank_tolerance = (
        singular_values[0]
        * max(result.jac.shape)
        * np.finfo(singular_values.dtype).eps
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    residual_degrees_of_freedom = int(result.jac.shape[0] - rank)
    if residual_degrees_of_freedom > 0 and rank == result.jac.shape[1]:
        residual_variance = float(
            np.dot(result.fun, result.fun) / residual_degrees_of_freedom
        )
        # Avoid squaring the Jacobian's condition number through J.T @ J.
        parameter_variances = residual_variance * np.sum(
            (right_singular_vectors.T / singular_values) ** 2, axis=1
        )
        parameter_std = np.sqrt(np.maximum(parameter_variances, 0.0))
    else:
        residual_variance = float("nan")
        parameter_std = np.full(result.jac.shape[1], np.nan)
    offset_std_rad = np.full(dof, np.nan)
    offset_parameter_start = 12 if self_calibrating else 0
    offset_std_rad[estimated_indices] = parameter_std[offset_parameter_start:]
    return {
        "offsets": offsets,
        "estimated_indices": estimated_indices,
        "camera_t_base": final_camera_t_base,
        "ee_t_target": final_ee_t_target,
        "self_calibrating": self_calibrating,
        "initialization": initialization,
        "success": bool(result.success),
        "message": result.message,
        "evaluations": int(result.nfev),
        "cost": float(result.cost),
        "position_error_mean_m": float(position_errors.mean()),
        "position_error_max_m": float(position_errors.max()),
        "position_error_rms_m": float(np.sqrt(np.mean(position_errors**2))),
        "position_error_p95_m": float(np.percentile(position_errors, 95)),
        "orientation_error_mean_rad": float(orientation_errors.mean()),
        "orientation_error_max_rad": float(orientation_errors.max()),
        "orientation_error_rms_rad": float(np.sqrt(np.mean(orientation_errors**2))),
        "orientation_error_p95_rad": float(np.percentile(orientation_errors, 95)),
        "position_errors_m": position_errors,
        "orientation_errors_rad": orientation_errors,
        "jacobian_rank": rank,
        "parameter_count": int(result.jac.shape[1]),
        "full_rank": rank == int(result.jac.shape[1]),
        "condition_number": condition_number,
        "singular_values": singular_values,
        "offset_std_rad": offset_std_rad,
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "residual_variance": residual_variance,
    }
