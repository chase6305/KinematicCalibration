"""POE calibration ported from RobotKineCal; see UPSTREAM.md and its MIT notice."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from spatialmath import SE3, Twist3

from ._batch import evaluate_poe_batch, pose_log_batch, skew_batch

if TYPE_CHECKING:
    import roboticstoolbox as rtb


def _real_array(value, name):
    """Reject complex, boolean, and textual data before float conversion."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    return np.asarray(raw, dtype=float)


def _rank_from_singular_values(singular_values, matrix_shape):
    """Match NumPy's default numerical rank without factoring the matrix again."""
    tolerance = singular_values[0] * max(matrix_shape) * np.finfo(singular_values.dtype).eps
    return int(np.count_nonzero(singular_values > tolerance))


def _regressor_statistics(matrix):
    """Derive rank, spectrum and 2-norm condition from a single SVD."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = _rank_from_singular_values(singular_values, matrix.shape)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    return rank, singular_values, condition


def _pose_log(pose):
    """SE(3) logarithm in [v, w] order, stable near zero and pi rotations.

    Computing the rotational logarithm through arccos(trace(R)) and dividing
    by sin(theta) loses precision near pi. SciPy's quaternion-based rotation
    vector avoids that cancellation. The inverse SO(3) left Jacobian uses a
    half-angle expression that is finite at pi and a series near zero.
    """
    angular = Rotation.from_matrix(pose.R).as_rotvec()
    theta = norm(angular)
    if theta < 1e-4:
        coefficient = 1.0 / 12.0 + theta**2 / 720.0 + theta**4 / 30240.0
    else:
        coefficient = (1.0 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
    linear = (
        pose.t
        - 0.5 * np.cross(angular, pose.t)
        + coefficient * np.cross(angular, np.cross(angular, pose.t))
    )
    return np.r_[linear, angular]


class CalibrationResult:
    """
    Class to store the result of a kinematic calibration.
    """

    def __init__(
        self,
        robot_model: rtb.Robot,
        convergence_tolerance: float = 1e-4,
        joints: list | None = None,
        divergence_patience: int = 3,
    ) -> None:
        """
        Initialize the calibration result.

        Parameters
        -----------
        robot_model: roboticstoolbox.Robot
            The robot model used for the calibration.
        convergence_tolerance: float
            The tolerance to consider that the error has converged.
        """
        self.robot_model = robot_model
        self.joints = (
            list(joints)
            if joints is not None
            else [link for link in self.robot_model.links if link.isjoint]
        )
        self.N_JOINTS = len(self.joints)
        self.convergence_tolerance = convergence_tolerance
        self.has_converged = False
        self.is_diverging = False
        self.termination_reason = None
        self.convergence_criterion = None
        self.divergence_patience = divergence_patience
        self._consecutive_error_increases = 0
        self.nb_iterations_executed = 0
        self.iteration_results = []

    class IterationResult:
        """
        Class to store the result of an iteration of the calibration.
        """

        def __init__(
            self,
            joint_screw_definitions: list[np.ndarray],
            zero_conf_EE_pose: SE3,
            A_all: np.ndarray,
            twist_corrections: np.ndarray,
            twist_errors: np.ndarray,
            position_errors: list[float],
            orientation_errors: list[float],
            step_size: float = 1.0,
            post_update_twist_errors_norm: float | None = None,
            pre_update_objective_norm: float | None = None,
            post_update_objective_norm: float | None = None,
            objective_scale: float = 1.0,
            row_weights: np.ndarray | None = None,
        ) -> None:
            """
            Initialize the iteration result.

            Parameters
            ----------
            joint_screw_definitions: list
                One base-frame 6D screw axis per joint, in [v, w] order.
            zero_conf_EE_pose: SE3
                Base_T_EE when all joint positions are zero.
            A_all: numpy.ndarray
                Stacked observation regressor, with shape
                (OBS_SIZE*N_OBSERVATIONS, 4*N_JOINTS+OBS_SIZE).
            twist_corrections: numpy.ndarray
                Proposed parameter corrections with shape (4*N_JOINTS+OBS_SIZE, 1).
                The applied corrections also include step_size.
            twist_errors: numpy.ndarray
                Pre-update stacked residuals, shape (OBS_SIZE*N_OBSERVATIONS, 1).
            position_errors: List[float]
                Pre-update Euclidean endpoint errors in metres, one per observation.
            orientation_errors: List[float]
                Pre-update rotation-angle errors in radians, one per observation.
            """
            # Store a snapshot: the optimizer continues mutating its model after
            # this result has been created.
            self.zero_conf_EE_pose = SE3(zero_conf_EE_pose.A.copy())
            self.joint_screw_definitions = [
                axis.copy() for axis in joint_screw_definitions
            ]
            self.A_all = A_all.copy()
            self.twist_corrections = twist_corrections.copy()
            self.step_size = step_size
            self.applied_twist_corrections = step_size * twist_corrections.copy()
            self.twist_errors = twist_errors.copy()
            self.pre_update_twist_errors = self.twist_errors.copy()
            self.pre_update_twist_errors_norm = norm(self.twist_errors)
            self.post_update_twist_errors_norm = post_update_twist_errors_norm
            self.pre_update_objective_norm = (
                self.pre_update_twist_errors_norm
                if pre_update_objective_norm is None
                else float(pre_update_objective_norm)
            )
            self.post_update_objective_norm = (
                post_update_twist_errors_norm
                if post_update_objective_norm is None
                else float(post_update_objective_norm)
            )
            self.position_errors = list(position_errors)
            self.objective_scale = float(objective_scale)
            self.proposed_parameter_update_norm = float(norm(twist_corrections))
            self.relative_objective_change = None
            self.orientation_errors = list(orientation_errors)

            # Number of joints
            self.N_JOINTS = len(joint_screw_definitions)

            # Number of observations
            self.N_OBSERVATIONS = len(position_errors)

            # Length of a single observation
            # Will be 3 for position only observations and 6 for pose observations
            self.OBS_SIZE = int(len(self.twist_errors) / self.N_OBSERVATIONS)

            # Number of kinematic parameters in the model
            self.N_PARAMS = 4 * self.N_JOINTS + self.OBS_SIZE

            # Whether the observations are 3D/positions or 6D/poses
            self.position_only = self.OBS_SIZE == 3

            # Observability diagnostics for the linearized calibration problem.
            (
                self.matrix_rank,
                self.singular_values,
                self.condition_number,
            ) = _regressor_statistics(self.A_all)
            self.row_weights = (
                np.ones(self.A_all.shape[0])
                if row_weights is None
                else _real_array(row_weights, "row_weights").copy()
            )
            if self.row_weights.shape != (self.A_all.shape[0],) or not np.all(
                np.isfinite(self.row_weights) & (self.row_weights > 0)
            ):
                raise ValueError(
                    "row_weights must be a positive finite vector matching regressor rows."
                )
            self.weighted_A_all = self.row_weights[:, None] * self.A_all
            self.weighted_twist_errors = self.row_weights[:, None] * self.twist_errors
            if np.all(self.row_weights == 1.0):
                self.weighted_matrix_rank = self.matrix_rank
                self.weighted_singular_values = self.singular_values.copy()
                self.weighted_condition_number = self.condition_number
            else:
                (
                    self.weighted_matrix_rank,
                    self.weighted_singular_values,
                    self.weighted_condition_number,
                ) = _regressor_statistics(self.weighted_A_all)
            self.uncertainty_method = "local_linear_weighted_residual_heuristic"

        def compute_uncertainty_estimate(self, A_all, x_all, y_all):
            """
            Compute a local linear residual-based uncertainty heuristic.

            This estimate does not model damping, nonlinear model error or
            uncertainty in sample weights. Unidentifiable models return NaNs.

            Parameters
            -----------
            A_all: numpy.ndarray
                Design matrix, including residual row weights when applicable.
                Shape: (OBS_SIZE*N_OBSERVATIONS, 4*N_JOINTS+OBS_SIZE).
            x_all: numpy.ndarray
                Proposed parameter corrections, shape (4*N_JOINTS+OBS_SIZE, 1).
            y_all: numpy.ndarray
                Residual vector with the same row weights as A_all.
                Shape: (OBS_SIZE*N_OBSERVATIONS, 1).

            Returns
            --------
            A vector of length N_JOINTS containing the norm of the four
            parameter variances for each joint. Terminal-pose parameters
            contribute to the fit but are not included in the returned vector.
            """
            degrees_of_freedom = A_all.shape[0] - self.N_PARAMS
            if degrees_of_freedom <= 0:
                return np.full(self.N_JOINTS, np.nan)

            _, singular_values, right_singular_vectors = np.linalg.svd(
                A_all, full_matrices=False
            )
            if (
                _rank_from_singular_values(singular_values, A_all.shape)
                < self.N_PARAMS
            ):
                # A pseudoinverse of a singular information matrix gives zero
                # variance along its null space, falsely implying certainty.
                return np.full(self.N_JOINTS, np.nan)

            residual = y_all - A_all @ x_all
            residual_variance = float(np.sum(residual**2)) / degrees_of_freedom
            # Work from the design-matrix SVD instead of squaring its condition
            # number in A.T @ A. Vh rows are the right singular vectors.
            parameter_variances = residual_variance * np.sum(
                (right_singular_vectors.T / singular_values) ** 2, axis=1
            )
            joint_variances = parameter_variances[: -self.OBS_SIZE].reshape(-1, 4)
            return np.linalg.norm(joint_variances, axis=1)

        def get_statistics(self):
            """
            Compute statistics on the results of the iteration.

            Returns
            --------
            A dictionary containing the following statistics:
            - twist_errors_norm: The norm of the twist errors.
            - position_errors_mean: The mean of the position errors.
            - position_errors_max: The maximum of the position errors.
            - orientation_errors_mean: The mean of the orientation errors.
            - orientation_errors_max: The maximum of the orientation errors.
            - joints_uncertainty: The uncertainty estimate of the calibration.
            """
            stats = {}
            stats["twist_errors_norm"] = norm(self.twist_errors)
            stats["position_errors_mean"] = np.mean(self.position_errors)
            stats["position_errors_max"] = np.max(self.position_errors)
            stats["orientation_errors_mean"] = np.mean(self.orientation_errors)
            stats["orientation_errors_max"] = np.max(self.orientation_errors)
            stats["joints_uncertainty"] = self.compute_uncertainty_estimate(
                self.weighted_A_all, self.twist_corrections, self.weighted_twist_errors
            )
            stats["matrix_rank"] = self.matrix_rank
            stats["condition_number"] = self.condition_number
            stats["singular_values"] = self.singular_values.copy()
            stats["pre_update_objective_norm"] = self.pre_update_objective_norm
            stats["post_update_objective_norm"] = self.post_update_objective_norm
            stats["objective_scale"] = self.objective_scale
            stats["proposed_parameter_update_norm"] = (
                self.proposed_parameter_update_norm
            )
            stats["relative_objective_change"] = self.relative_objective_change
            stats["weighted_matrix_rank"] = self.weighted_matrix_rank
            stats["weighted_singular_values"] = self.weighted_singular_values.copy()
            stats["weighted_condition_number"] = self.weighted_condition_number
            stats["uncertainty_method"] = self.uncertainty_method
            return stats

        def print(self):
            stats = self.get_statistics()

            print(f"\tNorm of twist errors: {stats['twist_errors_norm']:.4f}")
            print(f"\tAvg. Position error: {stats['position_errors_mean']:.4f}")
            print(f"\tMax. Position error: {stats['position_errors_max']:.4f}")
            print(f"\tAvg. Orientation error: {stats['orientation_errors_mean']:.4f}")
            print(f"\tMax. Orientation error: {stats['orientation_errors_max']:.4f}")
            with np.printoptions(precision=4):
                print(f"\tJoints uncertainty: {stats['joints_uncertainty']}")
            print(f"\tRegressor rank: {stats['matrix_rank']}/{self.N_PARAMS}")
            print(f"\tRegressor condition number: {stats['condition_number']:.4e}")
            print(f"\tApplied step size: {self.step_size:.4e}")
            if self.post_update_twist_errors_norm is not None:
                print(
                    "\tPost-update twist error norm: "
                    f"{self.post_update_twist_errors_norm:.4e}"
                )

    def add_iteration_result(self, iteration_result: IterationResult):
        """
        Add the results of an iteration to the list of iteration results.

        Parameters
        -----------
        iteration_result: IterationResult
            The result of an iteration of the calibration.
        """
        self.iteration_results.append(iteration_result)
        self.nb_iterations_executed += 1

        # Check if the optimization process has converged or is diverging
        if self.nb_iterations_executed > 1:
            previous_result = self.iteration_results[-2]
            current_result = self.iteration_results[-1]

            current_objective = (
                current_result.pre_update_objective_norm
                / current_result.objective_scale
            )
            previous_objective = (
                previous_result.pre_update_objective_norm
                / previous_result.objective_scale
            )
            objective_maximum = max(current_objective, previous_objective)
            current_result.relative_objective_change = (
                0.0
                if objective_maximum == 0.0
                else abs(current_objective - previous_objective)
                / objective_maximum
                * ((current_objective + previous_objective) / objective_maximum)
            )
            if abs(current_objective - previous_objective) < self.convergence_tolerance:
                self.has_converged = True
                self.termination_reason = "converged"
                self.convergence_criterion = "objective_change"
            elif current_objective > previous_objective:
                self._consecutive_error_increases += 1
                if self._consecutive_error_increases >= self.divergence_patience:
                    self.is_diverging = True
                    self.termination_reason = "diverged"
            else:
                self._consecutive_error_increases = 0

    def get_screw_axes(self):
        """
        Compute screw definitions for use with the PoE formula.

        Returns
        --------
        A list of joint screw axes.
        """
        result_from_last_iteration = self.iteration_results[-1]
        return [
            axis.copy() for axis in result_from_last_iteration.joint_screw_definitions
        ]

    def get_zero_conf_EE_pose(self):
        """
        Returns
        --------
        A SE3 object defining the pose of the robot end-effector when all joint positions are zero.
        """
        result_from_last_iteration = self.iteration_results[-1]
        return SE3(result_from_last_iteration.zero_conf_EE_pose.A.copy())

    def get_urdf_xyzrpy(self, zero_conf_joint_poses: list[SE3]):
        """
        Compute the RPY-XYZ format of the joint definitions for use in a URDF file.

        In a URDF file, each rotary joint is defined by its axis of rotation and the position of the joint relative to the previous link.

        Parameters
        -----------
        zero_conf_joint_poses: list
            List of SE3 objects defining the pose of each joint and the end-effector when the robot is in the zero configuration.

        Returns
        --------
        A list of dictionaries where each dictionary contains the following keys:
        - name: The name of the joint (e.g. "link1-link2").
        - xyz: The position of the joint relative to the previous link.
        - rpy: The orientation of the joint relative to the previous link.
        """

        # A PoE chain does not describe joint position, it only describes
        # how the end-effector moves when the joints move. Hence, we cannot
        # use a PoERobot to generate URDF joint definitions.
        if self.robot_model is None:
            raise TypeError(
                "A screw-axis model does not define URDF link frames. "
                "Export through the URDF backend that supplied the screw axes."
            )
        import roboticstoolbox as rtb

        if isinstance(self.robot_model, rtb.PoERobot):
            raise TypeError(
                "A PoERobot does not describe joint positions, use a DHRobot or ERobot instead."
            )

        joint_definitions = []
        previous_link_pose = SE3()
        for i, joint in enumerate(self.joints):
            joint_name = f"{joint.parent.name}-{joint.name}"
            zero_conf_joint_pose = zero_conf_joint_poses[i]

            # Pose of the joint, relative to the parent frame,
            # when the robot is in the zero configuration
            X = previous_link_pose.inv() * zero_conf_joint_pose
            rpy = X.rpy(unit="rad", order="zyx")
            xyz = X.t

            joint_def = {}
            joint_def["name"] = joint_name
            joint_def["xyz"] = xyz
            joint_def["rpy"] = rpy
            joint_definitions.append(joint_def)

            previous_link_pose = zero_conf_joint_pose

        # End-effector
        joint_name = f"{self.joints[-1].name}-{self.joints[-1].children[0].name}"
        zero_conf_ee_pose = self.iteration_results[-1].zero_conf_EE_pose
        X = previous_link_pose.inv() * zero_conf_ee_pose
        rpy = X.rpy(unit="rad", order="zyx")
        xyz = X.t
        joint_def = {}
        joint_def["name"] = joint_name
        joint_def["xyz"] = xyz
        joint_def["rpy"] = rpy
        joint_definitions.append(joint_def)

        return joint_definitions


class SerialRobotKineCal:
    """
    Class implementing the method described in:
        Li, C., Wu, Y., Löwe, H., & Li, Z. (2016). POE-based robot kinematic calibration using axis configuration space and the adjoint error model.
        IEEE Transactions on Robotics, 32(5), 1264-1279.
    which consists in an iterative least squares method where at each iteration the screw axes of the joints are
    updated to minimize the error between the measured and computed end effector pose.

    The number of parameters to estimate is minimal (4 per joint), yielding greater calibration speed. In contrast to DH parameters,
    screw axes are much easier to optimize over, and the calibration is less likely to get stuck in local minima.

    The input data consists of joint positions and end-effector poses or points. When supplying end-effector points as observations,
    a greater number of observations is required to yield the same accuracy as when supplying end-effector poses. Furthermore, the
    orientation of the end-effector is not identifiable when using points, only its position is identifiable.
    """

    def __init__(
        self, robot_model: rtb.Robot, ee_name: str | None = None, verbose=False
    ):
        """
        Build lists of links and joints from the roboticstoolbox.Robot model and the selected end effector,
        assuming that the robot is a serial chain from the base to the end effector and that the pose of the
        end effector can be measured relative to the base.

        Parameters
        -----------
        robot_model: roboticstoolbox.Robot
            The roboticstoolbox model of the robot.
        ee_name: str
            The name of the end effector link. If None, the last link is assumed to be the end effector.
        verbose: bool
            If True, print additional information during the calibration process.
        """
        try:
            import roboticstoolbox as rtb
        except ImportError as exc:
            raise ImportError(
                "The robot-model constructor requires roboticstoolbox-python. "
                "Install the 'robotics' extra or use SerialRobotKineCal.from_screw_axes()."
            ) from exc
        if not isinstance(robot_model, rtb.Robot):
            raise TypeError("robot_model must be an instance of roboticstoolbox.Robot.")
        self.robot_model = robot_model

        # If True, print additional information during the calibration process
        self.verbose = verbose

        # If True, only the position information is used for calibration
        self.position_only = False

        # Each element of self.joints is a roboticstoolbox Link object
        self.joints = []

        # Each element of joint_screw_axis is the screw axis of the joint
        self.joint_screw_axis = []

        # Pose of the end-effector when all joint positions are zero
        self.zero_conf_EE_pose = None

        # The structure of PoERobot is different from the one DHRobot or ERobot
        if isinstance(robot_model, rtb.PoERobot):
            # RTB 1.3 inserts fixed base and tool sentinel links but reports
            # robot.n == 0 because PoELink ETS objects are world-frame constants.
            # Older releases exposed only the actual POE links. Support both
            # representations without relying on the broken ``n`` property.
            if robot_model.n == 0 and len(robot_model.links) >= 2:
                self.joints = list(robot_model.links[1:-1])
            else:
                self.joints = list(robot_model.links)
            self.joint_screw_axis = [link.S.A.copy() for link in self.joints]
            self.zero_conf_EE_pose = self.robot_model.T0
            # With a PoERobot, local poses are undefined since
            # the PoE formula does not describe joint poses.
            self.joint_local_nominal_poses = None
            self.joint_zero_conf_poses = None
        elif isinstance(robot_model, rtb.DHRobot):
            # DHRobot does not support fkine(q, end=link). Robotics Toolbox can
            # provide its equivalent POE representation directly, which is both
            # simpler and less error-prone than reconstructing screw axes from DH.
            screw_axes, self.zero_conf_EE_pose = robot_model.twists()
            self.joints = list(robot_model.links)
            self.joint_screw_axis = [axis.S.copy() for axis in screw_axes]
            self.ee_name = ee_name or self.joints[-1].name
            if ee_name is not None and ee_name != self.joints[-1].name:
                raise ValueError(
                    "Selecting an intermediate end effector is not supported for DHRobot."
                )
            self.joint_local_nominal_poses = [link.A(0) for link in self.joints]
            self.joint_zero_conf_poses = []
            zero_pose = SE3(robot_model.base)
            for link in self.joints:
                zero_pose *= link.A(0)
                self.joint_zero_conf_poses.append(SE3(zero_pose.A.copy()))
        else:
            # If the user has not specified the end effector link,
            # assume that the last link is the end effector.
            if ee_name is None:
                if not self.robot_model.links:
                    raise ValueError("The robot model does not contain any links.")
                ee_name = self.robot_model.links[-1].name
            self.ee_name = ee_name

            # Verify that the end effector link exists
            ee_link = self.robot_model.link_dict.get(ee_name)
            if ee_link is None:
                raise ValueError(f"End-effector link '{ee_name}' does not exist.")

            # Follow parent links from the selected end effector to the base.
            # This excludes gripper or other branch joints that do not affect the
            # selected end effector, and provides an unambiguous serial order.
            chain_links = []
            link = ee_link
            visited_links = set()
            while link is not None:
                if id(link) in visited_links:
                    raise ValueError(
                        "The robot model contains a cycle in its link graph."
                    )
                visited_links.add(id(link))
                chain_links.append(link)
                link = link.parent
            chain_links.reverse()

            if self.robot_model.base_link not in chain_links:
                raise ValueError(
                    f"End-effector link '{ee_name}' is not connected to the robot base."
                )

            # Each element of self.joints is a roboticstoolbox Link object
            self.joints = [link for link in chain_links if link.isjoint]
            if not self.joints:
                raise ValueError("The selected end-effector chain contains no joints.")

            # Each element of joint_local_nominal_poses is the pose of the i-th link relative to the previous link
            # when the joint angles are zero.
            self.joint_local_nominal_poses = [joint.A() for joint in self.joints]

            # For each joint, define the screw axis of a revolute joint rotating about Z
            # and record the pose of the joint when the robot is in the zero configuration.
            self.joint_zero_conf_poses = []
            unit_axes = {"Rx": np.eye(3)[0], "Ry": np.eye(3)[1], "Rz": np.eye(3)[2]}
            for chain_link in chain_links:
                zero_conf_joint_pose = robot_model.fkine(
                    np.zeros(robot_model.n), chain_link.name
                )
                if chain_link.isjoint:
                    joint_et = chain_link.v
                    local_axis = unit_axes.get(joint_et.axis)
                    if local_axis is None:
                        raise ValueError(
                            f"Unsupported joint axis {joint_et.axis!r} for {chain_link.name!r}."
                        )
                    direction = zero_conf_joint_pose.R @ local_axis
                    if joint_et.isflip:
                        direction = -direction
                    screw_axis = np.r_[
                        np.cross(zero_conf_joint_pose.t, direction), direction
                    ]
                    self.joint_screw_axis.append(screw_axis)
                    # Record the pose of the joint when the robot is in the zero configuration
                    self.joint_zero_conf_poses.append(zero_conf_joint_pose)
                if chain_link == ee_link:
                    self.zero_conf_EE_pose = zero_conf_joint_pose

            if self.zero_conf_EE_pose is None:
                raise ValueError(
                    f"Could not compute the zero pose of end-effector '{ee_name}'."
                )

        # Verify that the robot has only revolute joints
        # A future version could support prismatic joints.
        for j in self.joints:
            if j.isprismatic:
                raise ValueError("Only revolute joints are currently supported.")

        # Number of joints
        self.N_JOINTS = len(self.joint_screw_axis)

        # List of observed end-effector poses that will be populated by set_data
        self._observations = []

        # List of measured joint positions that will be populated by set_data
        self._joint_positions = []

        # Placeholder for the B matrix that has to be computed only once.
        self._B_Matrix = None

        # Preserve the nominal state so callers can run independent calibration
        # experiments with the same object.
        self._initial_model_state = self._model_snapshot()

    @classmethod
    def from_screw_axes(
        cls,
        screw_axes,
        zero_conf_ee_pose,
        *,
        joint_names=None,
        verbose=False,
    ):
        """Construct a calibrator without Robotics Toolbox.

        ``screw_axes`` has shape ``(N, 6)`` in linear/angular ``[v, w]``
        order, expressed in the base frame at zero configuration. Each angular
        axis must be unit length and perpendicular to its linear component.
        ``zero_conf_ee_pose`` is the ``base_T_ee`` home transform, either a
        finite 4x4 array or a single :class:`spatialmath.SE3`. Joint values use
        radians and translations use metres.

        The returned instance supports the same solver and POE result API.
        Since screw axes alone do not define link frames, use the model backend
        for URDF export; ``get_urdf_xyzrpy`` is unavailable for this constructor.
        """
        axes = _real_array(screw_axes, "screw_axes")
        if axes.ndim != 2 or axes.shape[1] != 6 or axes.shape[0] == 0:
            raise ValueError("screw_axes must be a non-empty array of shape (N, 6).")
        if not np.all(np.isfinite(axes)):
            raise ValueError("screw_axes contains non-finite values.")
        if not np.allclose(np.linalg.norm(axes[:, 3:], axis=1), 1.0, atol=1e-8, rtol=0):
            raise ValueError(
                "Each revolute screw axis must have a unit angular component."
            )
        if not np.allclose(np.sum(axes[:, :3] * axes[:, 3:], axis=1), 0.0, atol=1e-8):
            raise ValueError(
                "Only zero-pitch revolute screw axes are supported (v dot w = 0)."
            )
        pose_array = _real_array(
            zero_conf_ee_pose.A
            if isinstance(zero_conf_ee_pose, SE3)
            else zero_conf_ee_pose,
            "zero_conf_ee_pose",
        )
        if pose_array.shape != (4, 4) or not np.all(np.isfinite(pose_array)):
            raise ValueError("zero_conf_ee_pose must be a finite 4x4 SE(3) transform.")
        try:
            home = SE3(pose_array.copy(), check=True)
        except ValueError as exc:
            raise ValueError(
                "zero_conf_ee_pose must be a valid SE(3) transform."
            ) from exc
        names = (
            list(joint_names)
            if joint_names is not None
            else [f"joint_{index + 1}" for index in range(len(axes))]
        )
        if len(names) != len(axes) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError(
                "joint_names must contain one non-empty string per screw axis."
            )
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique.")
        instance = cls.__new__(cls)
        instance.robot_model = None
        instance.verbose = verbose
        instance.position_only = False
        instance.joint_names = names
        instance.joints = [SimpleNamespace(name=name) for name in names]
        instance.joint_screw_axis = [axis.copy() for axis in axes]
        instance.zero_conf_EE_pose = home
        instance.joint_local_nominal_poses = None
        instance.joint_zero_conf_poses = None
        instance.N_JOINTS = len(axes)
        instance._observations = []
        instance._joint_positions = []
        instance._B_Matrix = None
        instance._initial_model_state = instance._model_snapshot()
        return instance

    def reset(self):
        """Restore the nominal kinematic model used at construction time.

        Observations are retained, allowing another solve with different
        numerical options without reloading the dataset.
        """
        self._restore_model_snapshot(self._initial_model_state)
        self._B_Matrix = None

    def set_observations(self, joint_positions, observations):
        """
        Parameters
        ----------
        joint_positions: list
            List of numpy arrays where each array has a shape of 1xN_JOINTS.
        observations: list
            List of SE3 objects in the case where observations are poses
            OR of 3x1 arrays in the case where observations are points.
        """
        if len(joint_positions) != len(observations):
            raise ValueError("The size of both lists must be equal.")
        if len(observations) == 0:
            raise ValueError("At least one observation is required.")

        normalized_joint_positions = []
        for index, positions in enumerate(joint_positions):
            positions = _real_array(positions, f"Joint configuration {index}").reshape(
                -1
            )
            if positions.shape != (self.N_JOINTS,):
                raise ValueError(
                    f"Joint configuration {index} must contain exactly "
                    f"{self.N_JOINTS} values."
                )
            if not np.all(np.isfinite(positions)):
                raise ValueError(
                    f"Joint configuration {index} contains non-finite values."
                )
            normalized_joint_positions.append(positions.copy())

        if isinstance(observations[0], SE3):
            self.position_only = False
            self.OBS_SIZE = 6
            if not all(isinstance(observation, SE3) for observation in observations):
                raise ValueError("All observations must have the same type.")
            normalized_observations = []
            for index, observation in enumerate(observations):
                matrix = _real_array(observation.A, f"Pose observation {index}")
                if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Pose observation {index} must be a finite single SE3 pose."
                    )
                try:
                    normalized_observations.append(SE3(matrix.copy(), check=True))
                except ValueError as exc:
                    raise ValueError(
                        f"Pose observation {index} is not a valid SE3 pose."
                    ) from exc
        else:
            self.position_only = True
            self.OBS_SIZE = 3
            normalized_observations = []
            for index, observation in enumerate(observations):
                try:
                    point = _real_array(
                        observation, f"Position observation {index}"
                    ).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Observations must be SE3 poses or 3D points."
                    ) from exc
                if point.shape != (3,):
                    raise ValueError(
                        f"Position observation {index} must contain exactly 3 values."
                    )
                if not np.all(np.isfinite(point)):
                    raise ValueError(
                        f"Position observation {index} contains non-finite values."
                    )
                normalized_observations.append(point.copy())

        # Number of kinematic parameters in the model
        self.N_PARAMS = 4 * self.N_JOINTS + self.OBS_SIZE

        # Number of observations (not equals to the length of the observation vector)
        self.N_OBSERVATIONS = len(observations)

        self._observations = normalized_observations
        self._joint_positions = normalized_joint_positions
        self._joint_positions_array = np.asarray(normalized_joint_positions)
        self._observation_array = np.asarray(
            normalized_observations
            if self.position_only
            else [pose.A for pose in normalized_observations]
        )
        # Reloading observations may switch between 3D points and 6D poses.
        # The basis matrix embeds OBS_SIZE and must be rebuilt in that case.
        self._B_Matrix = None

    def forward_kinematics(self, link_index: int, joint_positions: np.ndarray):
        """
        Compute a POE motion prefix or the complete end-effector pose.

        Parameters
        -----------
        link_index: int
            Zero returns identity; 1 through N_JOINTS apply that many joint
            motions. N_JOINTS+1 also appends the end-effector home transform.
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.

        Returns
        --------
        spatialmath.SE3
            The requested motion product. Intermediate prefixes omit link home
            transforms and therefore do not represent physical link frames.
        """

        # Verify that i is within the range of the number of joints
        if link_index < 0 or link_index > self.N_JOINTS + 1:
            raise ValueError(
                f"Link index 'i' must be between 0 and {self.N_JOINTS + 1}."
            )

        joint_positions = _real_array(joint_positions, "joint_positions").reshape(-1)
        if joint_positions.shape != (self.N_JOINTS,):
            raise ValueError(
                f"joint_positions must contain exactly {self.N_JOINTS} values."
            )
        if not np.all(np.isfinite(joint_positions)):
            raise ValueError("joint_positions contains non-finite values.")

        T = SE3()
        for i in range(min(link_index, self.N_JOINTS)):
            T *= Twist3(self.joint_screw_axis[i]).exp(joint_positions[i])

        # Only the terminal index includes the end-effector home transform.
        if link_index == self.N_JOINTS + 1:
            T *= self.zero_conf_EE_pose

        return T

    def _batch_joint_positions(self, joint_positions):
        positions = _real_array(joint_positions, "joint_positions")
        if (
            positions.ndim != 2
            or positions.shape[1] != self.N_JOINTS
            or not len(positions)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                f"joint_positions must be a finite non-empty (N, {self.N_JOINTS}) array."
            )
        return positions

    def _model_batch(self, joint_positions, *, with_regressor=False):
        return evaluate_poe_batch(
            self.joint_screw_axis,
            self.zero_conf_EE_pose.A,
            joint_positions,
            self.B_Matrix() if with_regressor else None,
            self.OBS_SIZE if with_regressor else 6,
        )

    def forward_kinematics_batch(self, joint_positions):
        """Return terminal ``Base_T_EE`` transforms for an ``(N, DOF)`` array.

        This is equivalent to stacking ``forward_kinematics(N_JOINTS+1, q)``
        without creating a SpatialMath object for every intermediate transform.
        The return shape is ``(N, 4, 4)`` and no observation data is needed.
        """
        positions = self._batch_joint_positions(joint_positions)
        return self._model_batch(positions)[0]

    def regressor_blocks(self, joint_positions):
        """Return spatial adjoint blocks equivalent to stacked ``A_Matrix``.

        Call ``set_observations`` first to choose 3D point or 6D pose parameter
        dimensions. The return shape is ``(N, 6, 4*DOF+OBS_SIZE)``; point-only
        endpoint projection is applied separately by the solver, as in the
        scalar reference implementation. No cache survives a model update.
        """
        if not hasattr(self, "OBS_SIZE"):
            raise RuntimeError("Call set_observations() before building regressors.")
        positions = self._batch_joint_positions(joint_positions)
        return self._model_batch(positions, with_regressor=True)[1]

    def update_twist_definitions(
        self, param_corrections: np.ndarray, step_size: float = 1.0
    ):
        """
        Update the twist definitions of the joints such as to account for the error in the EE pose.

        Parameters
        -----------
        param_corrections: numpy.ndarray
            The parameter corrections to apply to each joint that, together, account for the error in the EE pose.
            The shape of the array is 4*N_JOINTS+OBS_SIZE x 1.
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        """
        # Verify that the shape of the array is correct
        param_corrections = _real_array(param_corrections, "param_corrections").reshape(
            -1
        )
        if not np.all(np.isfinite(param_corrections)):
            raise ValueError("param_corrections contains non-finite values.")
        if param_corrections.shape != (4 * self.N_JOINTS + self.OBS_SIZE,):
            raise ValueError(
                f"The shape of the parameter corrections array must be (4*N_JOINTS+OBS_SIZE, 1)=({4 * self.N_JOINTS + self.OBS_SIZE}, 1)."
            )

        # Verify that the step size is positive
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("The step size must be positive.")
        param_corrections = step_size * param_corrections

        # Get the B matrix that maps the 4 parameters of each joint to a 6-dimensional twist
        B_all = self.B_Matrix()

        # Update the pose of the joints
        for i in range(self.N_JOINTS):
            B_i = B_all[6 * i : 6 * (i + 1), 4 * i : 4 * (i + 1)]
            k_i = param_corrections[4 * i : 4 * (i + 1)].reshape(4)
            # Eq 29 in Adjoint Error Model paper
            # This produces a 6-dimensional twist correction from the 4 parameters
            n_i = B_i @ k_i

            # Update the screw axis of the joint
            new_screw_axis_i = Twist3(n_i).SE3().Ad() @ self.joint_screw_axis[i]
            self.joint_screw_axis[i] = new_screw_axis_i

            if self.joint_zero_conf_poses is not None:
                # Update the pose of the joint when the robot is in the zero configuration
                self.joint_zero_conf_poses[i] = (
                    Twist3(n_i).SE3() @ self.joint_zero_conf_poses[i]
                )

        # Update the pose of the end effector
        # Eq 29 in Adjoint Error Model paper
        if self.position_only:
            self.zero_conf_EE_pose.t += param_corrections[-3:].reshape(3)
        else:
            k_st = param_corrections[-6:].reshape(6)
            new_ee_pose = Twist3(k_st).SE3() @ self.zero_conf_EE_pose
            self.zero_conf_EE_pose = new_ee_pose

    def B_Matrix(self):
        """
        Matrix of bases. Each B is a matrix mapping 4 parameters to a twist about a given screw axis.

        The minimal parametrization of a joint is 4-dimensional (as with DH parameters) but a twist is 6-dimensional.
        Hence, identifying twist corrections directly results in an over parametrization (although it is possible).
        This algorithm uses a minimal parametrization (4 parameters per joint) and B is used to map between a set of
        4 parameters to a 6 dimensional twist.
        """

        # Keep the chosen correction basis fixed while the fitted axes change.
        # Reloading observations or resetting the model rebuilds this basis.

        # If the B matrix has already been computed, return it
        if self._B_Matrix is not None:
            return self._B_Matrix

        B_all = np.zeros(
            (6 * self.N_JOINTS + self.OBS_SIZE, 4 * self.N_JOINTS + self.OBS_SIZE)
        )

        for i in range(self.N_JOINTS):
            joint_screw_axis = self.joint_screw_axis[i]
            v = joint_screw_axis[0:3]
            w = joint_screw_axis[3:6]
            q = np.cross(w, v)
            # Find mutually perpendicular vectors w_1 and w_2 that are perpendicular to w_n
            # Deterministically choose the Cartesian axis least aligned with w.
            # This avoids random results and near-zero cross products.
            reference_axis = np.eye(3)[np.argmin(np.abs(w))]
            w_1 = np.cross(w, reference_axis)
            w_1 /= np.linalg.norm(w_1)
            w_2 = np.cross(w, w_1)
            w_2 /= np.linalg.norm(w_2)
            # Fill in the B matrix with column basis vectors
            B = np.zeros((6, 4))
            B[0:3, 0] = w_1
            B[0:3, 1] = w_2
            B[0:3, 2] = np.cross(q, w_1)
            B[3:6, 2] = w_1
            B[0:3, 3] = np.cross(q, w_2)
            B[3:6, 3] = w_2
            B_all[6 * i : 6 * (i + 1), 4 * i : 4 * (i + 1)] = B

        # Set E_st
        B_all[-self.OBS_SIZE :, -self.OBS_SIZE :] = np.eye(self.OBS_SIZE)

        # If position_only, B is a (6*N_JOINTS+3) x (4*N_JOINTS+3) matrix
        # otherwise, B is a (6*N_JOINTS+6) x (4*N_JOINTS+6) matrix
        self._B_Matrix = B_all
        return B_all

    def Q_Matrix(self, joint_positions: np.ndarray):
        """Return spatial motion blocks, shape (6, 6*N_JOINTS+OBS_SIZE)."""

        Q = np.ndarray((6, 6 * self.N_JOINTS + self.OBS_SIZE))
        prev_Ad = np.eye(6)
        for i in range(self.N_JOINTS):
            # Eq 20 of Adjoint Error Model paper
            Ad_i = (
                prev_Ad @ Twist3(self.joint_screw_axis[i]).exp(joint_positions[i]).Ad()
            )
            # Eq 25 of Adjoint Error Model paper
            Q_i = prev_Ad - Ad_i
            Q[:, 6 * i : 6 * (i + 1)] = Q_i
            prev_Ad = Ad_i

        # Set Q_st according to Eq 25 of Adjoint Error Model paper
        Q[:, -self.OBS_SIZE :] = prev_Ad[:, : self.OBS_SIZE]

        return Q

    def A_Matrix(self, joint_positions: np.ndarray):
        """
        Build a matrix of elements that will be multiplied by the joint pose perturbations such as to account for the TCP error.
        See equations (16) and (25) in the paper.

        Parameters
        -----------
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.

        Returns
        --------
        The A matrix for the current observation, shape (6, 4*N_JOINTS+OBS_SIZE).
        """
        Q = self.Q_Matrix(joint_positions)
        B = self.B_Matrix()

        A = Q @ B

        # If position_only, A is a 6 x (4*N_JOINTS+3) matrix
        # otherwise, A is a 6 x (4*N_JOINTS+6) matrix
        return A

    def _twist_error_vector(self):
        """Return the stacked residual vector for the current model state."""
        poses = self._model_batch(self._joint_positions_array)[0]
        return self._residual_batch(poses).reshape(-1, 1)

    def _residual_batch(self, model_poses):
        if self.position_only:
            return self._observation_array - model_poses[:, :3, 3]
        rotations = self._observation_array[:, :3, :3] @ model_poses[
            :, :3, :3
        ].transpose(0, 2, 1)
        translations = self._observation_array[:, :3, 3] - np.einsum(
            "nij,nj->ni", rotations, model_poses[:, :3, 3]
        )
        return pose_log_batch(rotations, translations)

    def _linearization(self):
        poses, blocks = self._model_batch(
            self._joint_positions_array, with_regressor=True
        )
        residuals = self._residual_batch(poses)
        observed_positions = (
            self._observation_array
            if self.position_only
            else self._observation_array[:, :3, 3]
        )
        position_errors = np.linalg.norm(observed_positions - poses[:, :3, 3], axis=1)
        if self.position_only:
            # Endpoint projection [I, -skew(p)] from Eq. (26).
            blocks = blocks[:, :3] - skew_batch(poses[:, :3, 3]) @ blocks[:, 3:]
            orientation_errors = np.zeros(self.N_OBSERVATIONS)
        else:
            orientation_errors = np.linalg.norm(residuals[:, 3:], axis=1)
        return (
            residuals.reshape(-1, 1),
            blocks.reshape(self.OBS_SIZE * self.N_OBSERVATIONS, self.N_PARAMS),
            position_errors.tolist(),
            orientation_errors.tolist(),
        )

    def _model_snapshot(self):
        """Capture mutable calibration state for a recoverable trial update."""
        joint_poses = None
        if self.joint_zero_conf_poses is not None:
            joint_poses = [SE3(pose.A.copy()) for pose in self.joint_zero_conf_poses]
        return (
            [axis.copy() for axis in self.joint_screw_axis],
            SE3(self.zero_conf_EE_pose.A.copy()),
            joint_poses,
        )

    def _restore_model_snapshot(self, snapshot):
        """Restore state captured by :meth:`_model_snapshot`."""
        screw_axes, zero_pose, joint_poses = snapshot
        self.joint_screw_axis = [axis.copy() for axis in screw_axes]
        self.zero_conf_EE_pose = SE3(zero_pose.A.copy())
        self.joint_zero_conf_poses = (
            None
            if joint_poses is None
            else [SE3(pose.A.copy()) for pose in joint_poses]
        )

    def solve(
        self,
        max_iterations: int = 100,
        step_size: float = 1.0,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        damping: float = 0.0,
        backtracking: bool = True,
        min_step_size: float = 1e-4,
        update_tolerance: float = 1e-10,
        relative_objective_tolerance: float = 1e-8,
        sample_weights=None,
    ):
        """
        Fit joint screw axes and the terminal pose by iterative least squares.

        Each iteration linearizes the weighted residuals and proposes parameter
        corrections. Optional backtracking restores the preceding model when
        a trial step increases the objective. Termination records convergence,
        numerical stagnation, divergence, search failure or the iteration limit.

        Parameters
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        step_size: float
            Initial multiplier for each proposed parameter correction (default = 1.0).
        position_weight: float
            Weight applied to translational residuals (default = 1.0).
            For pose observations this weights the linear part of the spatial
            SE(3) logarithm, which couples translation and rotation; it is not
            an independent Euclidean endpoint-position weight. For point
            observations it weights Euclidean position residuals.
        orientation_weight: float
            Residual multiplier for the three angular components of pose
            observations (default = 1.0).
        damping: float
            Non-negative Tikhonov damping coefficient for the least-squares
            problem (default = 0.0).
        backtracking: bool
            If True, halve the step until the weighted residual norm does not
            increase, or reject the update if all permitted steps fail
            (default = True).
        min_step_size: float
            Smallest step considered during backtracking (default = 1e-4).
        update_tolerance: float
            Maximum proposed parameter-correction norm for small-update stopping
            after failed backtracking (default = 1e-10). Coordinates mix metres
            and radians; this is not a physical accuracy bound.
        relative_objective_tolerance: float
            Maximum relative change in squared objective between consecutive
            linearizations for small-update stopping (default = 1e-8).
        sample_weights: array_like | None
            Strictly positive finite real residual multipliers of shape
            ``(N_OBSERVATIONS,)``. Each value multiplies every residual row of
            its observation, in addition to position/orientation weights.
            Values are not squared-loss weights: pass sqrt(precision) when
            precision is the desired squared-loss multiplier. Zeros and
            implicit sample removal are rejected. Default: all ones.
        """

        if not self._observations:
            raise RuntimeError(
                "No observations have been set. Call set_observations() first."
            )
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a positive finite number.")
        if not np.isfinite(position_weight) or position_weight <= 0:
            raise ValueError("position_weight must be a positive finite number.")
        if not np.isfinite(orientation_weight) or orientation_weight <= 0:
            raise ValueError("orientation_weight must be a positive finite number.")
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("damping must be a non-negative finite number.")
        if not isinstance(backtracking, bool):
            raise TypeError("backtracking must be a boolean.")
        if not np.isfinite(min_step_size) or min_step_size <= 0:
            raise ValueError("min_step_size must be a positive finite number.")
        if min_step_size > step_size:
            raise ValueError("min_step_size cannot be greater than step_size.")
        for value, name in [
            (update_tolerance, "update_tolerance"),
            (relative_objective_tolerance, "relative_objective_tolerance"),
        ]:
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number.")

        if self.position_only:
            if 3 * self.N_OBSERVATIONS < 4 * self.N_JOINTS + 3:
                raise ValueError("Not enough observations to solve.")
        else:
            if 6 * self.N_OBSERVATIONS < 4 * self.N_JOINTS + 6:
                raise ValueError("Not enough observations to solve.")

        calibration_result = CalibrationResult(self.robot_model, joints=self.joints)

        if self.position_only:
            row_weights = np.full(self.OBS_SIZE * self.N_OBSERVATIONS, position_weight)
        else:
            observation_weights = np.array(
                [position_weight] * 3 + [orientation_weight] * 3
            )
            row_weights = np.tile(observation_weights, self.N_OBSERVATIONS)
        if sample_weights is not None:
            sample_multipliers = _real_array(sample_weights, "sample_weights")
            if sample_multipliers.shape != (self.N_OBSERVATIONS,) or not np.all(
                np.isfinite(sample_multipliers) & (sample_multipliers > 0)
            ):
                raise ValueError(
                    "sample_weights must be a positive finite vector "
                    "of shape (N_OBSERVATIONS,)."
                )
            with np.errstate(over="ignore", under="ignore"):
                row_weights *= np.repeat(sample_multipliers, self.OBS_SIZE)
        if not np.all(np.isfinite(row_weights) & (row_weights > 0)):
            raise ValueError(
                "Combined residual weights must remain positive and finite."
            )
        objective_scale = float(np.max(row_weights))

        for _ in range(max_iterations):
            residuals, regressor, position_errors, orientation_errors = (
                self._linearization()
            )
            weighted_regressor = row_weights[:, None] * regressor
            weighted_residuals = row_weights[:, None] * residuals

            identifiable_parameter_indices = np.array(range(self.N_PARAMS))
            regressor_rank = np.linalg.matrix_rank(weighted_regressor)
            if regressor_rank < len(identifiable_parameter_indices):
                if (
                    self.position_only
                    and len(identifiable_parameter_indices) - regressor_rank == 2
                ):
                    # A point on the final revolute axis can confound the
                    # terminal pose with the last joint's geometry.
                    print(
                        "WARNING: Verify that the measurement point does not lie "
                        "on the axis of the last revolute joint."
                    )
                print(
                    "WARNING: Not all parameters are identifiable "
                    f"(rank of regressor is {regressor_rank}/"
                    f"{len(identifiable_parameter_indices)}). "
                    "You might be lacking diverse data."
                )

                # Keep columns in parameter order when they increase the rank.
                identifiable_parameter_indices = []
                for parameter_index in range(self.N_PARAMS):
                    if (
                        np.linalg.matrix_rank(
                            weighted_regressor[
                                :, identifiable_parameter_indices + [parameter_index]
                            ]
                        )
                        == len(identifiable_parameter_indices) + 1
                    ):
                        identifiable_parameter_indices.append(parameter_index)
                    else:
                        joint_index, joint_parameter_index = divmod(parameter_index, 4)
                        print(
                            f"Parameter {parameter_index} (joint {joint_index + 1}, "
                            f"param {joint_parameter_index + 1}) is not identifiable."
                        )
                identifiable_regressor = weighted_regressor[
                    :, identifiable_parameter_indices
                ]
                regressor_rank = np.linalg.matrix_rank(identifiable_regressor)
                if regressor_rank != len(identifiable_parameter_indices):
                    raise RuntimeError(
                        "Failed to isolate a full-rank set of identifiable parameters."
                    )
            else:
                identifiable_regressor = weighted_regressor

            # Solve the weighted linearized problem, with optional damping.
            if damping > 0:
                solve_regressor = np.vstack(
                    [
                        identifiable_regressor,
                        np.sqrt(damping) * np.eye(identifiable_regressor.shape[1]),
                    ]
                )
                solve_residuals = np.vstack(
                    [
                        weighted_residuals,
                        np.zeros((identifiable_regressor.shape[1], 1)),
                    ]
                )
            else:
                solve_regressor = identifiable_regressor
                solve_residuals = weighted_residuals
            parameter_corrections = np.linalg.lstsq(
                solve_regressor, solve_residuals, rcond=1e-12
            )[0]

            # Restore the full parameter layout, leaving excluded columns at zero.
            full_corrections = np.zeros((self.N_PARAMS, 1))
            for reduced_index, parameter_index in enumerate(
                identifiable_parameter_indices
            ):
                full_corrections[parameter_index] = parameter_corrections[reduced_index]
            parameter_corrections = full_corrections

            # Update the model. With backtracking enabled, reject trial steps
            # that increase the residual and restore the exact previous state.
            model_before_update = self._model_snapshot()
            pre_update_error_norm = norm(residuals)
            pre_update_objective_norm = norm(weighted_residuals)
            applied_step_size = step_size
            line_search_failed = False
            while True:
                self._restore_model_snapshot(model_before_update)
                self.update_twist_definitions(parameter_corrections, applied_step_size)
                post_update_errors = self._twist_error_vector()
                post_update_error_norm = norm(post_update_errors)
                post_update_objective_norm = norm(
                    row_weights[:, None] * post_update_errors
                )
                if (
                    not backtracking
                    or post_update_objective_norm <= pre_update_objective_norm
                ):
                    break
                applied_step_size *= 0.5
                if applied_step_size < min_step_size:
                    self._restore_model_snapshot(model_before_update)
                    applied_step_size = 0.0
                    post_update_error_norm = pre_update_error_norm
                    post_update_objective_norm = pre_update_objective_norm
                    line_search_failed = True
                    break

            iteration_result = CalibrationResult.IterationResult(
                self.joint_screw_axis,
                self.zero_conf_EE_pose,
                regressor,
                parameter_corrections,
                residuals,
                position_errors,
                orientation_errors,
                applied_step_size,
                post_update_error_norm,
                pre_update_objective_norm,
                post_update_objective_norm,
                objective_scale,
                row_weights,
            )
            calibration_result.add_iteration_result(iteration_result)

            if line_search_failed:
                calibration_result.is_diverging = False
                if (
                    pre_update_objective_norm / objective_scale
                    <= calibration_result.convergence_tolerance
                ):
                    # At machine precision, no trial step may be measurably
                    # better. Treat this as convergence instead of a failed
                    # search caused by floating-point noise.
                    calibration_result.has_converged = True
                    calibration_result.termination_reason = "converged"
                    calibration_result.convergence_criterion = "residual_norm"
                elif (
                    len(calibration_result.iteration_results) > 1
                    and any(
                        previous.step_size > 0
                        for previous in calibration_result.iteration_results[:-1]
                    )
                    and iteration_result.proposed_parameter_update_norm
                    <= update_tolerance
                    and iteration_result.relative_objective_change
                    <= relative_objective_tolerance
                ):
                    # A noisy fixed point can propose a correction below
                    # floating-point resolution. Keep this distinct from a
                    # failed search with a substantial available update.
                    calibration_result.has_converged = True
                    calibration_result.termination_reason = "small_update"
                    calibration_result.convergence_criterion = (
                        "small_update_and_relative_objective_stall"
                    )
                else:
                    calibration_result.has_converged = False
                    calibration_result.termination_reason = "line_search_failed"
                    calibration_result.convergence_criterion = None

            if self.verbose:
                print(f"Iteration #{calibration_result.nb_iterations_executed} result:")
                iteration_result.print()

            if (
                line_search_failed
                or calibration_result.has_converged
                or calibration_result.is_diverging
            ):
                if self.verbose:
                    if line_search_failed and not calibration_result.has_converged:
                        print("The line search could not find an error-reducing step.")
                    elif calibration_result.has_converged:
                        print("The kinematic calibration has converged.")
                    else:
                        print("The kinematic calibration is diverging.")
                break

        if calibration_result.termination_reason is None:
            calibration_result.termination_reason = "max_iterations"

        return calibration_result

    def get_calibration(self, calibration_result: CalibrationResult):
        """
        Compute the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.

        Returns
        --------
            screw_axes, zero_conf_EE_pose
        """
        return (
            calibration_result.get_screw_axes(),
            calibration_result.get_zero_conf_EE_pose(),
        )

    def get_urdf_xyzrpy(self, calibration_result: CalibrationResult):
        """
        Compute the RPY-XYZ format of the joint definitions for use in a URDF file.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.

        Returns
        --------
        A list of dictionaries where each dictionary contains the following keys:
        - name: The name of the joint (e.g. "link1-link2").
        - xyz: The position of the joint relative to the previous link.
        - rpy: The orientation of the joint relative to the previous link.
        """
        return calibration_result.get_urdf_xyzrpy(self.joint_zero_conf_poses)

    def print_screw_axes(self, calibration_result: CalibrationResult):
        """
        Print the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        """
        screw_definitions = calibration_result.get_screw_axes()
        zero_conf_EE_pose = calibration_result.get_zero_conf_EE_pose()

        print("Definition of the screw axes for use with the PoE formula:")
        for i, s in enumerate(screw_definitions):
            v = s[0:3]
            w = s[3:6]
            print(f"\tJoint {i + 1}:")
            print(f"\t\tv: {v}")
            print(f"\t\tw: {w}")
        print("\tZero configuration pose:")
        print(f"\t\t{zero_conf_EE_pose.A[0, :]}")
        print(f"\t\t{zero_conf_EE_pose.A[1, :]}")
        print(f"\t\t{zero_conf_EE_pose.A[2, :]}")
        print(f"\t\t{zero_conf_EE_pose.A[3, :]}")

    def print_urdf_joint_definitions(self, calibration_result: CalibrationResult):
        """
        Print the joint definitions for use in a URDF file.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        """
        if self.joint_local_nominal_poses is not None:
            urdf_joint_definitions = self.get_urdf_xyzrpy(calibration_result)

            print("Definition of the joints in RPY-XYZ format for use in a URDF:")
            for j in urdf_joint_definitions:
                print(f"\tJoint {j['name']}")
                print(f"\t\tXYZ: {j['xyz']}")
                print(f"\t\tRPY: {j['rpy']}")
        else:
            print("Cannot print URDF joint definitions for a PoERobot.")
