"""Deterministic URDF kinematics without a physics or rendering dependency.

Only fixed/revolute/continuous serial paths are supported. No resource meshes
are opened. This makes the same model usable on a server and in a Viser scene.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(text: str | None, default: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in (text or default).split()])
    except ValueError as exc:
        raise ValueError("URDF vector must contain three finite numbers") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("URDF vector must contain three finite numbers")
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class NumpyBackend:
    """A selected URDF path with arbitrary local joint axes.

    URDF RPY is fixed-axis XYZ: R = Rz(yaw) Ry(pitch) Rx(roll).
    ``forward(q)`` always expresses the end in the selected base frame.
    This backend models geometry only; it does not perform collision checking.
    """

    def __init__(self, joints: tuple[Joint, ...], base_link: str, end_link: str):
        if not joints:
            raise ValueError("selected chain must contain joints")
        names, links = set(), {base_link}
        previous = base_link
        checked = []
        for joint in joints:
            if (
                not joint.name
                or joint.name in names
                or joint.parent != previous
                or not joint.child
                or joint.child in links
            ):
                raise ValueError(
                    "joints must form a connected, acyclic serial chain with unique names"
                )
            if joint.kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"unsupported joint kind: {joint.kind!r}")
            raw_origin, raw_axis = np.asarray(joint.origin), np.asarray(joint.axis)
            if raw_origin.dtype.kind not in "iuf" or raw_axis.dtype.kind not in "iuf":
                raise ValueError(
                    "joint origins and axes must contain real numeric values"
                )
            origin = raw_origin.astype(float, copy=True)
            axis = raw_axis.astype(float, copy=True)
            if (
                origin.shape != (4, 4)
                or not np.isfinite(origin).all()
                or not np.allclose(origin[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or not np.allclose(
                    origin[:3, :3].T @ origin[:3, :3], np.eye(3), atol=1e-8, rtol=0
                )
                or not np.isclose(np.linalg.det(origin[:3, :3]), 1, atol=1e-8, rtol=0)
            ):
                raise ValueError("joint origin must be a rigid 4x4 transform")
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError("joint axis must be a finite vector with shape (3,)")
            if joint.kind != "fixed":
                if not np.isclose(np.linalg.norm(axis), 1, atol=1e-8, rtol=0):
                    raise ValueError("joint axis must have unit length")
                if (
                    not np.isfinite([joint.lower, joint.upper]).all()
                    or joint.lower >= joint.upper
                ):
                    raise ValueError("joint limits must be finite and strictly ordered")
            names.add(joint.name)
            links.add(joint.child)
            previous = joint.child
            origin.setflags(write=False)
            axis.setflags(write=False)
            checked.append(replace(joint, origin=origin, axis=axis))
        if previous != end_link:
            raise ValueError("last chain link does not match end_link")
        self.joints = tuple(checked)
        self.base_link = base_link
        self.end_link = end_link
        self.joint_names = tuple(j.name for j in self.joints if j.kind != "fixed")
        self.joint_limits = np.asarray(
            [(j.lower, j.upper) for j in self.joints if j.kind != "fixed"], dtype=float
        ).reshape(-1, 2)
        self.joint_limits.setflags(write=False)
        if not self.joint_names:
            raise ValueError("selected chain must contain at least one revolute joint")

    @classmethod
    def from_urdf(
        cls, path: str | Path, *, end_link: str, base_link: str | None = None
    ) -> "NumpyBackend":
        """Load a serial path; reject mimic, prismatic, cycles and bad topology."""
        root = ET.parse(path).getroot()
        if root.tag != "robot":
            raise ValueError("expected a URDF robot element")
        link_names = [link.get("name") for link in root.findall("link")]
        if None in link_names or len(set(link_names)) != len(link_names):
            raise ValueError("URDF links must have unique names")
        links = set(link_names)
        by_child: dict[str, Joint] = {}
        joint_names: set[str] = set()
        for element in root.findall("joint"):
            name = element.get("name")
            parent_node, child_node = element.find("parent"), element.find("child")
            if (
                not name
                or name in joint_names
                or parent_node is None
                or child_node is None
            ):
                raise ValueError("URDF joints need unique names, parent and child")
            parent, child = parent_node.get("link"), child_node.get("link")
            if parent not in links or child not in links or child in by_child:
                raise ValueError("invalid URDF parent/child topology")
            joint_names.add(name)
            origin_node, axis_node = element.find("origin"), element.find("axis")
            xyz = _vector(
                None if origin_node is None else origin_node.get("xyz"), "0 0 0"
            )
            rpy = _vector(
                None if origin_node is None else origin_node.get("rpy"), "0 0 0"
            )
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            origin[:3, 3] = xyz
            axis = _vector(None if axis_node is None else axis_node.get("xyz"), "1 0 0")
            kind = element.get("type", "")
            if element.find("mimic") is not None:
                kind = "mimic"
            limit = element.find("limit")
            lower, upper = -np.pi, np.pi
            if kind == "revolute":
                if (
                    limit is None
                    or limit.get("lower") is None
                    or limit.get("upper") is None
                ):
                    raise ValueError(f"revolute joint {name!r} requires finite limits")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                if not np.isfinite([lower, upper]).all() or lower >= upper:
                    raise ValueError(f"invalid limits for {name!r}")
            if kind in ("revolute", "continuous"):
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError(f"zero joint axis for {name!r}")
                axis = axis / norm
            by_child[child] = Joint(
                name, parent, child, kind, origin, axis, lower, upper
            )
        if end_link not in links or (base_link is not None and base_link not in links):
            raise ValueError("unknown base or end link")
        path_joints = []
        current = end_link
        visited = set()
        while current != base_link and current in by_child:
            if current in visited:
                raise ValueError("cycle in URDF chain")
            visited.add(current)
            joint = by_child[current]
            if joint.kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"unsupported joint {joint.name!r}: {joint.kind}")
            path_joints.append(joint)
            current = joint.parent
        if base_link is not None and current != base_link:
            raise ValueError("base link is not an ancestor of the end link")
        return cls(tuple(reversed(path_joints)), current, end_link)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def _validate_q(self, joint_positions: np.ndarray) -> np.ndarray:
        raw = np.asarray(joint_positions)
        if raw.dtype.kind not in "iuf" or raw.shape != (self.dof,):
            raise ValueError(f"joint_positions must be finite with shape ({self.dof},)")
        with np.errstate(over="ignore", invalid="ignore"):
            positions = raw.astype(float, copy=False)
        if not np.isfinite(positions).all():
            raise ValueError(f"joint_positions must be finite with shape ({self.dof},)")
        return positions

    def _chain_transforms(self, joint_positions: np.ndarray):
        """Yield distinct Base_T_Link arrays, sharing the chain calculation."""
        positions = self._validate_q(joint_positions)
        pose = np.eye(4)
        yield self.base_link, pose
        joint_index = 0
        for joint in self.joints:
            pose = pose @ joint.origin
            if joint.kind != "fixed":
                rotation = Rotation.from_rotvec(
                    joint.axis * positions[joint_index]
                ).as_matrix()
                # Right-multiplying a pure rotation leaves translation intact.
                pose[:, :3] = pose[:, :3] @ rotation
                joint_index += 1
            yield joint.child, pose

    def link_transforms(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        return dict(self._chain_transforms(joint_positions))

    def forward(self, joint_positions: np.ndarray) -> np.ndarray:
        # Terminal FK does not need a dictionary or copies of every link pose.
        for _, pose in self._chain_transforms(joint_positions):
            pass
        return pose

    def poe_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Return space screw axes (N,6), ordered [v,w], and Base_T_End(0)."""
        zero_poses = self.link_transforms(np.zeros(self.dof))
        axes = []
        for joint in self.joints:
            if joint.kind == "fixed":
                continue
            pose = zero_poses[joint.child]
            direction = pose[:3, :3] @ joint.axis
            axes.append(np.r_[np.cross(pose[:3, 3], direction), direction])
        return np.asarray(axes), zero_poses[self.end_link]

    def perturbed(
        self,
        *,
        seed: int = 7,
        translation_std_m: float = 0.002,
        rotation_std_rad: float = 0.003,
    ) -> "NumpyBackend":
        """Create independent truth geometry for repeatable synthetic experiments."""
        if (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            not np.isfinite([translation_std_m, rotation_std_rad]).all()
            or min(translation_std_m, rotation_std_rad) < 0
        ):
            raise ValueError(
                "perturbation standard deviations must be finite and non-negative"
            )
        rng = np.random.default_rng(seed)
        joints = []
        for joint in self.joints:
            delta = np.eye(4)
            delta[:3, 3] = rng.normal(0, translation_std_m, 3)
            delta[:3, :3] = Rotation.from_rotvec(
                rng.normal(0, rotation_std_rad, 3)
            ).as_matrix()
            joints.append(replace(joint, origin=joint.origin @ delta))
        return NumpyBackend(tuple(joints), self.base_link, self.end_link)
