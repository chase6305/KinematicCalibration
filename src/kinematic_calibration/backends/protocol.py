"""Small engine boundary shared by simulation, acquisition, and viewers."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class KinematicsBackend(Protocol):
    """A serial chain, with a fixed base and named terminal frame.

    ``link_transforms`` includes the base followed by each chain link.
    Implementations must reject unsupported URDF constructs explicitly.
    """

    joint_names: tuple[str, ...]
    joint_limits: np.ndarray
    base_link: str
    end_link: str

    @property
    def dof(self) -> int: ...

    def forward(self, joint_positions: np.ndarray) -> np.ndarray:
        """Return Base_T_End, shape (4,4)."""
        ...

    def link_transforms(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        """Return Base_T_Link for every link along the selected chain."""
        ...
