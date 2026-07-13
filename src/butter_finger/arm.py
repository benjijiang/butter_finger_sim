"""Simulator-independent arm interface.

Architecture rule: application code commands the arm ONLY in radians via
ArmBackend. High-level code must never send PWM directly.

    Application command in radians
                |
           ArmBackend API
            /          \\
    PyBulletArm     RaspberryPiArm
    joint target    future calibrated PWM
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class UnknownJointError(ValueError):
    """A joint name is not one of the configured arm joints."""


class JointLimitError(ValueError):
    """A commanded position is outside the configured joint limits."""


class BackendUnavailableError(RuntimeError):
    """The backend's runtime dependency (e.g. PyBullet) is not available."""


class ArmBackend(ABC):
    """Abstract four-joint arm. All positions are radians."""

    @abstractmethod
    def move_joint(self, joint: str, position_rad: float) -> None:
        """Command one joint in radians."""
        raise NotImplementedError

    @abstractmethod
    def move_joints(self, targets_rad: Mapping[str, float]) -> None:
        """Command multiple joints in radians."""
        raise NotImplementedError

    @abstractmethod
    def get_joint_positions(self) -> dict[str, float]:
        """Return joint positions in radians."""
        raise NotImplementedError

    @abstractmethod
    def go_home(self) -> None:
        """Move to the configured simulated reference pose."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Release simulator or hardware resources."""
        raise NotImplementedError

    def __enter__(self) -> "ArmBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
