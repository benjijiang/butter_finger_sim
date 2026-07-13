"""Butter Finger: shared control API for the four-joint desktop arm.

Application code talks to the arm only through the ArmBackend interface,
always in radians. Backends translate radians into whatever their target
needs: PyBullet joint targets in simulation, or (in the future, on the
Raspberry Pi) calibrated PWM microseconds.

Importing this package does NOT import PyBullet or any hardware library;
backends load their dependencies lazily when instantiated.
"""
from __future__ import annotations

from butter_finger.arm import (
    ArmBackend,
    BackendUnavailableError,
    JointLimitError,
    UnknownJointError,
)
from butter_finger.backends.pybullet_arm import PyBulletArm
from butter_finger.backends.raspberry_pi_arm import RaspberryPiArm
from butter_finger.config import JOINT_NAMES, ArmConfig, load_arm_config

__all__ = [
    "ArmBackend",
    "ArmConfig",
    "BackendUnavailableError",
    "JOINT_NAMES",
    "JointLimitError",
    "PyBulletArm",
    "RaspberryPiArm",
    "UnknownJointError",
    "load_arm_config",
]
