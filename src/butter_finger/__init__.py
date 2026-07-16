"""Butter Finger: shared control API for the four-joint desktop arm.

Simulation-facing application code talks to the arm only through the
ArmBackend interface, always in radians. Backends translate radians into
whatever their target needs: PyBullet joint targets in simulation, or (in
the Raspberry Pi) calibrated PWM microseconds. PWMRobotArm remains available
as a low-level microseconds API for hardware diagnostics.

Importing this package does NOT import PyBullet or any hardware library;
backends load their dependencies lazily when instantiated.
"""
from __future__ import annotations

from butter_finger.actions import ActionRunner, UnknownActionError
from butter_finger.arm import (
    ArmBackend,
    BackendUnavailableError,
    JointLimitError,
    JointStateUnavailableError,
    UnknownJointError,
)
from butter_finger.backends.pwm_robot_arm import PWMRobotArm
from butter_finger.backends.pybullet_arm import PyBulletArm
from butter_finger.backends.raspberry_pi_arm import RaspberryPiArm
from butter_finger.config import (
    JOINT_NAMES,
    ActionConfig,
    ActionStep,
    ArmAction,
    ArmConfig,
    CameraConfig,
    IdleConfig,
    JointCalibration,
    PhysicalConfig,
    load_action_config,
    load_arm_config,
    load_camera_config,
    load_idle_config,
    load_physical_config,
)
from butter_finger.idle import IdleController

__all__ = [
    "ActionConfig",
    "ActionRunner",
    "ActionStep",
    "ArmBackend",
    "ArmAction",
    "ArmConfig",
    "BackendUnavailableError",
    "CameraConfig",
    "IdleConfig",
    "IdleController",
    "JOINT_NAMES",
    "JointCalibration",
    "JointLimitError",
    "JointStateUnavailableError",
    "PWMRobotArm",
    "PhysicalConfig",
    "PyBulletArm",
    "RaspberryPiArm",
    "UnknownActionError",
    "UnknownJointError",
    "load_action_config",
    "load_arm_config",
    "load_camera_config",
    "load_idle_config",
    "load_physical_config",
]
