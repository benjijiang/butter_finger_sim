"""PWM-level hardware layer for the real Butter Finger arm.

This is the verified port of the original ``board_demo/butter_finger.py``
``RobotArm`` class that runs on the Raspberry Pi 5. It commands the servos
in raw PWM pulse widths (MICROSECONDS) through the Hiwonder SDK:

    Raspberry Pi 5 --UART (/dev/ttyAMA0 @ 1000000)--> RasAdapter5A --PWM--> servos

It is deliberately NOT an ArmBackend: the ArmBackend interface is radians.
RaspberryPiArm wraps this low-level class with the measured linear
calibration. Direct PWM control remains available for hardware diagnostics.
Limits come from config/joints.yaml and out-of-range pulses are rejected,
never clamped.

SDK resolution (lazy, at instantiation only, so the rest of the package
works on machines without the SDK or a serial port):

  1. ``import ros_robot_controller_sdk`` if already importable.
  2. The directory in the BUTTER_FINGER_SDK_PATH environment variable.
  3. The parent directory of this repository checkout (the repo is expected
     to live inside ``board_demo/``, next to ``ros_robot_controller_sdk.py``).
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from butter_finger.arm import (
    BackendUnavailableError,
    JointLimitError,
    UnknownJointError,
)
from butter_finger.config import PhysicalConfig, load_physical_config

_SDK_MODULE_NAME = "ros_robot_controller_sdk"

_MISSING_SDK_MSG = (
    "Could not import the Hiwonder SDK module 'ros_robot_controller_sdk'. "
    "This PWM hardware layer only works on the Raspberry Pi where "
    "board_demo/ros_robot_controller_sdk.py exists. Searched: sys.path, "
    "$BUTTER_FINGER_SDK_PATH, and {parent}. Set BUTTER_FINGER_SDK_PATH to "
    "the directory containing ros_robot_controller_sdk.py, or use "
    "PyBulletArm for simulation instead."
)


def _repo_parent_dir() -> Path:
    """The directory containing this repository checkout (e.g. board_demo/)."""
    return Path(__file__).resolve().parents[4]


def _import_sdk():
    try:
        return __import__(_SDK_MODULE_NAME)
    except ImportError:
        pass

    candidates = []
    env_path = os.environ.get("BUTTER_FINGER_SDK_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_repo_parent_dir())

    for directory in candidates:
        if (directory / f"{_SDK_MODULE_NAME}.py").is_file():
            directory_str = str(directory)
            if directory_str not in sys.path:
                sys.path.insert(0, directory_str)
            try:
                return __import__(_SDK_MODULE_NAME)
            except ImportError as exc:
                raise BackendUnavailableError(
                    f"Found {_SDK_MODULE_NAME}.py in {directory} but importing "
                    f"it failed (missing pyserial?): {exc}"
                ) from exc

    raise BackendUnavailableError(_MISSING_SDK_MSG.format(parent=_repo_parent_dir()))


class PWMRobotArm:
    """The real four-joint arm, commanded in PWM pulse widths (microseconds).

    Parameters
    ----------
    board:
        A pre-constructed SDK ``Board`` object (used by tests to inject a
        fake). When omitted, the Hiwonder SDK is imported lazily and a real
        ``Board`` is opened on the serial port.
    config:
        Pre-loaded PhysicalConfig; loaded from config/joints.yaml when omitted.
    """

    def __init__(self, board=None, config: PhysicalConfig | None = None) -> None:
        self._config = config if config is not None else load_physical_config()
        if board is None:
            sdk = _import_sdk()
            board = sdk.Board()
        self.board = board

    @property
    def config(self) -> PhysicalConfig:
        return self._config

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._config.joint_order

    def validate_pulse(self, joint: str, pulse_us: float) -> int:
        """Check a pulse against the tested hardware limits; never clamp."""
        if joint not in self._config.pwm_ports:
            raise UnknownJointError(
                f"Unknown joint {joint!r}; expected one of {list(self._config.pwm_ports)}"
            )
        limits = self._config.pulse_limits_us[joint]
        if not limits.contains(pulse_us):
            raise JointLimitError(
                f"{joint} pulse {pulse_us} us is outside the tested range "
                f"[{limits.min_us}, {limits.max_us}] us"
            )
        return round(pulse_us)

    def move_joint(self, joint: str, pulse_us: float, duration: float = 1.0) -> None:
        """Move one joint to a PWM pulse width (microseconds).

        The controller board interpolates the motion over ``duration`` seconds.
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than zero")
        pulse = self.validate_pulse(joint, pulse_us)
        port = self._config.pwm_ports[joint]
        print(f"Moving {joint}: pulse {pulse}, duration {duration:.1f}s")
        self.board.pwm_servo_set_position(duration, [[port, pulse]])

    def move_joints(self, targets_us: Mapping[str, float], duration: float = 1.0) -> None:
        """Move multiple joints simultaneously (PWM microseconds).

        All targets are validated before any command is sent, so an invalid
        request leaves the arm untouched.
        """
        if duration <= 0:
            raise ValueError("Duration must be greater than zero")
        commands = []
        for joint, pulse_us in targets_us.items():
            pulse = self.validate_pulse(joint, pulse_us)
            print(f"{joint}: pulse {pulse}")
            commands.append([self._config.pwm_ports[joint], pulse])
        self.board.pwm_servo_set_position(duration, commands)

    def home(self, duration: float = 3.0) -> None:
        """Move all joints to the recorded physical home pose."""
        print("Returning to home pose")
        self.move_joints(self._config.home_pwm_us, duration)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def disconnect(self) -> None:
        """Close the SDK serial port when one is present; safe repeatedly."""
        disable_reception = getattr(self.board, "enable_reception", None)
        if callable(disable_reception):
            disable_reception(False)
        port = getattr(self.board, "port", None)
        close = getattr(port, "close", None)
        if callable(close) and getattr(port, "is_open", True):
            close()
