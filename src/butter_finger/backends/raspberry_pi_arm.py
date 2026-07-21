"""Calibrated radians backend for the physical Raspberry Pi arm.

Data path:

    Raspberry Pi 5 --UART--> Hiwonder RasAdapter5A V1.0 --PWM--> servos

    PWM ports: base=1, shoulder=3, elbow=4, wrist=5 (2 and 6 unused).

The adapter only interpolates within measured two-point calibration ranges.
It never clamps or extrapolates. The hardware is open-loop, so reported
positions are the last complete command estimate, not sensor feedback.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from butter_finger.arm import (
    ArmBackend,
    JointLimitError,
    JointStateUnavailableError,
    UnknownJointError,
)
from butter_finger.backends.pwm_robot_arm import PWMRobotArm
from butter_finger.config import PhysicalConfig, load_physical_config

_HOME_DURATION_S = 3.0


class RaspberryPiArm(ArmBackend):
    """Physical arm adapter accepting only calibrated radian targets.

    ``pwm_arm`` may be injected for tests. When omitted, constructing this
    class lazily opens the real Hiwonder board through ``PWMRobotArm``.

    ``stream_duration_s`` sets the board interpolation window used for
    non-blocking commands (``duration_s=None``). Leave it None to keep
    ``PWMRobotArm``'s one-second default, which suits occasional single
    targets. A streamed controller that re-commands many times per second
    should pass roughly its own loop period instead, so each command can
    actually complete before the next one replaces it.
    """

    def __init__(
        self,
        pwm_arm: PWMRobotArm | None = None,
        config: PhysicalConfig | None = None,
        stream_duration_s: float | None = None,
    ) -> None:
        pwm_config = (
            pwm_arm.config
            if pwm_arm is not None and hasattr(pwm_arm, "config")
            else None
        )
        if config is None:
            config = pwm_config if pwm_config is not None else load_physical_config()
        elif pwm_config is not None and config != pwm_config:
            raise ValueError(
                "RaspberryPiArm and its injected PWMRobotArm must use the "
                "same PhysicalConfig"
            )
        self._config = config
        self._pwm_arm = (
            pwm_arm if pwm_arm is not None else PWMRobotArm(config=self._config)
        )
        self._validate_duration(stream_duration_s)
        self._stream_duration_s = (
            None if stream_duration_s is None else float(stream_duration_s)
        )
        self._last_commanded_rad: dict[str, float] = {}

    @property
    def config(self) -> PhysicalConfig:
        return self._config

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._config.joint_order

    def _validate_joint(self, joint: str) -> None:
        if joint not in self._config.calibrations:
            raise UnknownJointError(
                f"Unknown joint {joint!r}; expected one of "
                f"{list(self._config.joint_order)}"
            )

    def _validated_targets(
        self, targets_rad: Mapping[str, float]
    ) -> dict[str, float]:
        validated: dict[str, float] = {}
        for joint, raw_position in targets_rad.items():
            self._validate_joint(joint)
            if (
                isinstance(raw_position, bool)
                or not isinstance(raw_position, (int, float))
                or not math.isfinite(raw_position)
            ):
                raise ValueError(
                    f"Target for joint {joint!r} must be a finite number"
                )
            position = float(raw_position)
            calibration = self._config.calibrations[joint]
            if not calibration.contains_rad(position):
                raise JointLimitError(
                    f"Target {position} rad for joint {joint!r} is outside "
                    f"the calibrated range [{calibration.lower_rad}, "
                    f"{calibration.upper_rad}] rad"
                )
            validated[joint] = position
        return validated

    @staticmethod
    def _validate_duration(duration_s: float | None) -> None:
        if duration_s is None:
            return
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError("duration_s must be a finite number greater than zero")

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        """Validate all radians without issuing a PWM command."""
        self._validated_targets(targets_rad)

    def rad_to_pwm(self, joint: str, position_rad: float) -> float:
        """Convert one validated joint target to an unrounded PWM value."""
        position = self._validated_targets({joint: position_rad})[joint]
        return self._config.calibrations[joint].rad_to_pwm(position)

    def pwm_to_rad(self, joint: str, pulse_us: float) -> float:
        """Convert an in-calibration PWM value to its radian estimate."""
        self._validate_joint(joint)
        return self._config.calibrations[joint].pwm_to_rad(pulse_us)

    def move_joint(
        self,
        joint: str,
        position_rad: float,
        *,
        duration_s: float | None = None,
    ) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(
        self,
        targets_rad: Mapping[str, float],
        *,
        duration_s: float | None = None,
    ) -> None:
        targets = self._validated_targets(targets_rad)
        self._validate_duration(duration_s)
        if not targets:
            return

        targets_pwm = {
            joint: self._config.calibrations[joint].rad_to_pwm(position)
            for joint, position in targets.items()
        }
        if duration_s is None:
            # Non-blocking target update. The board always interpolates over
            # some window; PWMRobotArm's default is a full second, which is far
            # too long for a streamed control loop that re-commands many times
            # per second — every command would be superseded after travelling a
            # few percent, leaving the arm about a second behind its target.
            if self._stream_duration_s is None:
                self._pwm_arm.move_joints(targets_pwm)
            else:
                self._pwm_arm.move_joints(
                    targets_pwm, duration=self._stream_duration_s
                )
        else:
            duration = float(duration_s)
            self._pwm_arm.move_joints(targets_pwm, duration=duration)
        self._last_commanded_rad.update(targets)
        if duration_s is not None:
            self._pwm_arm.wait(float(duration_s))

    def get_joint_positions(self) -> dict[str, float]:
        missing = [
            joint
            for joint in self._config.joint_order
            if joint not in self._last_commanded_rad
        ]
        if missing:
            raise JointStateUnavailableError(
                "The real arm has no joint-angle feedback and no complete "
                f"last-command estimate yet; missing {missing}. Call go_home() "
                "or command every joint first."
            )
        return {
            joint: self._last_commanded_rad[joint]
            for joint in self._config.joint_order
        }

    def go_home(self) -> None:
        self._pwm_arm.home(duration=_HOME_DURATION_S)
        self._last_commanded_rad = {
            joint: self._config.calibrations[joint].pwm_to_rad(
                self._config.home_pwm_us[joint]
            )
            for joint in self._config.joint_order
        }
        self._pwm_arm.wait(_HOME_DURATION_S)

    def disconnect(self) -> None:
        disconnect = getattr(self._pwm_arm, "disconnect", None)
        if callable(disconnect):
            disconnect()
