"""Stub for the future RADIANS physical-arm backend on the Raspberry Pi 5.

NOT IMPLEMENTED. This file exists only to reserve the architectural seam.
It imports no hardware library and must never gain a real servo command
until every safety precondition below is met.

Real hardware is currently driven at the PWM level instead: see
backends/pwm_robot_arm.py (PWMRobotArm, microseconds), the verified port of
the original board_demo control code. This stub is the future radians
adapter that will sit on top of that PWM layer.

Data path (for reference only):

    Raspberry Pi 5 --UART--> Hiwonder RasAdapter5A V1.0 --PWM--> servos

    PWM ports: base=1, shoulder=3, elbow=4, wrist=5 (2 and 6 unused).

Safety preconditions before implementing:
  1. Measured PWM-to-angle calibration for every joint (none exists yet).
  2. Verified safe PWM limits for every joint. Tested on the real machine:
     base/elbow/wrist 505-2495 us, shoulder 1200-2220 us. The shoulder home
     value (2200 us) was revalidated on 2026-07-13.
  3. Commands without verified calibration must be rejected, never clamped
     to guesses.

Never command real hardware using the placeholder simulation limits.
"""
from __future__ import annotations

from collections.abc import Mapping

from butter_finger.arm import ArmBackend

_NOT_IMPLEMENTED_MSG = (
    "RaspberryPiArm (radians) is not implemented yet: no measured "
    "PWM-to-angle calibration exists. Use PyBulletArm for simulation, or "
    "PWMRobotArm (PWM microseconds) to drive the real arm on the "
    "Raspberry Pi."
)


class RaspberryPiArm(ArmBackend):
    """Placeholder for the physical Butter Finger arm. Raises on every call."""

    def __init__(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def move_joint(
        self,
        joint: str,
        position_rad: float,
        *,
        duration_s: float | None = None,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def move_joints(
        self,
        targets_rad: Mapping[str, float],
        *,
        duration_s: float | None = None,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def get_joint_positions(self) -> dict[str, float]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def go_home(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def disconnect(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
