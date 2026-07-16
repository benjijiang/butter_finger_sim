"""Non-blocking simulation idle motion for the camera-ended arm.

The controller owns only the no-person fallback scan. A future attention
layer should stop calling ``update`` while it has a tracked person, command
its own targets through ArmBackend, and call ``resume`` when tracking ends.
"""
from __future__ import annotations

import math

from butter_finger.arm import ArmBackend
from butter_finger.config import IdleConfig, load_idle_config


class IdleController:
    """Generate a slow, bounded triangle-wave scan around ``idle_ready``."""

    def __init__(
        self,
        arm: ArmBackend,
        config: IdleConfig | None = None,
    ) -> None:
        self.arm = arm
        self.config = config if config is not None else load_idle_config()
        self._span_rad = self.config.upper_rad - self.config.lower_rad
        self._phase_rad = (
            self.config.pose_rad[self.config.scan_joint] - self.config.lower_rad
        )
        self._position_rad = self.config.pose_rad[self.config.scan_joint]
        self._direction = 1

    @property
    def position_rad(self) -> float:
        """Current commanded position of the configured scan joint."""
        return self._position_rad

    @property
    def direction(self) -> int:
        """Current scan direction: ``1`` toward upper, ``-1`` toward lower."""
        return self._direction

    def resume(self) -> None:
        """Return to idle_ready and restart the scan toward its upper bound."""
        self._phase_rad = (
            self.config.pose_rad[self.config.scan_joint] - self.config.lower_rad
        )
        self._position_rad = self.config.pose_rad[self.config.scan_joint]
        self._direction = 1
        self.arm.move_joints(self.config.pose_rad)

    def update(self, dt_s: float) -> None:
        """Advance the scan by ``dt_s`` and send one non-blocking target."""
        if (
            isinstance(dt_s, bool)
            or not isinstance(dt_s, (int, float))
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            raise ValueError("dt_s must be a finite number greater than zero")

        self._phase_rad += self.config.scan_speed_rad_s * float(dt_s)
        cycle_rad = 2.0 * self._span_rad
        offset_rad = self._phase_rad % cycle_rad
        if offset_rad < self._span_rad:
            self._position_rad = self.config.lower_rad + offset_rad
            self._direction = 1
        else:
            self._position_rad = self.config.upper_rad - (
                offset_rad - self._span_rad
            )
            self._direction = -1

        targets = dict(self.config.pose_rad)
        targets[self.config.scan_joint] = self._position_rad
        self.arm.move_joints(targets)
