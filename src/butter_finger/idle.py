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

    def resume(
        self,
        *,
        from_position_rad: float | None = None,
        direction: int = 1,
    ) -> None:
        """Return to the idle pose and restart the scan.

        By default the scan joint goes back to its ``idle_ready`` angle, which
        is what a showcase or a cold start wants. An attention layer that has
        been panning to follow a person should instead pass that joint's
        current angle as ``from_position_rad``: a face usually reappears near
        where it vanished, so carrying on from the last bearing both avoids a
        large snap back to center and reacquires sooner. Values outside the
        scan bounds are clamped.

        ``direction`` seeds which way the sweep starts (``1`` toward the upper
        bound, ``-1`` toward the lower). The triangle wave still reverses at
        the bounds, so a direction that leaves no travel flips immediately.
        """
        if from_position_rad is None:
            start = self.config.pose_rad[self.config.scan_joint]
        else:
            if (
                isinstance(from_position_rad, bool)
                or not isinstance(from_position_rad, (int, float))
                or not math.isfinite(from_position_rad)
            ):
                raise ValueError("from_position_rad must be a finite number")
            start = min(
                max(float(from_position_rad), self.config.lower_rad),
                self.config.upper_rad,
            )
        if (
            isinstance(direction, bool)
            or not isinstance(direction, int)
            or direction not in (1, -1)
        ):
            raise ValueError("direction must be 1 or -1")

        if direction == 1:
            self._phase_rad = start - self.config.lower_rad
        else:
            self._phase_rad = self._span_rad + (self.config.upper_rad - start)
        self._position_rad = start
        self._direction = direction

        targets = dict(self.config.pose_rad)
        targets[self.config.scan_joint] = start
        self.arm.move_joints(targets)

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
