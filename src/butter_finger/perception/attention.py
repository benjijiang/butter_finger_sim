"""FaceFollower: arbitrate between face tracking and the idle scan.

This is the "attention layer" the idle controller's docstring anticipated:
while a face is visible it drives :class:`FaceTracker`; when the face is lost
for longer than a short grace period it hands off to :class:`IdleController`,
whose slow base sweep looks for a face across the arm's full yaw range. As
soon as a face is reacquired it switches back to tracking.

The hand-off itself is a third state. Going straight to the idle scan meant
one unbounded ``move_joints`` — the only command in the whole follow stack not
subject to a slew limit — which on hardware asks the board to cover the entire
tracking excursion within its short streaming window, and the arm slams. So
losing a face instead enters ``returning``: the posture joints ease toward the
idle pose at ``return_rate_rad_s`` over as many ticks as that takes, the pan
joint holds the bearing where the face was last seen, and only then does the
scan start — from that bearing, heading the way the face was last moving.

Pure control over the radians-only ``ArmBackend`` — no PyBullet, no OpenCV —
so it is fully unit-testable. The caller supplies the per-tick detection (from
a camera or a synthetic source) and the elapsed time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from butter_finger.idle import IdleController
from butter_finger.perception.detection import Detection
from butter_finger.perception.tracker import FaceTracker, TrackerStep


# Default posture-return speed. Matched to the idle scan's own sweep rate
# (config/idle.yaml: 3.1416 rad over 5 s ~= 0.63 rad/s) so losing a face reads
# as one continuous settling motion rather than a lunge followed by a crawl.
DEFAULT_RETURN_RATE_RAD_S = 0.6


@dataclass(frozen=True)
class FollowStatus:
    """Outcome of one arbitration tick, for logging and tests."""

    state: str  # "tracking", "returning", or "idle"
    detected: bool
    tracker_step: TrackerStep | None = None
    idle_position_rad: float | None = None


class FaceFollower:
    """Switch between tracking a visible face and idle-scanning for one."""

    STATE_TRACKING = "tracking"
    STATE_RETURNING = "returning"
    STATE_IDLE = "idle"

    def __init__(
        self,
        tracker: FaceTracker,
        idle: IdleController,
        *,
        lost_grace_s: float = 0.5,
        return_rate_rad_s: float = DEFAULT_RETURN_RATE_RAD_S,
    ) -> None:
        if (
            isinstance(lost_grace_s, bool)
            or not isinstance(lost_grace_s, (int, float))
            or not math.isfinite(lost_grace_s)
            or lost_grace_s < 0
        ):
            raise ValueError("lost_grace_s must be a finite number >= 0")
        if (
            isinstance(return_rate_rad_s, bool)
            or not isinstance(return_rate_rad_s, (int, float))
            or not math.isfinite(return_rate_rad_s)
            or return_rate_rad_s <= 0
        ):
            raise ValueError("return_rate_rad_s must be a finite number > 0")
        self._tracker = tracker
        self._idle = idle
        self._lost_grace_s = float(lost_grace_s)
        self._return_rate_rad_s = float(return_rate_rad_s)
        # Start idle so a fresh arm scans for a face until it finds one.
        self._state = self.STATE_IDLE
        self._time_since_face_s = float("inf")

    @property
    def state(self) -> str:
        return self._state

    def update(self, dt_s: float, detection: Detection | None) -> FollowStatus:
        """Advance one tick from the elapsed time and the latest detection."""
        if (
            isinstance(dt_s, bool)
            or not isinstance(dt_s, (int, float))
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            raise ValueError("dt_s must be a finite number greater than zero")

        if detection is not None:
            self._time_since_face_s = 0.0
            self._state = self.STATE_TRACKING
            step = self._tracker.step(detection)
            return FollowStatus(state=self._state, detected=True, tracker_step=step)

        # No face this tick.
        self._time_since_face_s += dt_s

        if self._state == self.STATE_TRACKING:
            if self._time_since_face_s < self._lost_grace_s:
                # Brief dropout: hold position, do not snap to idle yet.
                return FollowStatus(state=self._state, detected=False)
            # Grace exceeded: begin easing back to the idle posture.
            self._state = self.STATE_RETURNING

        if self._state == self.STATE_RETURNING:
            settled, scan_rad = self._return_step(dt_s)
            if not settled:
                return FollowStatus(
                    state=self._state,
                    detected=False,
                    idle_position_rad=scan_rad,
                )
            # Posture reached: scan onward from the held bearing, in the
            # direction the face was last travelling.
            self._state = self.STATE_IDLE
            self._idle.resume(
                from_position_rad=scan_rad,
                direction=self._tracker.last_pan_direction or 1,
            )
            return FollowStatus(
                state=self._state,
                detected=False,
                idle_position_rad=self._idle.position_rad,
            )

        # Already idle: keep scanning.
        self._idle.update(dt_s)
        return FollowStatus(
            state=self._state,
            detected=False,
            idle_position_rad=self._idle.position_rad,
        )

    def _return_step(self, dt_s: float) -> tuple[bool, float]:
        """Ease one rate-limited step toward the idle pose, holding the pan.

        Returns ``(settled, scan_joint_rad)``. Every joint but the scan joint
        moves at most ``return_rate_rad_s * dt_s``; the scan joint is
        re-commanded to where it already is, so the bearing the face was last
        seen at is preserved rather than driven back to center.
        """
        arm = self._idle.arm
        config = self._idle.config
        scan_joint = config.scan_joint
        max_step_rad = self._return_rate_rad_s * dt_s

        positions = arm.get_joint_positions()
        held_rad = positions[scan_joint]
        targets = {scan_joint: held_rad}
        settled = True
        for joint, goal_rad in config.pose_rad.items():
            if joint == scan_joint:
                continue
            remaining_rad = goal_rad - positions[joint]
            if abs(remaining_rad) <= max_step_rad:
                targets[joint] = goal_rad
            else:
                targets[joint] = positions[joint] + math.copysign(
                    max_step_rad, remaining_rad
                )
                settled = False

        arm.move_joints(targets)
        return settled, held_rad
