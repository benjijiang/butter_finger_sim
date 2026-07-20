"""FaceFollower: arbitrate between face tracking and the idle scan.

This is the "attention layer" the idle controller's docstring anticipated:
while a face is visible it drives :class:`FaceTracker`; when the face is lost
for longer than a short grace period it hands off to :class:`IdleController`,
whose slow base sweep looks for a face across the arm's full yaw range. As
soon as a face is reacquired it switches back to tracking.

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


@dataclass(frozen=True)
class FollowStatus:
    """Outcome of one arbitration tick, for logging and tests."""

    state: str  # "tracking" or "idle"
    detected: bool
    tracker_step: TrackerStep | None = None
    idle_position_rad: float | None = None


class FaceFollower:
    """Switch between tracking a visible face and idle-scanning for one."""

    STATE_TRACKING = "tracking"
    STATE_IDLE = "idle"

    def __init__(
        self,
        tracker: FaceTracker,
        idle: IdleController,
        *,
        lost_grace_s: float = 0.5,
    ) -> None:
        if (
            isinstance(lost_grace_s, bool)
            or not isinstance(lost_grace_s, (int, float))
            or not math.isfinite(lost_grace_s)
            or lost_grace_s < 0
        ):
            raise ValueError("lost_grace_s must be a finite number >= 0")
        self._tracker = tracker
        self._idle = idle
        self._lost_grace_s = float(lost_grace_s)
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
            # Grace exceeded: hand off to the idle scan from idle_ready.
            self._state = self.STATE_IDLE
            self._idle.resume()
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
