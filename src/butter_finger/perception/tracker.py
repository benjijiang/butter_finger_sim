"""FaceTracker: the visual-servo control law.

Given a face :class:`~butter_finger.perception.detection.Detection`, it nudges
three joints so the face drifts toward the image center and holds a target
apparent size:

    base  (pan)      <- horizontal pixel error
    wrist (tilt)     <- vertical pixel error
    shoulder (dist)  <- apparent face size vs. target (bigger => closer)

(The actual joint names come from ``TrackingConfig``.) The law is pure math on
top of the radians-only ``ArmBackend`` API — no PyBullet, no OpenCV — so it is
fully unit-testable and never talks to hardware directly.

Two safety-relevant behaviors:
  * Every commanded target is clamped into the joint's simulation limits, so
    the loop never triggers ``JointLimitError`` at a range boundary.
  * Each step moves a joint by at most ``max_step_rad``, so the arm slews
    smoothly instead of snapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from butter_finger.arm import ArmBackend
from butter_finger.config import ArmConfig
from butter_finger.perception.config import TrackingConfig
from butter_finger.perception.detection import Detection


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _deadband(value: float, band: float) -> float:
    return 0.0 if abs(value) < band else value


@dataclass(frozen=True)
class TrackerStep:
    """Result of one control step, for logging and tests."""

    detected: bool
    targets: dict[str, float] = field(default_factory=dict)
    error_x: float = 0.0
    error_y: float = 0.0
    error_size: float = 0.0
    moved: bool = False


class FaceTracker:
    """Proportional pan/tilt/distance controller over an ``ArmBackend``."""

    def __init__(self, arm: ArmBackend, config: ArmConfig, tracking: TrackingConfig) -> None:
        self._arm = arm
        self._config = config
        self._tracking = tracking

    @property
    def start_pose(self) -> dict[str, float]:
        """Clamped joint targets for the configured tracking start pose."""
        pose = self._config.poses.get(self._tracking.start_pose)
        if pose is None:
            raise KeyError(
                f"tracking start_pose {self._tracking.start_pose!r} is not "
                f"defined in poses.yaml (have {sorted(self._config.poses)})"
            )
        return {joint: self._clamp_to_limits(joint, angle) for joint, angle in pose.items()}

    def go_to_start(self) -> None:
        """Command the arm to the tracking start pose (caller steps the sim)."""
        self._arm.move_joints(self.start_pose)

    def step(self, detection: Detection | None) -> TrackerStep:
        """Command one control update from a detection (or hold on None)."""
        if detection is None:
            return TrackerStep(detected=False)

        trk = self._tracking
        half_w = detection.image_width / 2.0
        half_h = detection.image_height / 2.0
        error_x = _deadband((detection.cx - half_w) / half_w, trk.deadband_xy)
        error_y = _deadband((detection.cy - half_h) / half_h, trk.deadband_xy)
        error_size = _deadband(
            (detection.width_fraction - trk.target_face_fraction) / trk.target_face_fraction,
            trk.deadband_distance,
        )

        positions = self._arm.get_joint_positions()
        commands = (
            (trk.pan_joint, trk.sign_pan, trk.gain_pan, error_x),
            (trk.tilt_joint, trk.sign_tilt, trk.gain_tilt, error_y),
            (trk.distance_joint, trk.sign_distance, trk.gain_distance, error_size),
        )

        targets: dict[str, float] = {}
        moved = False
        for joint, sign, gain, error in commands:
            delta = _clamp(sign * gain * error, -trk.max_step_rad, trk.max_step_rad)
            if delta != 0.0:
                moved = True
            targets[joint] = self._clamp_to_limits(joint, positions[joint] + delta)

        if targets:
            self._arm.move_joints(targets)

        return TrackerStep(
            detected=True,
            targets=targets,
            error_x=error_x,
            error_y=error_y,
            error_size=error_size,
            moved=moved,
        )

    def _clamp_to_limits(self, joint: str, value: float) -> float:
        limits = self._config.sim_limits[joint]
        return _clamp(value, limits.lower_rad, limits.upper_rad)
