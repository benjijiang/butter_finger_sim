"""Face-tracking tests: control law, clamping, config, scripted detector.

Dependency-light: these use only the config loaders and a fake in-memory arm.
No PyBullet, no OpenCV, no camera — so they run on any machine.
"""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from butter_finger.arm import ArmBackend, JointLimitError
from butter_finger.config import load_arm_config
from butter_finger.perception.config import load_tracking_config
from butter_finger.perception.detection import Detection, ScriptedFaceDetector
from butter_finger.perception.tracker import FaceTracker

IMAGE_W = 480
IMAGE_H = 640


class FakeArm(ArmBackend):
    """In-memory arm that raises on out-of-range moves, like PyBulletArm."""

    def __init__(self, config, start=None):
        self._config = config
        self._pos = {j: 0.0 for j in config.joint_order}
        if start:
            self._pos.update(start)
        self.move_calls = 0

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        for joint, position in targets_rad.items():
            limits = self._config.sim_limits[joint]
            if not limits.contains(position):
                raise JointLimitError(f"{joint}={position} outside {limits}")

    def move_joint(self, joint, position_rad, *, duration_s=None):
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(self, targets_rad: Mapping[str, float], *, duration_s=None) -> None:
        self.validate_targets(targets_rad)
        self._pos.update(targets_rad)
        self.move_calls += 1

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._pos)

    def go_home(self) -> None:
        self._pos = {j: 0.0 for j in self._config.joint_order}

    def disconnect(self) -> None:
        pass


@pytest.fixture(scope="module")
def config():
    return load_arm_config()


@pytest.fixture(scope="module")
def tracking():
    return load_tracking_config()


def face(cx, cy, w=96, h=96):
    return Detection(cx=cx, cy=cy, w=w, h=h, image_width=IMAGE_W, image_height=IMAGE_H)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_tracking_config_loads_and_joints_valid(config, tracking):
    for joint in (tracking.pan_joint, tracking.tilt_joint, tracking.distance_joint):
        assert joint in config.joint_order
    assert tracking.start_pose in config.poses
    assert tracking.max_step_rad > 0
    assert 0.0 < tracking.target_face_fraction < 1.0


def test_start_pose_within_limits(config, tracking):
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    for joint, angle in tracker.start_pose.items():
        assert config.sim_limits[joint].contains(angle)


# ---------------------------------------------------------------------------
# Scripted detector
# ---------------------------------------------------------------------------

def test_scripted_detector_sequence_then_holds():
    a, b = face(10, 10), face(20, 20)
    det = ScriptedFaceDetector([a, None, b])
    assert det.detect(None) is a
    assert det.detect(None) is None
    assert det.detect(None) is b
    assert det.detect(None) is b  # exhausted -> holds last


# ---------------------------------------------------------------------------
# Control law
# ---------------------------------------------------------------------------

def test_no_detection_holds(config, tracking):
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    result = tracker.step(None)
    assert result.detected is False
    assert result.moved is False
    assert arm.move_calls == 0


def test_centered_face_is_within_deadband(config, tracking):
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    target_w = tracking.target_face_fraction * IMAGE_W
    result = tracker.step(face(IMAGE_W / 2, IMAGE_H / 2, w=target_w, h=target_w))
    assert result.detected is True
    assert result.error_x == 0.0 and result.error_y == 0.0
    assert result.moved is False


def test_pan_and_tilt_directions(config, tracking):
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    start = arm.get_joint_positions()
    result = tracker.step(face(IMAGE_W * 0.9, IMAGE_H * 0.9))
    pan, tilt = tracking.pan_joint, tracking.tilt_joint
    assert (result.targets[pan] - start[pan]) * tracking.sign_pan > 0
    assert (result.targets[tilt] - start[tilt]) * tracking.sign_tilt > 0


def test_distance_responds_to_face_size(config, tracking):
    # Start from the tracking pose so the distance joint has headroom in both
    # directions (shoulder sits at 0.0 == its upper limit otherwise).
    arm = FakeArm(config, start=config.poses[tracking.start_pose])
    tracker = FaceTracker(arm, config, tracking)
    dist = tracking.distance_joint
    start = arm.get_joint_positions()
    big = tracking.target_face_fraction * IMAGE_W * 2.0
    result = tracker.step(face(IMAGE_W / 2, IMAGE_H / 2, w=big, h=big))
    assert result.error_size > 0
    assert (result.targets[dist] - start[dist]) * tracking.sign_distance > 0


def test_step_never_exceeds_max_step(config, tracking):
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    start = arm.get_joint_positions()
    result = tracker.step(face(IMAGE_W, IMAGE_H, w=IMAGE_W, h=IMAGE_H))  # saturated
    for joint, target in result.targets.items():
        assert abs(target - start[joint]) <= tracking.max_step_rad + 1e-9


def test_targets_are_clamped_and_never_raise(config, tracking):
    pan = tracking.pan_joint
    limits = config.sim_limits[pan]
    # Start pan at its upper limit and keep pushing the same way.
    arm = FakeArm(config, start={pan: limits.upper_rad})
    tracker = FaceTracker(arm, config, tracking)
    offset = IMAGE_W if tracking.sign_pan < 0 else 0
    for _ in range(50):
        tracker.step(face(offset, IMAGE_H / 2))  # must not raise JointLimitError
        pos = arm.get_joint_positions()[pan]
        assert limits.lower_rad - 1e-9 <= pos <= limits.upper_rad + 1e-9


def test_pan_error_converges(config, tracking):
    """Closed loop with a simple pinhole model drives horizontal error to ~0."""
    arm = FakeArm(config)
    tracker = FaceTracker(arm, config, tracking)
    pan = tracking.pan_joint

    bearing = 0.4  # radians the face sits off the current heading (within limits)
    model_sign = 1.0 if tracking.sign_pan > 0 else -1.0
    scale = 200.0  # pixels per radian

    def detection_from_state():
        base = arm.get_joint_positions()[pan]
        cx = IMAGE_W / 2 + model_sign * scale * (bearing - base)
        cx = max(0.0, min(float(IMAGE_W), cx))
        return face(cx, IMAGE_H / 2)

    errors = []
    for _ in range(200):
        result = tracker.step(detection_from_state())
        errors.append(abs(result.error_x))

    assert errors[-1] < errors[0]
    assert errors[-1] < 0.1  # converged near center
