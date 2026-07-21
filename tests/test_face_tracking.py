"""Face-tracking tests: control law, clamping, config, scripted detector.

Dependency-light: these use only the config loaders and a fake in-memory arm.
No PyBullet, no OpenCV, no camera — so they run on any machine.
"""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from butter_finger import IdleController
from butter_finger.arm import ArmBackend, JointLimitError
from butter_finger.config import load_arm_config
from butter_finger.perception.attention import FaceFollower
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


# ---------------------------------------------------------------------------
# Attention layer: switch between tracking and the idle scan
# ---------------------------------------------------------------------------

def make_follower(config, tracking):
    arm = FakeArm(config, start=config.poses[tracking.start_pose])
    tracker = FaceTracker(arm, config, tracking)
    follower = FaceFollower(tracker, IdleController(arm), lost_grace_s=0.5)
    return arm, follower


def test_follower_starts_idle_and_scans(config, tracking):
    arm, follower = make_follower(config, tracking)
    assert follower.state == "idle"
    pan = tracking.pan_joint
    before = arm.get_joint_positions()[pan]
    status = None
    for _ in range(5):
        status = follower.update(0.1, None)
    assert status.state == "idle"
    assert status.detected is False
    assert arm.get_joint_positions()[pan] != before  # the base swept


def test_follower_tracks_when_face_present(config, tracking):
    arm, follower = make_follower(config, tracking)
    status = follower.update(0.1, face(IMAGE_W * 0.9, IMAGE_H / 2))
    assert status.state == "tracking"
    assert status.detected is True
    assert status.tracker_step is not None and status.tracker_step.moved


def test_follower_holds_during_grace_then_hands_off_to_idle(config, tracking):
    arm, follower = make_follower(config, tracking)
    pan = tracking.pan_joint
    follower.update(0.1, face(IMAGE_W / 2, IMAGE_H / 2))
    assert follower.state == "tracking"

    # Brief dropout (< 0.5 s grace): stays tracking, no idle snap.
    status = follower.update(0.1, None)
    assert status.state == "tracking"

    # Grace exceeded: hand off to idle; resume() returns the base to idle_ready.
    status = follower.update(0.5, None)
    assert status.state == "idle"
    assert arm.get_joint_positions()[pan] == pytest.approx(0.0)


def test_follower_reacquires_after_idle(config, tracking):
    arm, follower = make_follower(config, tracking)
    for _ in range(5):
        follower.update(0.1, None)
    assert follower.state == "idle"
    status = follower.update(0.1, face(IMAGE_W / 2, IMAGE_H / 2))
    assert status.state == "tracking"
    assert status.detected is True


def test_idle_scan_covers_more_than_the_old_range(config, tracking):
    """The widened scan sweeps past the previous +/-0.55 rad bound."""
    arm, follower = make_follower(config, tracking)
    pan = tracking.pan_joint
    reached = 0.0
    for _ in range(400):
        follower.update(0.1, None)
        reached = max(reached, abs(arm.get_joint_positions()[pan]))
    assert reached > 0.55  # old bound; proves the full +/-90 deg range is used


@pytest.mark.parametrize("dt_s", [True, "1", 0.0, -1.0, float("nan"), float("inf")])
def test_follower_rejects_invalid_dt(config, tracking, dt_s):
    _, follower = make_follower(config, tracking)
    with pytest.raises(ValueError, match="dt_s"):
        follower.update(dt_s, None)


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in returning one marked BGR frame."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.props: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> None:
        self.props[prop] = value

    def read(self):
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0:8, 0:8] = 255  # marker in the NATIVE top-left corner
        return True, frame

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def fake_cv2(monkeypatch):
    """Install a fake cv2 so WebcamSource runs without OpenCV or a camera."""
    import sys
    import types

    module = types.ModuleType("cv2")
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_BUFFERSIZE = 38
    module.COLOR_BGR2RGB = 4
    module.VideoCapture = _FakeCapture
    module.cvtColor = lambda image, code: image[:, :, ::-1].copy()
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


def test_webcam_source_defaults_to_no_rotation(fake_cv2):
    from butter_finger.perception.sources import WebcamSource

    frame = WebcamSource(0, width=640, height=480).read()

    assert frame.shape == (480, 640, 3)


def test_webcam_rotation_matches_the_camera_config_output(fake_cv2):
    """The rotated real frame must match what capture_rgb() produces.

    The physical camera is mounted a quarter turn, so an unrotated frame feeds
    the upright-face cascade sideways faces and swaps the pan/tilt image axes.
    """
    import numpy as np

    from butter_finger.config import load_camera_config
    from butter_finger.perception.sources import WebcamSource

    camera = load_camera_config()
    source = WebcamSource(
        0,
        width=camera.native_width,
        height=camera.native_height,
        rotate_clockwise_deg=camera.rotate_clockwise_deg,
    )
    frame = source.read()

    assert frame.shape == camera.output_shape
    # A 90 degree clockwise turn sends the native top-left to the top-right.
    rows, cols = np.where(frame[:, :, 0] == 255)
    assert rows.min() == 0
    assert cols.max() == camera.output_width - 1


def test_webcam_source_keeps_the_driver_queue_shallow(fake_cv2):
    """A control loop needs the newest frame, not a backlog of stale ones."""
    from butter_finger.perception.sources import WebcamSource

    source = WebcamSource(0)

    assert source._capture.props[fake_cv2.CAP_PROP_BUFFERSIZE] == 1


@pytest.mark.parametrize("degrees", [45, 1, "90", True, -30])
def test_webcam_rejects_non_quarter_turn_rotations(fake_cv2, degrees):
    from butter_finger.perception.sources import WebcamSource

    with pytest.raises(ValueError, match="multiple of 90"):
        WebcamSource(0, rotate_clockwise_deg=degrees)
