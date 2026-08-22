"""Stage 3 Mac/Pi CLI gates and device-boundary tests."""
from __future__ import annotations

import builtins
from collections.abc import Mapping

import numpy as np
import pytest

from examples import hand_teleoperation_stage3, pi_teleop_receiver
from butter_finger.arm import ArmBackend
from butter_finger.remote_control.client import RemoteClientStatus
from butter_finger.remote_control.receiver import ReceiverResult


def test_mac_stage3_requires_confirmation_before_link_or_camera() -> None:
    calls: list[str] = []

    result = hand_teleoperation_stage3.main(
        [],
        remote_client_factory=lambda *a, **k: calls.append("client"),
        source_factory=lambda *a, **k: calls.append("camera"),
        tracker_factory=lambda *a, **k: calls.append("tracker"),
        cv2_module=object(),
    )

    assert result == 2
    assert calls == []


def test_pi_receiver_requires_exactly_one_explicit_mode() -> None:
    calls: list[str] = []
    with pytest.raises(SystemExit) as exc:
        pi_teleop_receiver.main(
            [],
            arm_factory=lambda config: calls.append("arm"),
        )
    assert exc.value.code == 2
    assert calls == []


class FakePiArm(ArmBackend):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        self.events.append(("validate", dict(targets_rad)))

    def move_joint(self, joint, position_rad, *, duration_s=None) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(self, targets_rad, *, duration_s=None) -> None:
        self.events.append(("move", dict(targets_rad), duration_s))

    def get_joint_positions(self) -> dict[str, float]:
        return {}

    def go_home(self) -> None:
        self.events.append(("home",))

    def disconnect(self) -> None:
        self.events.append(("disconnect",))


def test_pi_hardware_initializes_home_then_idle_before_listen() -> None:
    arm = FakePiArm()

    class FakeReceiver:
        def __init__(self, *args, **kwargs) -> None:
            arm.events.append(("receiver_construct",))

        def serve_once(self, *, ready_callback):
            arm.events.append(("listen",))
            ready_callback("127.0.0.1", 8765)
            return ReceiverResult("eof")

    result = pi_teleop_receiver.main(
        ["--confirm-hardware"],
        arm_factory=lambda config: arm,
        receiver_factory=FakeReceiver,
    )

    assert result == 0
    assert arm.events[0] == ("home",)
    assert arm.events[1][0] == "move"
    assert arm.events[1][2] == 2.0
    assert arm.events[2:] == [
        ("receiver_construct",),
        ("listen",),
        ("disconnect",),
    ]
    assert set(arm.events[1][1]) == {"base", "shoulder", "elbow", "wrist"}


def test_mac_stage3_runtime_does_not_import_hardware_sdk(monkeypatch) -> None:
    from butter_finger.teleoperation import viewer

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ros_robot_controller_sdk":
            raise AssertionError("Mac Stage 3 must not import the Hiwonder SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    class Source:
        def read(self):
            return np.zeros((4, 6, 3), dtype=np.uint8)

        def close(self):
            pass

    class Tracker:
        result_fps = 0.0
        dropped_count = 0
        last_error = None

        def submit(self, *args):
            pass

        def take_latest(self, **kwargs):
            return None

        def close(self):
            pass

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def connect(self):
            pass

        @property
        def status(self):
            return RemoteClientStatus(
                True,
                "dry_run",
                "DISARMED",
                None,
                None,
                None,
                None,
                None,
            )

        def publish_hold(self, reason):
            pass

        def mark_control_alive(self):
            pass

        def request_arm(self):
            pass

        def publish_candidate(self, joints_rad, *, now_s):
            pass

        def close(self):
            self.closed = True

    client_holder = []

    def client_factory(*args, **kwargs):
        client = Client()
        client_holder.append(client)
        return client

    class CV2:
        def imshow(self, *args):
            pass

        def waitKey(self, delay):
            return ord("q")

        def destroyAllWindows(self):
            pass

    monkeypatch.setattr(viewer, "draw_hands", lambda cv, frame, obs: frame)
    monkeypatch.setattr(viewer, "draw_runtime_status", lambda *a, **k: None)
    monkeypatch.setattr(viewer, "draw_remote_status", lambda *a, **k: None)

    result = hand_teleoperation_stage3.main(
        ["--confirm-remote-hardware"],
        source_factory=lambda *a, **k: Source(),
        tracker_factory=lambda *a, **k: Tracker(),
        remote_client_factory=client_factory,
        cv2_module=CV2(),
        clock=lambda: 1.0,
    )

    assert result == 0
    assert client_holder[0].closed is True
