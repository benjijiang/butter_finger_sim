"""Real localhost TCP integration for the Stage 3 client and Pi receiver."""
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Mapping

from butter_finger.arm import ArmBackend
from butter_finger.config import load_arm_config
from butter_finger.remote_control.client import RemoteControlClient
from butter_finger.remote_control.config import (
    compute_profile_sha256,
    load_remote_teleoperation_config,
)
from butter_finger.remote_control.receiver import PiTeleopReceiver


class LoopbackArm(ArmBackend):
    def __init__(self) -> None:
        config = load_arm_config()
        self.limits = config.sim_limits
        self.positions = dict(config.poses["idle_ready"])
        self.moves: list[dict[str, float]] = []

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        for joint, value in targets_rad.items():
            if joint not in self.limits or not self.limits[joint].contains(float(value)):
                raise ValueError("outside calibrated range")

    def move_joint(self, joint, position_rad, *, duration_s=None) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(self, targets_rad, *, duration_s=None) -> None:
        self.validate_targets(targets_rad)
        self.positions.update({joint: float(value) for joint, value in targets_rad.items()})
        self.moves.append(dict(self.positions))

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.positions = load_arm_config().home_pose

    def disconnect(self) -> None:
        pass


def wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_client_receiver_handshake_arm_target_hold_and_eof() -> None:
    config = load_remote_teleoperation_config()
    arm = LoopbackArm()
    fingerprint = compute_profile_sha256(config)
    receiver = PiTeleopReceiver(
        arm,
        config,
        fingerprint,
        arm.positions,
        mode="dry_run",
    )
    results = []
    server_sock, client_sock = socket.socketpair()
    thread = threading.Thread(
        target=lambda: results.append(receiver.run_connection(server_sock)),
        daemon=True,
    )
    thread.start()
    client = RemoteControlClient(
        config,
        fingerprint,
        arm.positions,
        socket_factory=lambda *args, **kwargs: client_sock,
    )

    client.connect()
    assert client.status.server_state == "DISARMED"
    # Loading MediaPipe/camera may take longer than the status timeout. No
    # command is outstanding yet, so READY must remain a valid idle link.
    time.sleep(config.status_timeout_s + 0.05)
    assert client.status.failure is None
    client.publish_hold("operator_released")
    wait_until(lambda: client.status.reason == "operator_released")
    client.request_arm()
    wait_until(lambda: client.status.server_state == "ARMED")

    desired = dict(arm.positions)
    desired["base"] = 0.2
    client.publish_candidate(desired, now_s=1.0)
    client.publish_candidate(desired, now_s=1.1)
    wait_until(lambda: bool(arm.moves))

    assert 0.0 < arm.moves[-1]["base"] <= 0.027501
    assert set(arm.moves[-1]) == {"base", "shoulder", "elbow", "wrist"}
    # Stop refreshing the Mac control loop while leaving TCP alive. The
    # sender thread must replace its TARGET heartbeat with HOLD by itself.
    wait_until(
        lambda: client.status.server_state == "DISARMED"
        and client.status.reason == "mac_control_timeout",
        timeout=0.5,
    )
    move_count = len(arm.moves)
    time.sleep(0.08)
    assert len(arm.moves) == move_count

    client.close()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert results[0].reason == "eof"
    assert receiver.state == "DISARMED"
    server_sock.close()
