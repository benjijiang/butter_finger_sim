"""Pi receiver safety-state tests using sockets and a fake arm."""
from __future__ import annotations

import socket
import threading
import uuid
from collections.abc import Mapping
from dataclasses import replace

import pytest

from butter_finger.arm import ArmBackend
from butter_finger.config import JOINT_NAMES, load_arm_config
from butter_finger.remote_control.config import (
    compute_profile_sha256,
    load_remote_teleoperation_config,
)
from butter_finger.remote_control.protocol import NDJSONDecoder, encode_message
from butter_finger.remote_control.receiver import PiTeleopReceiver


class FakeArm(ArmBackend):
    def __init__(self) -> None:
        arm_config = load_arm_config()
        self.limits = arm_config.sim_limits
        self.moves: list[dict[str, float]] = []
        self.positions = dict(arm_config.poses["idle_ready"])
        self.disconnected = False

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        for joint, value in targets_rad.items():
            if joint not in self.limits or not self.limits[joint].contains(float(value)):
                raise ValueError(f"invalid {joint}")

    def move_joint(self, joint, position_rad, *, duration_s=None) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(self, targets_rad, *, duration_s=None) -> None:
        self.validate_targets(targets_rad)
        self.positions.update({joint: float(v) for joint, v in targets_rad.items()})
        self.moves.append(dict(self.positions))

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.positions = load_arm_config().home_pose

    def disconnect(self) -> None:
        self.disconnected = True


class Peer:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.sock.settimeout(1.0)
        self.decoder = NDJSONDecoder()
        self.pending: list[dict] = []

    def send(self, *messages: dict) -> None:
        self.sock.sendall(b"".join(encode_message(message) for message in messages))

    def receive(self) -> dict:
        while not self.pending:
            data = self.sock.recv(65536)
            assert data
            self.pending.extend(self.decoder.feed(data))
        return self.pending.pop(0)


def command(session: str, message_type: str, seq: int | None = None, **fields):
    result = {"v": 1, "type": message_type, "session_id": session, **fields}
    if seq is not None:
        result["seq"] = seq
    return result


def target(base: float = 0.2) -> dict[str, float]:
    return {
        "base": base,
        "shoulder": -0.3,
        "elbow": -0.5,
        "wrist": -1.0,
    }


def setup_receiver():
    config = replace(
        load_remote_teleoperation_config(),
        publish_rate_hz=50.0,
        watchdog_timeout_s=0.12,
        status_timeout_s=0.25,
        socket_timeout_s=0.05,
        max_slew_dt_s=0.05,
    )
    arm = FakeArm()
    fingerprint = compute_profile_sha256(config)
    receiver = PiTeleopReceiver(
        arm,
        config,
        fingerprint,
        arm.positions,
        mode="dry_run",
    )
    server_sock, client_sock = socket.socketpair()
    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(receiver.run_connection(server_sock)),
        daemon=True,
    )
    thread.start()
    return receiver, arm, fingerprint, Peer(client_sock), server_sock, thread, result


def handshake(peer: Peer, fingerprint: str) -> str:
    session = str(uuid.uuid4())
    peer.send(command(session, "hello", profile_sha256=fingerprint))
    ready = peer.receive()
    assert ready["type"] == "ready"
    assert ready["state"] == "DISARMED"
    return session


def stop(peer: Peer, server_sock: socket.socket, thread: threading.Thread) -> None:
    peer.sock.close()
    thread.join(timeout=1.0)
    server_sock.close()
    assert not thread.is_alive()


def test_prearm_never_moves_and_hold_arm_target_is_rate_limited() -> None:
    receiver, arm, fingerprint, peer, server_sock, thread, result = setup_receiver()
    session = handshake(peer, fingerprint)
    peer.send(command(session, "hold", 1, reason="operator_released"))
    assert peer.receive()["state"] == "DISARMED"
    assert arm.moves == []

    peer.send(command(session, "arm", 2))
    assert peer.receive()["state"] == "ARMED"
    peer.send(command(session, "target", 3, joints_rad=target(0.2)))
    status = peer.receive()

    assert status["last_applied_seq"] == 3
    assert len(arm.moves) == 1
    assert 0.0 < arm.moves[-1]["base"] <= 0.013751
    assert tuple(arm.moves[-1]) == JOINT_NAMES

    applied = status["applied_joints_rad"]
    before = len(arm.moves)
    peer.send(command(session, "target", 4, joints_rad=applied))
    assert peer.receive()["last_applied_seq"] == 4
    assert len(arm.moves) == before

    peer.send(command(session, "hold", 5, reason="operator_released"))
    held = peer.receive()
    assert held["state"] == "DISARMED"
    stop(peer, server_sock, thread)
    assert result[0].reason == "eof"
    assert receiver.state == "DISARMED"


def test_latest_target_wins_and_control_loop_calls_arm_once_per_tick() -> None:
    _, arm, fingerprint, peer, server_sock, thread, _ = setup_receiver()
    session = handshake(peer, fingerprint)
    peer.send(command(session, "hold", 1, reason="operator_released"))
    peer.receive()
    peer.send(command(session, "arm", 2))
    peer.receive()

    before = len(arm.moves)
    peer.send(
        command(session, "target", 3, joints_rad=target(0.1)),
        command(session, "target", 4, joints_rad=target(0.3)),
    )
    status = peer.receive()

    assert status["seq"] == 4
    assert len(arm.moves) == before + 1
    stop(peer, server_sock, thread)


def test_watchdog_disarms_and_ends_session_without_homing() -> None:
    receiver, arm, fingerprint, peer, server_sock, thread, result = setup_receiver()
    session = handshake(peer, fingerprint)
    peer.send(command(session, "hold", 1, reason="operator_released"))
    peer.receive()
    peer.send(command(session, "arm", 2))
    peer.receive()

    watchdog = peer.receive()

    assert watchdog["state"] == "DISARMED"
    assert watchdog["reason"] == "watchdog_timeout"
    thread.join(timeout=1.0)
    assert result[0].reason == "watchdog_timeout"
    assert result[0].fatal is True
    assert arm.moves == []
    assert receiver.state == "DISARMED"
    peer.sock.close()
    server_sock.close()


@pytest.mark.parametrize(
    "bad_message",
    [
        lambda session: command(session, "target", 1, joints_rad=target()),
        lambda session: command(session, "hold", 1, reason="ok"),
    ],
)
def test_not_armed_and_duplicate_sequence_fail_closed(bad_message) -> None:
    _, arm, fingerprint, peer, server_sock, thread, result = setup_receiver()
    session = handshake(peer, fingerprint)
    first = bad_message(session)
    peer.send(first)
    response = peer.receive()
    if first["type"] == "hold":
        assert response["type"] == "status"
        peer.send(command(session, "hold", 1, reason="duplicate"))
        response = peer.receive()
    assert response["type"] == "error"
    thread.join(timeout=1.0)
    assert result[0].fatal is True
    assert arm.moves == []
    peer.sock.close()
    server_sock.close()


def test_out_of_calibration_target_is_rejected_before_arm_call() -> None:
    _, arm, fingerprint, peer, server_sock, thread, result = setup_receiver()
    session = handshake(peer, fingerprint)
    peer.send(command(session, "hold", 1, reason="operator_released"))
    peer.receive()
    peer.send(command(session, "arm", 2))
    peer.receive()
    peer.send(command(session, "target", 3, joints_rad=target(99.0)))

    assert peer.receive()["code"] == "INVALID_TARGET"
    thread.join(timeout=1.0)
    assert result[0].fatal is True
    assert arm.moves == []
    peer.sock.close()
    server_sock.close()
