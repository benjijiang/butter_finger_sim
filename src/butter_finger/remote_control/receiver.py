"""Raspberry Pi Stage 3 receiver and authoritative safety governor."""
from __future__ import annotations

import math
import select
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable

from butter_finger.arm import ArmBackend
from butter_finger.config import JOINT_NAMES
from butter_finger.remote_control.config import RemoteTeleoperationConfig
from butter_finger.remote_control.protocol import (
    NDJSONDecoder,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_message,
)


@dataclass(frozen=True)
class ReceiverResult:
    reason: str
    fatal: bool = False


class PiTeleopReceiver:
    """Serve exactly one authenticated-by-SSH Stage 3 TCP session.

    The supplied arm must already be at the configured startup pose. This
    class never homes on shutdown: EOF, watchdog, HOLD, and failures all stop
    issuing new commands and leave the board's last target in place.
    """

    def __init__(
        self,
        arm: ArmBackend,
        config: RemoteTeleoperationConfig,
        profile_sha256: str,
        initial_joints_rad: dict[str, float],
        *,
        mode: str,
        clock: Callable[[], float] = time.monotonic,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"dry_run", "hardware"}:
            raise ValueError("mode must be dry_run or hardware")
        if set(initial_joints_rad) != set(JOINT_NAMES):
            raise ValueError("initial_joints_rad must contain all four joints")
        arm.validate_targets(initial_joints_rad)
        self.arm = arm
        self.config = config
        self.profile_sha256 = profile_sha256
        self.mode = mode
        self.clock = clock
        self.logger = logger
        self._initial_joints = {
            joint: float(initial_joints_rad[joint]) for joint in JOINT_NAMES
        }
        self.state = "DISARMED"
        self.last_applied_joints_rad = dict(self._initial_joints)
        self.last_applied_seq: int | None = None

    def serve_once(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        ready_callback: Callable[[str, int], None] | None = None,
    ) -> ReceiverResult:
        bind_host = self.config.bind_host if host is None else host
        bind_port = self.config.port if port is None else port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((bind_host, bind_port))
            server.listen(1)
            actual_host, actual_port = server.getsockname()[:2]
            if ready_callback is not None:
                ready_callback(str(actual_host), int(actual_port))
            connection, _ = server.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(self.config.socket_timeout_s)
                return self.run_connection(connection)
        finally:
            self.state = "DISARMED"
            server.close()

    def run_connection(self, connection: socket.socket) -> ReceiverResult:
        decoder = NDJSONDecoder(self.config.max_frame_bytes)
        session_id: str | None = None
        last_seq = 0
        hold_seen = False
        pending_target: dict[str, Any] | None = None
        now = self.clock()
        last_apply_s = now
        last_target_s: float | None = None
        next_tick_s = now + self.config.publish_period_s
        self.state = "DISARMED"
        self.last_applied_seq = None

        def send(message: dict[str, Any]) -> None:
            connection.sendall(
                encode_message(
                    message,
                    max_frame_bytes=self.config.max_frame_bytes,
                )
            )

        def send_status(seq: int, reason: str | None = None) -> None:
            assert session_id is not None
            message = {
                "v": PROTOCOL_VERSION,
                "type": "status",
                "session_id": session_id,
                "seq": seq,
                "state": self.state,
                "last_applied_seq": self.last_applied_seq,
                "applied_joints_rad": dict(self.last_applied_joints_rad),
                "reason": reason,
            }
            send(message)
            self._log("status", message)

        def fatal(code: str) -> ReceiverResult:
            self.state = "DISARMED"
            if session_id is not None:
                try:
                    send(
                        {
                            "v": PROTOCOL_VERSION,
                            "type": "error",
                            "session_id": session_id,
                            "code": code,
                            "fatal": True,
                        }
                    )
                except OSError:
                    pass
            self._log("fault", {"code": code})
            return ReceiverResult(code.lower(), fatal=True)

        while True:
            now = self.clock()
            if (
                self.state == "ARMED"
                and last_target_s is not None
                and now - last_target_s >= self.config.watchdog_timeout_s
            ):
                self.state = "DISARMED"
                if session_id is not None and last_seq > 0:
                    try:
                        send_status(last_seq, "watchdog_timeout")
                    except OSError:
                        pass
                return ReceiverResult("watchdog_timeout", fatal=True)

            deadlines = [next_tick_s]
            if self.state == "ARMED" and last_target_s is not None:
                deadlines.append(last_target_s + self.config.watchdog_timeout_s)
            timeout = max(0.0, min(deadlines) - now)
            try:
                readable, _, _ = select.select([connection], [], [], timeout)
            except (OSError, ValueError):
                was_armed = self.state == "ARMED"
                self.state = "DISARMED"
                return ReceiverResult(
                    "socket_error" if was_armed else "eof",
                    fatal=was_armed,
                )

            if readable:
                try:
                    data = connection.recv(65536)
                except socket.timeout:
                    data = None
                except OSError:
                    was_armed = self.state == "ARMED"
                    self.state = "DISARMED"
                    return ReceiverResult(
                        "socket_error" if was_armed else "eof",
                        fatal=was_armed,
                    )
                if data == b"":
                    self.state = "DISARMED"
                    try:
                        decoder.finish()
                    except ProtocolError:
                        return ReceiverResult("incomplete_frame", fatal=True)
                    return ReceiverResult("eof")
                if data:
                    try:
                        messages = decoder.feed(data)
                    except ProtocolError:
                        return fatal("PROTOCOL_ERROR")
                    for message in messages:
                        self._log("received", message)
                        message_type = message["type"]
                        if session_id is None:
                            if message_type != "hello":
                                return fatal("HELLO_REQUIRED")
                            session_id = message["session_id"]
                            if message["profile_sha256"] != self.profile_sha256:
                                return fatal("PROFILE_MISMATCH")
                            try:
                                send(
                                    {
                                        "v": PROTOCOL_VERSION,
                                        "type": "ready",
                                        "session_id": session_id,
                                        "mode": self.mode,
                                        "state": "DISARMED",
                                        "profile_sha256": self.profile_sha256,
                                    }
                                )
                            except OSError:
                                return ReceiverResult("socket_error", fatal=True)
                            continue

                        if message_type == "hello":
                            return fatal("DUPLICATE_HELLO")
                        if message.get("session_id") != session_id:
                            return fatal("SESSION_MISMATCH")
                        sequence = message.get("seq")
                        if not isinstance(sequence, int) or sequence <= last_seq:
                            return fatal("SEQUENCE_ERROR")
                        last_seq = sequence

                        if message_type == "hold":
                            self.state = "DISARMED"
                            pending_target = None
                            last_target_s = None
                            hold_seen = True
                            try:
                                send_status(sequence, message["reason"])
                            except OSError:
                                return ReceiverResult("eof")
                        elif message_type == "arm":
                            if self.state != "DISARMED" or not hold_seen:
                                return fatal("ARM_REJECTED")
                            self.state = "ARMED"
                            pending_target = None
                            last_target_s = self.clock()
                            try:
                                send_status(sequence, None)
                            except OSError:
                                return ReceiverResult("socket_error", fatal=True)
                        elif message_type == "target":
                            if self.state != "ARMED":
                                return fatal("NOT_ARMED")
                            targets = message["joints_rad"]
                            try:
                                self.arm.validate_targets(targets)
                            except (TypeError, ValueError):
                                return fatal("INVALID_TARGET")
                            pending_target = message
                            last_target_s = self.clock()
                        else:
                            return fatal("CLIENT_MESSAGE_REQUIRED")

            now = self.clock()
            if now < next_tick_s:
                continue
            if now - next_tick_s > self.config.publish_period_s:
                next_tick_s = now + self.config.publish_period_s
            else:
                next_tick_s += self.config.publish_period_s

            if self.state != "ARMED" or pending_target is None:
                continue
            target_message = pending_target
            pending_target = None
            dt_s = min(
                max(0.0, now - last_apply_s),
                self.config.max_slew_dt_s,
            )
            last_apply_s = now
            desired = target_message["joints_rad"]
            applied = self._slew_target(desired, dt_s)
            try:
                self.arm.validate_targets(applied)
                changed = any(
                    abs(applied[joint] - self.last_applied_joints_rad[joint])
                    > self.config.command_epsilon_rad
                    for joint in JOINT_NAMES
                )
                if changed:
                    self.arm.move_joints(applied)
            except Exception:
                return fatal("HARDWARE_ERROR")
            self.last_applied_joints_rad = applied
            self.last_applied_seq = target_message["seq"]
            try:
                send_status(target_message["seq"], None)
            except OSError:
                self.state = "DISARMED"
                return ReceiverResult("socket_error", fatal=True)

    def _slew_target(
        self,
        desired: dict[str, float],
        dt_s: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for joint in JOINT_NAMES:
            previous = self.last_applied_joints_rad[joint]
            delta = float(desired[joint]) - previous
            maximum = self.config.pi_joint_rate_limits_rad_s[joint] * dt_s
            result[joint] = previous + min(maximum, max(-maximum, delta))
            if not math.isfinite(result[joint]):
                raise ValueError("Pi slew governor produced a non-finite target")
        return result

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger(
                {
                    "event": event,
                    "monotonic_s": self.clock(),
                    "payload": payload,
                }
            )
