"""Mac-side Stage 3 TCP client and hand-control arming coordinator."""
from __future__ import annotations

import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from butter_finger.config import JOINT_NAMES
from butter_finger.remote_control.config import RemoteTeleoperationConfig
from butter_finger.remote_control.protocol import (
    NDJSONDecoder,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_message,
)
from butter_finger.teleoperation.types import DryRunStep, TrackingState


class RemoteControlError(RuntimeError):
    """The Stage 3 transport or server entered a terminal failure."""


@dataclass(frozen=True)
class RemoteClientStatus:
    connected: bool
    mode: str | None
    server_state: str
    last_sent_seq: int | None
    last_applied_seq: int | None
    applied_joints_rad: dict[str, float] | None
    reason: str | None
    failure: str | None


@dataclass(frozen=True)
class Stage3ControlStatus:
    link: RemoteClientStatus
    release_frames: int
    release_required: bool
    arm_requested: bool
    local_reason: str | None


class RemoteControlClient:
    """Persistent, non-reconnecting TCP client with a single latest slot."""

    def __init__(
        self,
        config: RemoteTeleoperationConfig,
        profile_sha256: str,
        initial_joints_rad: dict[str, float],
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        socket_factory: Callable[..., socket.socket] = socket.create_connection,
    ) -> None:
        if set(initial_joints_rad) != set(JOINT_NAMES):
            raise ValueError("initial_joints_rad must contain all four joints")
        self.config = config
        self.profile_sha256 = profile_sha256
        self.host = host
        self.port = config.port if port is None else port
        self.clock = clock
        self.socket_factory = socket_factory
        self.session_id = str(uuid.uuid4())

        self._socket: socket.socket | None = None
        self._decoder = NDJSONDecoder(config.max_frame_bytes)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._sender: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._desired: tuple[str, dict[str, Any], bool] | None = None
        self._desired_generation = 0
        self._sequence = 0
        self._connected = False
        self._mode: str | None = None
        self._server_state = "DISCONNECTED"
        self._last_sent_seq: int | None = None
        self._last_applied_seq: int | None = None
        self._applied_joints: dict[str, float] | None = None
        self._reason: str | None = None
        self._failure: str | None = None
        self._last_status_s: float | None = None
        self._last_status_seq: int | None = None
        self._awaiting_status_since: float | None = None
        self._last_control_alive_s: float | None = None
        self._shaped_target = {
            joint: float(initial_joints_rad[joint]) for joint in JOINT_NAMES
        }
        self._last_shape_s: float | None = None

    def connect(self) -> None:
        if self._socket is not None:
            raise RemoteControlError("client is already connected")
        try:
            sock = self.socket_factory(
                (self.host, self.port),
                timeout=self.config.connect_timeout_s,
            )
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                # Test transports such as socketpair are not TCP sockets.
                # Production AF_INET connections must support TCP_NODELAY.
                if self.socket_factory is socket.create_connection:
                    raise
            sock.settimeout(self.config.socket_timeout_s)
            self._socket = sock
            self._send_raw(
                {
                    "v": PROTOCOL_VERSION,
                    "type": "hello",
                    "session_id": self.session_id,
                    "profile_sha256": self.profile_sha256,
                }
            )
            ready, extras = self._wait_for_ready(sock)
            if ready["session_id"] != self.session_id:
                raise RemoteControlError("server READY used the wrong session")
            if ready["profile_sha256"] != self.profile_sha256:
                raise RemoteControlError("Mac/Pi Stage 3 profile mismatch")
            with self._lock:
                self._connected = True
                self._mode = ready["mode"]
                self._server_state = ready["state"]
                self._last_status_s = self.clock()
            for message in extras:
                self._handle_server_message(message)
            self._sender = threading.Thread(
                target=self._sender_loop,
                name="stage3-sender",
                daemon=True,
            )
            self._reader = threading.Thread(
                target=self._reader_loop,
                name="stage3-reader",
                daemon=True,
            )
            self._sender.start()
            self._reader.start()
        except Exception as exc:
            self._close_socket()
            if isinstance(exc, RemoteControlError):
                raise
            raise RemoteControlError(f"could not connect to Pi receiver: {exc}") from exc

    @property
    def status(self) -> RemoteClientStatus:
        with self._lock:
            return RemoteClientStatus(
                connected=self._connected,
                mode=self._mode,
                server_state=self._server_state,
                last_sent_seq=self._last_sent_seq,
                last_applied_seq=self._last_applied_seq,
                applied_joints_rad=(
                    None if self._applied_joints is None else dict(self._applied_joints)
                ),
                reason=self._reason,
                failure=self._failure,
            )

    def publish_hold(self, reason: str) -> None:
        self.mark_control_alive()
        self._set_desired("hold", {"reason": reason}, one_shot=False)

    def request_arm(self) -> None:
        self.mark_control_alive()
        if self.status.server_state != "DISARMED":
            return
        self._set_desired("arm", {}, one_shot=True)

    def publish_candidate(
        self,
        joints_rad: dict[str, float],
        *,
        now_s: float,
    ) -> dict[str, float]:
        if set(joints_rad) != set(JOINT_NAMES):
            raise ValueError("candidate must contain all four joints")
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        self.mark_control_alive()
        if self._last_shape_s is None:
            dt_s = 0.0
        else:
            dt_s = min(
                max(0.0, now_s - self._last_shape_s),
                self.config.max_slew_dt_s,
            )
        self._last_shape_s = now_s
        shaped: dict[str, float] = {}
        for joint in JOINT_NAMES:
            desired = joints_rad[joint]
            if (
                isinstance(desired, bool)
                or not isinstance(desired, (int, float))
                or not math.isfinite(float(desired))
            ):
                raise ValueError(f"invalid Mac candidate for {joint}")
            previous = self._shaped_target[joint]
            maximum = self.config.mac_joint_rate_limits_rad_s[joint] * dt_s
            delta = float(desired) - previous
            shaped[joint] = previous + min(maximum, max(-maximum, delta))
        self._shaped_target = shaped
        self._set_desired(
            "target",
            {"joints_rad": dict(shaped)},
            one_shot=False,
        )
        return dict(shaped)

    def mark_control_alive(self) -> None:
        """Refresh the Mac camera/control-loop watchdog from the UI thread."""
        with self._lock:
            self._last_control_alive_s = self.clock()

    def close(self, *, reason: str = "client_shutdown") -> None:
        should_hold = self._socket is not None and self.status.connected
        # Serialize shutdown behind any in-flight target, then make HOLD the
        # final possible frame. This prevents a captured sender-loop target
        # from overtaking the shutdown command.
        with self._send_lock:
            self._stop.set()
            with self._lock:
                self._desired = None
            if should_hold and self._socket is not None:
                try:
                    with self._lock:
                        self._sequence += 1
                        sequence = self._sequence
                    packet = encode_message(
                        {
                            "v": PROTOCOL_VERSION,
                            "type": "hold",
                            "session_id": self.session_id,
                            "seq": sequence,
                            "reason": reason,
                        },
                        max_frame_bytes=self.config.max_frame_bytes,
                    )
                    self._socket.sendall(packet)
                    with self._lock:
                        self._last_sent_seq = sequence
                except (OSError, ProtocolError):
                    pass
        self._close_socket()
        current = threading.current_thread()
        for thread in (self._sender, self._reader):
            if thread is not None and thread is not current:
                thread.join(timeout=0.5)
        with self._lock:
            self._connected = False
            if self._server_state != "FAULT":
                self._server_state = "DISCONNECTED"

    def _set_desired(
        self,
        message_type: str,
        fields: dict[str, Any],
        *,
        one_shot: bool,
    ) -> None:
        with self._lock:
            if self._failure is not None:
                raise RemoteControlError(self._failure)
            if not self._connected:
                raise RemoteControlError("client is not connected")
            self._desired_generation += 1
            self._desired = (message_type, dict(fields), one_shot)

    def _wait_for_ready(
        self, sock: socket.socket
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        deadline = self.clock() + self.config.connect_timeout_s
        while self.clock() < deadline:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            if not data:
                raise RemoteControlError("Pi receiver closed before READY")
            messages = self._decoder.feed(data)
            for index, message in enumerate(messages):
                if message["type"] == "error":
                    raise RemoteControlError(f"Pi rejected HELLO: {message['code']}")
                if message["type"] != "ready":
                    raise RemoteControlError("Pi sent a non-READY handshake message")
                return message, messages[index + 1 :]
        raise RemoteControlError("timed out waiting for Pi READY")

    def _sender_loop(self) -> None:
        period = self.config.publish_period_s
        next_send = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            delay = next_send - now
            if delay > 0:
                self._stop.wait(min(delay, period))
                continue
            # Never catch up missed publishes: old real-time commands are not
            # useful, and a burst would violate the fixed 20 Hz contract.
            next_send = now + period
            with self._lock:
                desired = self._desired
                generation = self._desired_generation
                awaiting_status_since = self._awaiting_status_since
                last_control_alive = self._last_control_alive_s
            if desired is None:
                continue
            if (
                awaiting_status_since is not None
                and now - awaiting_status_since > self.config.status_timeout_s
            ):
                self._fail("Pi status timeout")
                return
            message_type, fields, one_shot = desired
            if (
                message_type in {"arm", "target"}
                and (
                    last_control_alive is None
                    or now - last_control_alive > self.config.mac_control_timeout_s
                )
            ):
                message_type = "hold"
                fields = {"reason": "mac_control_timeout"}
                one_shot = False
                with self._lock:
                    self._desired_generation += 1
                    generation = self._desired_generation
                    self._desired = (message_type, dict(fields), one_shot)
            try:
                self._send_command(message_type, fields)
            except (OSError, ProtocolError) as exc:
                self._fail(f"send failed: {exc}")
                return
            if one_shot:
                with self._lock:
                    if generation == self._desired_generation:
                        self._desired = None

    def _reader_loop(self) -> None:
        sock = self._socket
        assert sock is not None
        while not self._stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._fail(f"receive failed: {exc}")
                return
            if not data:
                if not self._stop.is_set():
                    self._fail("Pi receiver closed the connection")
                return
            try:
                messages = self._decoder.feed(data)
                for message in messages:
                    self._handle_server_message(message)
            except (ProtocolError, RemoteControlError) as exc:
                self._fail(f"invalid Pi response: {exc}")
                return

    def _handle_server_message(self, message: dict[str, Any]) -> None:
        if message["session_id"] != self.session_id:
            raise RemoteControlError("Pi response session mismatch")
        message_type = message["type"]
        if message_type == "status":
            with self._lock:
                self._server_state = message["state"]
                self._last_applied_seq = message["last_applied_seq"]
                self._applied_joints = message["applied_joints_rad"]
                self._reason = message["reason"]
                self._last_status_s = self.clock()
                self._last_status_seq = message["seq"]
                if (
                    self._last_sent_seq is None
                    or self._last_status_seq >= self._last_sent_seq
                ):
                    self._awaiting_status_since = None
        elif message_type == "error":
            raise RemoteControlError(f"Pi error: {message['code']}")
        else:
            raise RemoteControlError(f"unexpected Pi message {message_type!r}")

    def _send_command(self, message_type: str, fields: dict[str, Any]) -> int:
        sock = self._socket
        if sock is None:
            raise OSError("socket is closed")
        # Sequence allocation and the corresponding write share one lock so
        # the sender thread and best-effort shutdown HOLD cannot reorder frames.
        with self._send_lock:
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
            message = {
                "v": PROTOCOL_VERSION,
                "type": message_type,
                "session_id": self.session_id,
                "seq": sequence,
                **fields,
            }
            packet = encode_message(
                message,
                max_frame_bytes=self.config.max_frame_bytes,
            )
            sock.sendall(packet)
            with self._lock:
                self._last_sent_seq = sequence
                if self._awaiting_status_since is None:
                    self._awaiting_status_since = self.clock()
            return sequence

    def _send_raw(self, message: dict[str, Any]) -> None:
        sock = self._socket
        if sock is None:
            raise OSError("socket is closed")
        packet = encode_message(message, max_frame_bytes=self.config.max_frame_bytes)
        with self._send_lock:
            sock.sendall(packet)

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._failure = reason
            self._reason = reason
            self._server_state = "FAULT"
            self._connected = False
            self._desired = None
        self._stop.set()
        self._close_socket()

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


class Stage3Coordinator:
    """Translate Stage 2 safety states into HOLD/ARM/TARGET wire intent."""

    def __init__(self, client: RemoteControlClient, clutch_config: Any) -> None:
        self.client = client
        self._release_ratio = float(clutch_config.release_ratio)
        self._debounce_frames = int(clutch_config.debounce_frames)
        self._release_frames = 0
        self._release_latched = False
        self._arm_requested = False
        self._local_reason: str | None = "release_required"

    @property
    def status(self) -> Stage3ControlStatus:
        link = self.client.status
        if self._arm_requested and link.server_state == "ARMED":
            self._arm_requested = False
        return Stage3ControlStatus(
            link=link,
            release_frames=self._release_frames,
            release_required=not self._release_latched and link.server_state != "ARMED",
            arm_requested=self._arm_requested,
            local_reason=self._local_reason,
        )

    def force_hold(self, reason: str) -> None:
        """Fail closed and require a new explicit open-hand release latch."""
        self.client.publish_hold(reason)
        self._release_frames = 0
        self._release_latched = False
        self._arm_requested = False
        self._local_reason = reason

    def process_step(
        self,
        step: DryRunStep,
        *,
        fresh_result: bool,
        now_s: float,
    ) -> None:
        self.client.mark_control_alive()
        link = self.client.status
        if link.failure is not None:
            raise RemoteControlError(link.failure)

        update = step.target_update
        explicitly_open = (
            fresh_result
            and update.state is TrackingState.HOVER
            and update.pinch_ratio is not None
            and update.pinch_ratio > self._release_ratio
        )
        if explicitly_open:
            self._release_frames += 1
            if self._release_frames >= self._debounce_frames:
                self._release_latched = True
        elif fresh_result:
            self._release_frames = 0

        if step.hold_reason == "no_fresh_result" and link.server_state == "ARMED":
            # The asynchronous landmarker commonly has no new result between
            # adjacent display frames. Keep the latest target heartbeat until
            # DryRunTeleoperationController declares the configured stale gap.
            return

        if step.hold_reason == "clutch_reset":
            if self._release_latched and link.server_state == "DISARMED":
                self.client.request_arm()
                self._arm_requested = True
                self._release_latched = False
                self._release_frames = 0
                self._local_reason = "waiting_for_arm_ack"
            elif link.server_state != "ARMED":
                self.client.publish_hold("release_required")
                self._local_reason = "release_required"
            return

        commandable = (
            fresh_result
            and step.hold_reason is None
            and update.state is TrackingState.CLUTCHED
            and update.motion_eligible
        )
        if commandable and link.server_state == "ARMED":
            self.client.publish_candidate(step.output_joints_rad, now_s=now_s)
            self._local_reason = None
            self._arm_requested = False
            return

        if commandable and self._arm_requested:
            self._local_reason = "waiting_for_arm_ack"
            return

        if explicitly_open and self._release_latched:
            reason = "operator_released"
        else:
            reason = step.hold_reason or "not_commandable"
        if reason in {"operator_released", "tracking_hover"}:
            # Preserve the open-hand counter/latch while HOVER frames are
            # accumulating and while the user transitions from open to pinch.
            self.client.publish_hold(reason)
            self._arm_requested = False
            self._local_reason = reason
        else:
            self.force_hold(reason)
