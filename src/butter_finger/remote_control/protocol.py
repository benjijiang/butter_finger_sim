"""Strict protocol-v1 NDJSON framing for Stage 3 remote control."""
from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from butter_finger.config import JOINT_NAMES

PROTOCOL_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolError(ValueError):
    """A wire frame or message violates the Stage 3 protocol."""


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant {value!r} is forbidden")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _exact_keys(message: dict[str, Any], expected: set[str]) -> None:
    actual = set(message)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolError(f"invalid fields; missing={missing}, extra={extra}")


def _session(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("session_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("session_id must be a valid UUID") from exc
    if str(parsed) != value.lower():
        raise ProtocolError("session_id must use canonical UUID text")
    return str(parsed)


def _sequence(value: Any, label: str = "seq") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{label} must be a positive integer")
    return value


def _reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ProtocolError("reason must be null or a 1-128 character string")
    return value


def _joints(value: Any, *, nullable: bool = False) -> dict[str, float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, dict) or set(value) != set(JOINT_NAMES):
        raise ProtocolError(f"joints_rad must contain exactly {list(JOINT_NAMES)}")
    result: dict[str, float] = {}
    for joint in JOINT_NAMES:
        raw = value[joint]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProtocolError(f"joints_rad.{joint} must be a number")
        number = float(raw)
        if not math.isfinite(number):
            raise ProtocolError(f"joints_rad.{joint} must be finite")
        result[joint] = number
    return result


def validate_message(message: Any) -> dict[str, Any]:
    """Validate and normalize one client or server protocol message."""
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    version = message.get("v")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PROTOCOL_VERSION
    ):
        raise ProtocolError(f"protocol version must be {PROTOCOL_VERSION}")
    message_type = message.get("type")
    if not isinstance(message_type, str):
        raise ProtocolError("type must be a string")

    normalized = dict(message)
    if message_type == "hello":
        _exact_keys(message, {"v", "type", "session_id", "profile_sha256"})
        normalized["session_id"] = _session(message["session_id"])
        fingerprint = message["profile_sha256"]
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ProtocolError("profile_sha256 must be 64 lowercase hex characters")
    elif message_type == "hold":
        _exact_keys(message, {"v", "type", "session_id", "seq", "reason"})
        normalized["session_id"] = _session(message["session_id"])
        normalized["seq"] = _sequence(message["seq"])
        normalized["reason"] = _reason(message["reason"])
        if normalized["reason"] is None:
            raise ProtocolError("hold reason must not be null")
    elif message_type == "arm":
        _exact_keys(message, {"v", "type", "session_id", "seq"})
        normalized["session_id"] = _session(message["session_id"])
        normalized["seq"] = _sequence(message["seq"])
    elif message_type == "target":
        _exact_keys(
            message,
            {"v", "type", "session_id", "seq", "joints_rad"},
        )
        normalized["session_id"] = _session(message["session_id"])
        normalized["seq"] = _sequence(message["seq"])
        normalized["joints_rad"] = _joints(message["joints_rad"])
    elif message_type == "ready":
        _exact_keys(
            message,
            {"v", "type", "session_id", "mode", "state", "profile_sha256"},
        )
        normalized["session_id"] = _session(message["session_id"])
        if message["mode"] not in {"dry_run", "hardware"}:
            raise ProtocolError("ready mode must be dry_run or hardware")
        if message["state"] != "DISARMED":
            raise ProtocolError("ready state must be DISARMED")
        fingerprint = message["profile_sha256"]
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ProtocolError("profile_sha256 must be 64 lowercase hex characters")
    elif message_type == "status":
        _exact_keys(
            message,
            {
                "v",
                "type",
                "session_id",
                "seq",
                "state",
                "last_applied_seq",
                "applied_joints_rad",
                "reason",
            },
        )
        normalized["session_id"] = _session(message["session_id"])
        normalized["seq"] = _sequence(message["seq"])
        if message["state"] not in {"DISARMED", "ARMED", "FAULT"}:
            raise ProtocolError("invalid server state")
        last_applied = message["last_applied_seq"]
        if last_applied is not None:
            normalized["last_applied_seq"] = _sequence(
                last_applied, "last_applied_seq"
            )
            if normalized["last_applied_seq"] > normalized["seq"]:
                raise ProtocolError("last_applied_seq must not exceed status seq")
        normalized["applied_joints_rad"] = _joints(
            message["applied_joints_rad"], nullable=True
        )
        normalized["reason"] = _reason(message["reason"])
    elif message_type == "error":
        _exact_keys(
            message,
            {"v", "type", "session_id", "code", "fatal"},
        )
        normalized["session_id"] = _session(message["session_id"])
        code = message["code"]
        if not isinstance(code, str) or not code or len(code) > 64:
            raise ProtocolError("error code must be a 1-64 character string")
        if not isinstance(message["fatal"], bool):
            raise ProtocolError("error fatal must be a boolean")
    else:
        raise ProtocolError(f"unknown message type {message_type!r}")
    return normalized


def decode_line(line: bytes, *, max_frame_bytes: int = 4096) -> dict[str, Any]:
    if not line or len(line) > max_frame_bytes:
        raise ProtocolError("empty or oversized protocol frame")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("protocol frame is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_no_duplicates,
        )
    except ProtocolError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    return validate_message(value)


def encode_message(message: dict[str, Any], *, max_frame_bytes: int = 4096) -> bytes:
    normalized = validate_message(message)
    try:
        payload = json.dumps(
            normalized,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not JSON encodable: {exc}") from exc
    if len(payload) > max_frame_bytes:
        raise ProtocolError("encoded protocol frame exceeds max_frame_bytes")
    return payload + b"\n"


class NDJSONDecoder:
    """Incrementally split a TCP byte stream into strict protocol messages."""

    def __init__(self, max_frame_bytes: int = 4096) -> None:
        if isinstance(max_frame_bytes, bool) or max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.max_frame_bytes = int(max_frame_bytes)
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        if not isinstance(data, bytes):
            raise TypeError("NDJSONDecoder.feed requires bytes")
        self._buffer.extend(data)
        messages: list[dict[str, Any]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > self.max_frame_bytes:
                    raise ProtocolError("protocol frame exceeds max_frame_bytes")
                break
            if newline > self.max_frame_bytes:
                raise ProtocolError("protocol frame exceeds max_frame_bytes")
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            messages.append(
                decode_line(line, max_frame_bytes=self.max_frame_bytes)
            )
        return messages

    def finish(self) -> None:
        if self._buffer:
            raise ProtocolError("connection closed with an incomplete frame")
