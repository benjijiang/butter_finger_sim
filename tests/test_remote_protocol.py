"""Strict Stage 3 configuration and NDJSON protocol tests."""
from __future__ import annotations

import json
import uuid
from dataclasses import replace

import pytest

from butter_finger.remote_control.config import (
    compute_profile_sha256,
    load_remote_teleoperation_config,
)
from butter_finger.remote_control.protocol import (
    NDJSONDecoder,
    ProtocolError,
    decode_line,
    encode_message,
)

SESSION = str(uuid.UUID(int=1))
PROFILE = "a" * 64


def message(message_type: str, **fields):
    return {
        "v": 1,
        "type": message_type,
        "session_id": SESSION,
        **fields,
    }


def joints(**overrides):
    values = {"base": 0.0, "shoulder": -0.3, "elbow": -0.5, "wrist": -1.0}
    values.update(overrides)
    return values


def test_default_config_and_fingerprint_are_stable() -> None:
    config = load_remote_teleoperation_config()
    assert config.bind_host == "127.0.0.1"
    assert config.port == 8765
    assert config.publish_rate_hz == 20.0
    assert config.mac_control_timeout_s == 0.15
    assert config.watchdog_timeout_s == 0.25
    assert config.stream_duration_s == 0.075
    assert set(config.mac_joint_rate_limits_rad_s.values()) == {0.25}
    assert set(config.pi_joint_rate_limits_rad_s.values()) == {0.275}
    assert len(compute_profile_sha256(config)) == 64
    changed = replace(config, watchdog_timeout_s=0.30)
    assert compute_profile_sha256(changed) != compute_profile_sha256(config)


def test_fragmented_and_coalesced_frames_decode() -> None:
    hello = encode_message(message("hello", profile_sha256=PROFILE))
    hold = encode_message(message("hold", seq=1, reason="operator_released"))
    decoder = NDJSONDecoder()

    assert decoder.feed(hello[:5]) == []
    decoded = decoder.feed(hello[5:] + hold)

    assert [item["type"] for item in decoded] == ["hello", "hold"]
    decoder.finish()


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\n",
        b"{}\n",
        b'{"v":1,"type":"target","session_id":"bad"}\n',
        (
            '{"v":1,"type":"target","session_id":"%s","seq":1,'
            '"joints_rad":{"base":NaN,"shoulder":-0.3,"elbow":-0.5,'
            '"wrist":-1.0}}\n' % SESSION
        ).encode(),
    ],
)
def test_invalid_utf8_json_version_session_and_nan_are_rejected(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        NDJSONDecoder().feed(payload)


@pytest.mark.parametrize(
    "invalid",
    [
        message("target", seq=True, joints_rad=joints()),
        message("target", seq=1, joints_rad=joints(base=True)),
        message("target", seq=1, joints_rad={"base": 0.0}),
        message("target", seq=1, joints_rad={**joints(), "gripper": 0.0}),
        {**message("hold", seq=1, reason="hold"), "unknown": 1},
        message("hello", profile_sha256="ABC"),
        {**message("hello", profile_sha256=PROFILE), "v": 1.0},
        message("mystery", seq=1),
    ],
)
def test_strict_schema_rejects_invalid_messages(invalid) -> None:
    with pytest.raises(ProtocolError):
        encode_message(invalid)


def test_duplicate_json_fields_and_oversized_frame_are_rejected() -> None:
    duplicate = (
        '{"v":1,"v":1,"type":"hello","session_id":"%s",'
        '"profile_sha256":"%s"}' % (SESSION, PROFILE)
    ).encode()
    with pytest.raises(ProtocolError, match="duplicate"):
        decode_line(duplicate)

    decoder = NDJSONDecoder(max_frame_bytes=32)
    with pytest.raises(ProtocolError, match="exceeds"):
        decoder.feed(b"x" * 33)


def test_wire_payload_contains_radians_and_never_pwm() -> None:
    packet = encode_message(message("target", seq=3, joints_rad=joints()))
    parsed = json.loads(packet)
    assert "joints_rad" in parsed
    assert "pwm" not in packet.decode().lower()
