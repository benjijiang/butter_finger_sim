"""Fake-driven voice lifecycle, reporting, timeout, and cleanup tests."""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from butter_finger.arm import ArmBackend
from butter_finger.config import load_action_config, load_arm_config
from butter_finger.voice.config import load_voice_config
from butter_finger.voice.realtime import RealtimeError, RealtimeTransport
from butter_finger.voice.runtime import VoiceChatRuntime


class FakeArm(ArmBackend):
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.positions = load_arm_config().home_pose

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        self.events.append(("validate", dict(targets_rad)))

    def move_joint(
        self,
        joint: str,
        position_rad: float,
        *,
        duration_s: float | None = None,
    ) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(
        self,
        targets_rad: Mapping[str, float],
        *,
        duration_s: float | None = None,
    ) -> None:
        targets = dict(targets_rad)
        self.positions.update(targets)
        self.events.append(("move", targets, duration_s))

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.events.append(("home",))
        self.positions = load_arm_config().home_pose

    def disconnect(self) -> None:
        self.events.append(("disconnect",))


class FakeAudio:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.reads = 0
        self.reconnect_failures = 0

    def open(self) -> None:
        self.events.append(("open",))

    def read(self, sample_rate_hz: int) -> bytes:
        self.reads += 1
        self.events.append(("read", sample_rate_hz))
        return b"\x00\x00" * 16

    def write(self, pcm16: bytes, sample_rate_hz: int) -> None:
        self.events.append(("write", pcm16, sample_rate_hz))

    def drain(self) -> None:
        self.events.append(("drain",))

    def reconnect(self) -> None:
        self.events.append(("reconnect",))
        if self.reconnect_failures:
            self.reconnect_failures -= 1
            raise RuntimeError("still disconnected")

    def close(self) -> None:
        self.events.append(("close",))


class FakeWake:
    def __init__(self, detections: list[bool] | None = None) -> None:
        self.detections = list(detections or [])
        self.events: list[str] = []

    def process(self, _pcm16: bytes) -> bool:
        self.events.append("process")
        return self.detections.pop(0) if self.detections else False

    def reset(self) -> None:
        self.events.append("reset")

    def close(self) -> None:
        self.events.append("close")


class FakeTransport(RealtimeTransport):
    def __init__(self, events: list[dict[str, Any] | None]) -> None:
        self.events = list(events)
        self.sent: list[Mapping[str, Any]] = []
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def send_event(self, event: Mapping[str, Any]) -> None:
        self.sent.append(event)

    def receive_event(self, timeout_s: float) -> dict[str, Any] | None:
        assert timeout_s >= 0
        return self.events.pop(0) if self.events else None

    def close(self) -> None:
        self.closed = True


def usage(total: int) -> dict[str, Any]:
    return {
        "input_tokens": total // 2,
        "output_tokens": total - total // 2,
        "total_tokens": total,
    }


def test_conversation_assembles_transcript_usage_and_single_emotion() -> None:
    config = load_voice_config()
    audio = FakeAudio()
    wake = FakeWake()
    arm = FakeArm()
    audio_bytes = b"\x11\x22\x33\x44"
    transport = FakeTransport(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-1",
                "name": "express_emotion",
                "arguments": '{"action":"happy"}',
            },
            {
                "type": "response.done",
                "response": {
                    "usage": usage(3),
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "express_emotion",
                            "arguments": '{"action":"happy"}',
                        }
                    ],
                },
            },
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(audio_bytes).decode("ascii"),
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "Hello ",
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "friend!",
            },
            {
                "type": "response.done",
                "response": {"usage": usage(7), "output": []},
            },
            None,
        ]
    )
    messages: list[str] = []
    transcript_stream: list[str] = []
    clock = iter((0.0, 1.0, 100.0))
    runtime = VoiceChatRuntime(
        config=config,
        audio=audio,
        wake_detector=wake,
        arm=arm,
        transport_factory=lambda: transport,
        output=messages.append,
        stream_output=transcript_stream.append,
        monotonic=lambda: next(clock),
        audio_is_open=True,
    )

    runtime._conversation()

    moves = [event for event in arm.events if event[0] == "move"]
    assert len(moves) == len(load_action_config().actions["happy"].steps)
    assert ("write", audio_bytes, 24000) in audio.events
    assert ("drain",) in audio.events
    assert "".join(transcript_stream) == "Assistant: Hello friend!\n"
    assert "Emotion: happy" in messages
    assert any("total=10" in message for message in messages)
    sent_types = [event["type"] for event in transport.sent]
    assert sent_types.count("conversation.item.create") == 1
    speech_response = [
        event for event in transport.sent if event["type"] == "response.create"
    ]
    assert len(speech_response) == 1
    assert speech_response[0]["response"]["tool_choice"] == "none"
    assert transport.closed


def test_silence_timeout_closes_session_and_clears_transport() -> None:
    config = replace(load_voice_config(), inactivity_timeout_s=5.0)
    transport = FakeTransport([None])
    clock = iter((10.0, 16.0))
    runtime = VoiceChatRuntime(
        config=config,
        audio=FakeAudio(),
        wake_detector=FakeWake(),
        arm=FakeArm(),
        transport_factory=lambda: transport,
        monotonic=lambda: next(clock),
        audio_is_open=True,
    )

    runtime._conversation()

    assert transport.closed
    assert runtime._transport is None
    assert runtime._coordinator is None


def test_multiple_emotion_calls_are_rejected_before_any_motion() -> None:
    transport = FakeTransport(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "response.done",
                "response": {
                    "usage": usage(2),
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "one",
                            "name": "express_emotion",
                            "arguments": '{"action":"happy"}',
                        },
                        {
                            "type": "function_call",
                            "call_id": "two",
                            "name": "express_emotion",
                            "arguments": '{"action":"sad"}',
                        },
                    ],
                },
            },
        ]
    )
    arm = FakeArm()
    clock = iter((0.0, 1.0))
    runtime = VoiceChatRuntime(
        config=load_voice_config(),
        audio=FakeAudio(),
        wake_detector=FakeWake(),
        arm=arm,
        transport_factory=lambda: transport,
        monotonic=lambda: next(clock),
        audio_is_open=True,
    )

    with pytest.raises(RealtimeError, match="more than one"):
        runtime._conversation()

    assert not any(event[0] == "move" for event in arm.events)
    rejected = [
        event
        for event in transport.sent
        if event["type"] == "conversation.item.create"
    ]
    assert len(rejected) == 2


def test_startup_homes_then_moves_idle_and_shutdown_homes_again() -> None:
    audio = FakeAudio()
    arm = FakeArm()
    wake = FakeWake()
    runtime = VoiceChatRuntime(
        config=load_voice_config(),
        audio=audio,
        wake_detector=wake,
        arm=arm,
        transport_factory=lambda: FakeTransport([]),
    )

    runtime._startup()
    runtime.shutdown()

    assert arm.events[0] == ("home",)
    assert arm.events[1][0] == "validate"
    assert arm.events[2][0] == "move"
    assert arm.events[-2:] == [("home",), ("disconnect",)]
    assert audio.events == [("open",), ("close",)]
    assert wake.events[-1] == "close"


def test_audio_reconnect_retries_then_resumes_local_wake_mode() -> None:
    audio = FakeAudio()
    audio.reconnect_failures = 1
    messages: list[str] = []
    runtime = VoiceChatRuntime(
        config=replace(load_voice_config(), reconnect_delay_s=0.001),
        audio=audio,
        wake_detector=FakeWake(),
        arm=FakeArm(),
        transport_factory=lambda: FakeTransport([]),
        output=messages.append,
        audio_is_open=True,
    )

    assert runtime._recover_audio(RuntimeError("gone")) is True
    assert audio.events.count(("reconnect",)) == 2
    assert messages[-1] == "Bluetooth audio reconnected."


def test_wake_audio_is_local_until_detector_activates() -> None:
    created = 0

    def make_transport() -> FakeTransport:
        nonlocal created
        created += 1
        return FakeTransport([])

    runtime = VoiceChatRuntime(
        config=load_voice_config(),
        audio=FakeAudio(),
        wake_detector=FakeWake([False, True]),
        arm=FakeArm(),
        transport_factory=make_transport,
        audio_is_open=True,
    )

    assert runtime._wait_for_wake() is True
    assert created == 0
