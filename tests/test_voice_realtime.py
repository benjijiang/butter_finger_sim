"""Current OpenAI Realtime GA event-contract tests."""
from __future__ import annotations

import base64
from dataclasses import replace

from butter_finger.voice.config import load_voice_config
from butter_finger.voice.emotions import CONVERSATIONAL_ACTION_NAMES
from butter_finger.voice.realtime import (
    UsageReport,
    append_audio_event,
    build_session_update,
    decode_audio_delta,
    extract_function_calls,
    extract_response_transcript,
    spoken_response_event,
)


def test_session_uses_semantic_vad_required_emotion_and_pcm24() -> None:
    config = load_voice_config()

    event = build_session_update(config)
    session = event["session"]

    assert session["model"] == "gpt-realtime-2.1"
    assert session["reasoning"] == {"effort": "low"}
    assert session["tool_choice"] == "required"
    assert session["tools"][0]["strict"] is True
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "auto",
        "create_response": True,
        "interrupt_response": False,
    }
    action_schema = session["tools"][0]["parameters"]["properties"]["action"]
    assert tuple(action_schema["enum"]) == CONVERSATIONAL_ACTION_NAMES


def test_mini_model_is_configurable() -> None:
    config = replace(load_voice_config(), model="gpt-realtime-2.1-mini")
    assert build_session_update(config)["session"]["model"].endswith("-mini")


def test_audio_events_round_trip_pcm_bytes() -> None:
    pcm = b"\x01\x02\x03\x04"
    appended = append_audio_event(pcm)
    assert base64.b64decode(appended["audio"]) == pcm
    assert decode_audio_delta(
        {
            "type": "response.output_audio.delta",
            "delta": appended["audio"],
        }
    ) == pcm


def test_extracts_function_call_from_done_event() -> None:
    calls = extract_function_calls(
        {
            "type": "response.done",
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "express_emotion",
                        "arguments": '{"action":"happy"}',
                    }
                ]
            },
        }
    )
    assert len(calls) == 1
    assert calls[0].call_id == "call-1"
    assert calls[0].arguments == '{"action":"happy"}'


def test_spoken_response_disables_tools() -> None:
    response = spoken_response_event()["response"]
    assert response["output_modalities"] == ["audio"]
    assert response["tool_choice"] == "none"


def test_full_transcript_can_be_recovered_from_response_done() -> None:
    event = {
        "type": "response.done",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "audio", "transcript": "Fallback transcript."}
                    ],
                }
            ]
        },
    }
    assert extract_response_transcript(event) == "Fallback transcript."


def test_usage_reporting_includes_audio_and_text_details() -> None:
    report = UsageReport.from_response_done(
        {
            "response": {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                    "input_token_details": {
                        "audio_tokens": 10,
                        "text_tokens": 2,
                    },
                    "output_token_details": {
                        "audio_tokens": 6,
                        "text_tokens": 2,
                    },
                }
            }
        }
    )

    assert report.total_tokens == 20
    assert report.input_audio_tokens == 10
    assert report.output_audio_tokens == 6
    assert "total=20" in report.compact()
