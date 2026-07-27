"""Validated, dependency-free configuration for Pi voice chat."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from butter_finger.config import CONFIG_DIR

WAKE_PHRASE = "butter finger"
ALLOWED_MODELS = ("gpt-realtime-2.1", "gpt-realtime-2.1-mini")


@dataclass(frozen=True)
class VoiceConfig:
    wake_phrase: str
    wake_threshold: float
    wake_sample_rate_hz: int
    api_sample_rate_hz: int
    audio_chunk_ms: int
    device_name: str | None
    input_device_id: int | None
    output_device_id: int | None
    model: str
    voice: str
    reasoning_effort: str
    vad_eagerness: str
    inactivity_timeout_s: float
    reconnect_delay_s: float
    idle_pose: str
    idle_move_duration_s: float


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite number greater than zero")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_device_id(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be null or a non-negative integer")
    return value


def load_voice_config(config_dir: Path = CONFIG_DIR) -> VoiceConfig:
    """Load voice.yaml without importing any optional voice dependency."""
    path = config_dir / "voice.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "voice.yaml")
    if set(root) != {"voice"}:
        raise ValueError("voice.yaml must contain only the 'voice' section")
    voice = _mapping(root["voice"], "voice")
    expected = {"wake", "audio", "realtime", "arm"}
    if set(voice) != expected:
        raise ValueError(
            "voice configuration sections must be exactly "
            f"{sorted(expected)}"
        )

    wake = _mapping(voice["wake"], "voice.wake")
    audio = _mapping(voice["audio"], "voice.audio")
    realtime = _mapping(voice["realtime"], "voice.realtime")
    arm = _mapping(voice["arm"], "voice.arm")

    phrase = wake.get("phrase")
    if phrase != WAKE_PHRASE:
        raise ValueError(
            f"voice.wake.phrase must be exactly {WAKE_PHRASE!r}"
        )
    threshold = _positive_number(wake.get("threshold"), "voice.wake.threshold")
    wake_rate = _positive_int(
        wake.get("sample_rate_hz"), "voice.wake.sample_rate_hz"
    )

    api_rate = _positive_int(
        audio.get("api_sample_rate_hz"), "voice.audio.api_sample_rate_hz"
    )
    if api_rate != 24000:
        raise ValueError("voice.audio.api_sample_rate_hz must be 24000")
    chunk_ms = _positive_int(audio.get("chunk_ms"), "voice.audio.chunk_ms")
    device_name = audio.get("device_name")
    if device_name is not None and (
        not isinstance(device_name, str) or not device_name.strip()
    ):
        raise ValueError("voice.audio.device_name must be null or non-empty")

    model = realtime.get("model")
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"voice.realtime.model must be one of {list(ALLOWED_MODELS)}"
        )
    output_voice = realtime.get("voice")
    if not isinstance(output_voice, str) or not output_voice:
        raise ValueError("voice.realtime.voice must be a non-empty string")
    reasoning = realtime.get("reasoning_effort")
    if reasoning != "low":
        raise ValueError("voice.realtime.reasoning_effort must be 'low'")
    eagerness = realtime.get("vad_eagerness")
    if eagerness not in {"low", "medium", "high", "auto"}:
        raise ValueError(
            "voice.realtime.vad_eagerness must be low, medium, high, or auto"
        )

    idle_pose = arm.get("idle_pose")
    if idle_pose != "idle_ready":
        raise ValueError("voice.arm.idle_pose must be 'idle_ready'")

    return VoiceConfig(
        wake_phrase=phrase,
        wake_threshold=threshold,
        wake_sample_rate_hz=wake_rate,
        api_sample_rate_hz=api_rate,
        audio_chunk_ms=chunk_ms,
        device_name=device_name.strip() if isinstance(device_name, str) else None,
        input_device_id=_optional_device_id(
            audio.get("input_device_id"), "voice.audio.input_device_id"
        ),
        output_device_id=_optional_device_id(
            audio.get("output_device_id"), "voice.audio.output_device_id"
        ),
        model=model,
        voice=output_voice,
        reasoning_effort=reasoning,
        vad_eagerness=eagerness,
        inactivity_timeout_s=_positive_number(
            realtime.get("inactivity_timeout_s"),
            "voice.realtime.inactivity_timeout_s",
        ),
        reconnect_delay_s=_positive_number(
            audio.get("reconnect_delay_s"), "voice.audio.reconnect_delay_s"
        ),
        idle_pose=idle_pose,
        idle_move_duration_s=_positive_number(
            arm.get("idle_move_duration_s"),
            "voice.arm.idle_move_duration_s",
        ),
    )
