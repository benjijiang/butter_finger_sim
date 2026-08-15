"""Voice YAML validation and isolation tests."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from butter_finger.voice.config import (
    ALLOWED_MODELS,
    WAKE_PHRASE,
    load_voice_config,
)


def test_default_voice_config_matches_contract() -> None:
    config = load_voice_config()
    assert config.wake_phrase == "butter finger" == WAKE_PHRASE
    assert config.model == "gpt-realtime-2.1"
    assert "gpt-realtime-2.1-mini" in ALLOWED_MODELS
    assert config.reasoning_effort == "low"
    assert config.inactivity_timeout_s == 60
    assert config.idle_pose == "idle_ready"


def test_voice_config_rejects_wake_phrase_variation(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config" / "voice.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    invalid = copy.deepcopy(raw)
    invalid["voice"]["wake"]["phrase"] = "butter fingers"
    (tmp_path / "voice.yaml").write_text(
        yaml.safe_dump(invalid),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 'butter finger'"):
        load_voice_config(tmp_path)


def test_importing_voice_package_does_not_load_native_dependencies() -> None:
    import sys
    import butter_finger.voice  # noqa: F401

    assert "sounddevice" not in sys.modules
    assert "pocketsphinx" not in sys.modules
    assert "websocket" not in sys.modules
