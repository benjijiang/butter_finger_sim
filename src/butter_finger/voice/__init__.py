"""Optional Raspberry Pi voice assistant support.

The package deliberately does not import sounddevice, PocketSphinx, or
websocket-client at import time. Those dependencies are loaded only when the
corresponding runtime object is constructed.
"""
from __future__ import annotations

from butter_finger.voice.config import VoiceConfig, load_voice_config
from butter_finger.voice.emotions import CONVERSATIONAL_ACTION_NAMES

__all__ = [
    "CONVERSATIONAL_ACTION_NAMES",
    "VoiceConfig",
    "load_voice_config",
]
