"""The only arm actions a voice response is allowed to select."""
from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

CONVERSATIONAL_ACTION_NAMES: tuple[str, ...] = (
    "greet",
    "nod_yes",
    "shake_no",
    "attentive",
    "happy",
    "excited",
    "proud",
    "playful",
    "affectionate",
    "shy",
    "curious",
    "thinking",
    "confused",
    "sad",
    "disappointed",
    "bored",
    "sleepy",
    "surprised",
    "scared",
    "angry",
)


def validate_conversational_actions(
    configured_actions: Collection[str],
) -> None:
    """Ensure every model-visible action has local choreography."""
    missing = [
        action
        for action in CONVERSATIONAL_ACTION_NAMES
        if action not in configured_actions
    ]
    if missing:
        raise ValueError(
            "Voice actions are missing from actions.yaml: "
            f"{missing}"
        )


def emotion_tool_schema() -> Mapping[str, Any]:
    """Return the strict, enum-constrained local Realtime function schema."""
    return {
        "type": "function",
        "name": "express_emotion",
        "strict": True,
        "description": (
            "Choose exactly one existing Butter Finger arm action that "
            "matches the emotional tone of the assistant's next spoken reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(CONVERSATIONAL_ACTION_NAMES),
                    "description": "The single validated arm action to run.",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }
