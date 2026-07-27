"""OpenAI Realtime GA event construction, parsing, and WebSocket transport."""
from __future__ import annotations

import base64
import json
import queue
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import urlencode

from butter_finger.voice.config import VoiceConfig
from butter_finger.voice.emotions import emotion_tool_schema


class RealtimeError(RuntimeError):
    """A Realtime connection or event failed."""


@dataclass(frozen=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class UsageReport:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    input_text_tokens: int = 0
    output_text_tokens: int = 0

    @classmethod
    def from_response_done(cls, event: Mapping[str, Any]) -> "UsageReport":
        response = event.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        if not isinstance(usage, Mapping):
            return cls()
        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})
        if not isinstance(input_details, Mapping):
            input_details = {}
        if not isinstance(output_details, Mapping):
            output_details = {}

        def number(mapping: Mapping[str, Any], key: str) -> int:
            value = mapping.get(key, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        return cls(
            input_tokens=number(usage, "input_tokens"),
            output_tokens=number(usage, "output_tokens"),
            total_tokens=number(usage, "total_tokens"),
            input_audio_tokens=number(input_details, "audio_tokens"),
            output_audio_tokens=number(output_details, "audio_tokens"),
            input_text_tokens=number(input_details, "text_tokens"),
            output_text_tokens=number(output_details, "text_tokens"),
        )

    def __add__(self, other: "UsageReport") -> "UsageReport":
        return UsageReport(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_audio_tokens=self.input_audio_tokens + other.input_audio_tokens,
            output_audio_tokens=self.output_audio_tokens + other.output_audio_tokens,
            input_text_tokens=self.input_text_tokens + other.input_text_tokens,
            output_text_tokens=self.output_text_tokens + other.output_text_tokens,
        )

    def compact(self) -> str:
        return (
            f"total={self.total_tokens}, input={self.input_tokens} "
            f"(audio={self.input_audio_tokens}, text={self.input_text_tokens}), "
            f"output={self.output_tokens} "
            f"(audio={self.output_audio_tokens}, text={self.output_text_tokens})"
        )


class RealtimeTransport(Protocol):
    def connect(self) -> None: ...

    def send_event(self, event: Mapping[str, Any]) -> None: ...

    def receive_event(self, timeout_s: float) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


def assistant_instructions() -> str:
    return (
        "You are the friendly voice of a small desktop robot arm named Butter "
        "Finger. Answer naturally and concisely. For every user turn, first "
        "call express_emotion exactly once to select the arm action that best "
        "matches the emotional tone of your intended answer. Do not mention "
        "the tool or invent motion parameters. After the tool result, speak "
        "the answer in a tone consistent with the selected action."
    )


def build_session_update(config: VoiceConfig) -> dict[str, Any]:
    """Create a GA Realtime session with tool-required semantic VAD turns."""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": config.model,
            "instructions": assistant_instructions(),
            "output_modalities": ["audio"],
            "reasoning": {"effort": config.reasoning_effort},
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": config.api_sample_rate_hz,
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": config.vad_eagerness,
                        "create_response": True,
                        "interrupt_response": False,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": config.api_sample_rate_hz,
                    },
                    "voice": config.voice,
                },
            },
            "tools": [dict(emotion_tool_schema())],
            "tool_choice": "required",
        },
    }


def append_audio_event(pcm16: bytes) -> dict[str, str]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm16).decode("ascii"),
    }


def function_output_event(
    call_id: str,
    *,
    accepted: bool,
    action: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {"accepted": accepted}
    if action is not None:
        output["action"] = action
    if error is not None:
        output["error"] = error
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output, separators=(",", ":")),
        },
    }


def spoken_response_event() -> dict[str, Any]:
    """Disable tools for the spoken half of the current response."""
    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "tool_choice": "none",
            "instructions": (
                "Now speak the answer to the user's latest message. Do not "
                "call another tool and do not describe the arm gesture."
            ),
            "metadata": {"butter_finger_phase": "speech"},
        },
    }


def emotion_retry_event() -> dict[str, Any]:
    return {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "tool_choice": "required",
            "instructions": (
                "Retry express_emotion once with exactly one action from its "
                "enum and no extra arguments. Do not speak yet."
            ),
            "metadata": {"butter_finger_phase": "emotion_retry"},
        },
    }


def decode_audio_delta(event: Mapping[str, Any]) -> bytes:
    delta = event.get("delta")
    if not isinstance(delta, str):
        raise RealtimeError("response.output_audio.delta is missing base64 data")
    try:
        return base64.b64decode(delta, validate=True)
    except ValueError as exc:
        raise RealtimeError("response audio delta is invalid base64") from exc


def extract_function_calls(event: Mapping[str, Any]) -> tuple[FunctionCall, ...]:
    """Extract complete calls from either the delta completion or response."""
    event_type = event.get("type")
    if event_type == "response.function_call_arguments.done":
        call_id = event.get("call_id")
        name = event.get("name")
        arguments = event.get("arguments")
        if all(isinstance(value, str) for value in (call_id, name, arguments)):
            return (FunctionCall(call_id, name, arguments),)
        return ()
    if event_type != "response.done":
        return ()
    response = event.get("response")
    output = response.get("output", []) if isinstance(response, Mapping) else []
    if not isinstance(output, list):
        return ()
    calls: list[FunctionCall] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if all(isinstance(value, str) for value in (call_id, name, arguments)):
            calls.append(FunctionCall(call_id, name, arguments))
    return tuple(calls)


def extract_response_transcript(event: Mapping[str, Any]) -> str:
    """Read the full audio transcript retained in a response.done payload."""
    if event.get("type") != "response.done":
        return ""
    response = event.get("response")
    output = response.get("output", []) if isinstance(response, Mapping) else []
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            transcript = part.get("transcript")
            if isinstance(transcript, str):
                parts.append(transcript)
    return "".join(parts)


class WebSocketRealtimeTransport:
    """Threaded websocket-client adapter with a pollable inbound event queue."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        connect_timeout_s: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._model = model
        self._connect_timeout_s = connect_timeout_s
        self._socket: Any = None
        self._incoming: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self._receiver: threading.Thread | None = None

    def connect(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RealtimeError(
                "websocket-client is not installed; run "
                "pip install -e '.[voice]'"
            ) from exc
        url = "wss://api.openai.com/v1/realtime?" + urlencode(
            {"model": self._model}
        )
        try:
            self._socket = websocket.create_connection(
                url,
                header=[f"Authorization: Bearer {self._api_key}"],
                timeout=self._connect_timeout_s,
            )
            self._socket.settimeout(1.0)
        except Exception as exc:
            raise RealtimeError(f"Could not connect to OpenAI Realtime: {exc}") from exc
        self._closed.clear()
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name="butter-finger-realtime-receiver",
            daemon=True,
        )
        self._receiver.start()

    def _receive_loop(self) -> None:
        while not self._closed.is_set():
            try:
                message = self._socket.recv()
                if not message:
                    raise RealtimeError("OpenAI Realtime connection closed")
                event = json.loads(message)
                if isinstance(event, dict):
                    self._incoming.put(event)
            except Exception as exc:
                if exc.__class__.__name__ == "WebSocketTimeoutException":
                    continue
                if not self._closed.is_set():
                    self._incoming.put(exc)
                return

    def send_event(self, event: Mapping[str, Any]) -> None:
        if self._socket is None or self._closed.is_set():
            raise RealtimeError("Realtime transport is not connected")
        try:
            with self._send_lock:
                self._socket.send(json.dumps(event, separators=(",", ":")))
        except Exception as exc:
            raise RealtimeError(f"Could not send Realtime event: {exc}") from exc

    def receive_event(self, timeout_s: float) -> dict[str, Any] | None:
        try:
            item = self._incoming.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None
        if isinstance(item, BaseException):
            raise RealtimeError(f"Realtime receive failed: {item}") from item
        return item

    def close(self) -> None:
        self._closed.set()
        socket = self._socket
        self._socket = None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        receiver = self._receiver
        self._receiver = None
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=2.0)
