"""Safety-focused turn state and gesture scheduling."""
from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from butter_finger.voice.emotions import CONVERSATIONAL_ACTION_NAMES
from butter_finger.voice.realtime import FunctionCall, UsageReport


class ActionExecutor(Protocol):
    @property
    def action_names(self) -> tuple[str, ...]: ...

    def run(self, name: str) -> None: ...


class TurnState(str, Enum):
    LISTENING = "listening"
    WAITING_EMOTION = "waiting_emotion"
    SPEAKING = "speaking"
    STOPPING = "stopping"


@dataclass(frozen=True)
class EmotionDecision:
    accepted: bool
    call_id: str
    action: str | None = None
    error: str | None = None
    duplicate: bool = False
    should_retry: bool = False


@dataclass(frozen=True)
class CompletedTurn:
    action: str
    transcript: str
    usage: UsageReport


class TurnCoordinator:
    """Allow at most one validated gesture and block input until it finishes."""

    def __init__(self, action_executor: ActionExecutor) -> None:
        missing = [
            action
            for action in CONVERSATIONAL_ACTION_NAMES
            if action not in action_executor.action_names
        ]
        if missing:
            raise ValueError(f"Action executor is missing voice actions: {missing}")
        self._actions = action_executor
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="butter-finger-gesture",
        )
        self._state = TurnState.LISTENING
        self._seen_call_ids: set[str] = set()
        self._motion: Future[None] | None = None
        self._current_action: str | None = None
        self._transcript_parts: list[str] = []
        self._usage = UsageReport()
        self._retry_used = False
        self._lock = threading.Lock()

    @property
    def state(self) -> TurnState:
        with self._lock:
            return self._state

    @property
    def motion_running(self) -> bool:
        with self._lock:
            return self._motion is not None and not self._motion.done()

    def on_speech_started(self) -> bool:
        """Begin a turn only when neither speech nor motion is still active."""
        with self._lock:
            if self._state != TurnState.LISTENING:
                return False
            self._transcript_parts = []
            self._usage = UsageReport()
            self._retry_used = False
            return True

    def on_speech_stopped(self) -> bool:
        with self._lock:
            if self._state != TurnState.LISTENING:
                return False
            self._state = TurnState.WAITING_EMOTION
            return True

    def handle_function_call(self, call: FunctionCall) -> EmotionDecision:
        """Validate a model call before scheduling any physical motion."""
        with self._lock:
            if call.call_id in self._seen_call_ids:
                return EmotionDecision(
                    accepted=False,
                    call_id=call.call_id,
                    error="duplicate function call",
                    duplicate=True,
                )
            self._seen_call_ids.add(call.call_id)

            if self._state == TurnState.STOPPING:
                return EmotionDecision(
                    accepted=False,
                    call_id=call.call_id,
                    error="voice chat is stopping",
                )
            if self._state == TurnState.SPEAKING or self._motion is not None:
                return EmotionDecision(
                    accepted=False,
                    call_id=call.call_id,
                    error="a gesture or spoken response is already active",
                )
            if self._state not in {
                TurnState.LISTENING,
                TurnState.WAITING_EMOTION,
            }:
                return EmotionDecision(
                    accepted=False,
                    call_id=call.call_id,
                    error=f"cannot start a gesture while {self._state.value}",
                )

            error: str | None = None
            action: Any = None
            if call.name != "express_emotion":
                error = "unknown function"
            else:
                try:
                    arguments = json.loads(call.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = None
                    error = "arguments must be valid JSON"
                if error is None and (
                    not isinstance(arguments, dict)
                    or set(arguments) != {"action"}
                ):
                    error = "arguments must contain only 'action'"
                if error is None:
                    action = arguments["action"]
                    if (
                        not isinstance(action, str)
                        or action not in CONVERSATIONAL_ACTION_NAMES
                        or action not in self._actions.action_names
                    ):
                        error = "unknown emotional action"

            if error is not None:
                retry = not self._retry_used
                self._retry_used = True
                self._state = TurnState.WAITING_EMOTION
                return EmotionDecision(
                    accepted=False,
                    call_id=call.call_id,
                    error=error,
                    should_retry=retry,
                )

            self._current_action = action
            self._motion = self._pool.submit(self._actions.run, action)
            self._state = TurnState.SPEAKING
            return EmotionDecision(
                accepted=True,
                call_id=call.call_id,
                action=action,
            )

    def append_transcript(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._state == TurnState.SPEAKING:
                self._transcript_parts.append(text)

    def add_usage(self, usage: UsageReport) -> None:
        with self._lock:
            self._usage = self._usage + usage

    def complete_spoken_response(self) -> CompletedTurn:
        """Wait for motion and release input only after both outputs finish."""
        with self._lock:
            if self._state != TurnState.SPEAKING:
                raise RuntimeError("no spoken response is active")
            motion = self._motion
            action = self._current_action
        if motion is None or action is None:
            raise RuntimeError("speaking state has no gesture")
        motion.result()
        with self._lock:
            completed = CompletedTurn(
                action=action,
                transcript="".join(self._transcript_parts).strip(),
                usage=self._usage,
            )
            self._motion = None
            self._current_action = None
            self._state = TurnState.LISTENING
            self._transcript_parts = []
            self._usage = UsageReport()
            return completed

    def abandon_response(self) -> None:
        """Finish a validated gesture even if speech or the network failed."""
        with self._lock:
            motion = self._motion
        if motion is not None:
            motion.result()
        with self._lock:
            self._motion = None
            self._current_action = None
            if self._state != TurnState.STOPPING:
                self._state = TurnState.LISTENING

    def stop(self) -> None:
        with self._lock:
            self._state = TurnState.STOPPING
            motion = self._motion
        if motion is not None:
            motion.result()
        self._pool.shutdown(wait=True, cancel_futures=False)
