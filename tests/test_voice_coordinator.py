"""Gesture validation, deduplication, overlap, and turn-lock tests."""
from __future__ import annotations

import threading

import pytest

from butter_finger.voice.coordinator import TurnCoordinator, TurnState
from butter_finger.voice.emotions import CONVERSATIONAL_ACTION_NAMES
from butter_finger.voice.realtime import FunctionCall


class RecordingActions:
    action_names = CONVERSATIONAL_ACTION_NAMES

    def __init__(self, *, block: bool = False) -> None:
        self.calls: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def run(self, name: str) -> None:
        self.calls.append(name)
        self.started.set()
        assert self.release.wait(timeout=2)


@pytest.mark.parametrize("action", CONVERSATIONAL_ACTION_NAMES)
def test_every_allowed_model_action_runs_existing_choreography(action: str) -> None:
    actions = RecordingActions()
    coordinator = TurnCoordinator(actions)
    try:
        decision = coordinator.handle_function_call(
            FunctionCall(
                f"call-{action}",
                "express_emotion",
                f'{{"action":"{action}"}}',
            )
        )
        assert decision.accepted
        completed = coordinator.complete_spoken_response()
        assert completed.action == action
        assert actions.calls == [action]
    finally:
        coordinator.stop()


@pytest.mark.parametrize(
    "call",
    [
        FunctionCall("bad-json", "express_emotion", "{"),
        FunctionCall("extra", "express_emotion", '{"action":"happy","pwm":2200}'),
        FunctionCall("unknown", "express_emotion", '{"action":"moonwalk"}'),
        FunctionCall("wrong-tool", "move_joint", '{"action":"happy"}'),
    ],
)
def test_malformed_or_unknown_calls_never_move(call: FunctionCall) -> None:
    actions = RecordingActions()
    coordinator = TurnCoordinator(actions)
    try:
        decision = coordinator.handle_function_call(call)
        assert decision.accepted is False
        assert actions.calls == []
    finally:
        coordinator.stop()


def test_duplicate_and_busy_calls_never_queue_a_second_gesture() -> None:
    actions = RecordingActions(block=True)
    coordinator = TurnCoordinator(actions)
    first = FunctionCall("one", "express_emotion", '{"action":"happy"}')
    second = FunctionCall("two", "express_emotion", '{"action":"sad"}')
    try:
        accepted = coordinator.handle_function_call(first)
        assert accepted.accepted
        assert actions.started.wait(timeout=1)
        assert coordinator.motion_running

        duplicate = coordinator.handle_function_call(first)
        busy = coordinator.handle_function_call(second)

        assert duplicate.duplicate
        assert not busy.accepted
        assert actions.calls == ["happy"]
        assert coordinator.on_speech_started() is False
        actions.release.set()
        completed = coordinator.complete_spoken_response()
        assert completed.action == "happy"
        assert coordinator.state == TurnState.LISTENING
    finally:
        actions.release.set()
        coordinator.stop()


def test_speech_can_overlap_motion_but_next_turn_remains_blocked() -> None:
    actions = RecordingActions(block=True)
    coordinator = TurnCoordinator(actions)
    try:
        decision = coordinator.handle_function_call(
            FunctionCall("one", "express_emotion", '{"action":"curious"}')
        )
        assert decision.accepted
        assert actions.started.wait(timeout=1)

        coordinator.append_transcript("I wonder ")
        coordinator.append_transcript("about that.")
        assert coordinator.state == TurnState.SPEAKING
        assert coordinator.motion_running
        assert coordinator.on_speech_stopped() is False

        actions.release.set()
        completed = coordinator.complete_spoken_response()
        assert completed.transcript == "I wonder about that."
    finally:
        actions.release.set()
        coordinator.stop()


def test_abandon_response_waits_for_validated_motion_to_finish() -> None:
    actions = RecordingActions(block=True)
    coordinator = TurnCoordinator(actions)
    waiter = threading.Thread(target=coordinator.abandon_response)
    try:
        decision = coordinator.handle_function_call(
            FunctionCall("one", "express_emotion", '{"action":"attentive"}')
        )
        assert decision.accepted
        assert actions.started.wait(timeout=1)

        waiter.start()
        waiter.join(timeout=0.05)
        assert waiter.is_alive()

        actions.release.set()
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        assert actions.calls == ["attentive"]
    finally:
        actions.release.set()
        if waiter.ident is not None:
            waiter.join(timeout=1)
        coordinator.stop()
