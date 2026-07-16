"""Test emotional showcase playlist selection without opening PyBullet."""
from __future__ import annotations

import pytest

from butter_finger import load_action_config
from examples.emotion_showcase import (
    EMOTION_ACTION_NAMES,
    _run_idle_gap,
    select_action_names,
)


class FakeShowcaseArm:
    time_step_s = 0.25

    def __init__(self) -> None:
        self.step_calls: list[bool] = []

    def step(self, *, realtime: bool) -> None:
        self.step_calls.append(realtime)


class FakeIdle:
    def __init__(self) -> None:
        self.resume_calls = 0
        self.updates: list[float] = []

    def resume(self) -> None:
        self.resume_calls += 1

    def update(self, dt_s: float) -> None:
        self.updates.append(dt_s)


def test_all_selects_twenty_emotional_actions_in_defined_order() -> None:
    config = load_action_config()

    names = select_action_names([], play_all=True, config=config)

    assert names == EMOTION_ACTION_NAMES
    assert len(names) == 20
    assert set(names).issubset(config.actions)


def test_explicit_playlist_preserves_order() -> None:
    config = load_action_config()

    names = select_action_names(
        ["happy", "curious", "sad", "surprised"],
        play_all=False,
        config=config,
    )

    assert names == ("happy", "curious", "sad", "surprised")


@pytest.mark.parametrize(
    ("requested", "play_all", "message"),
    [
        ([], False, "provide action names"),
        (["happy"], True, "either --all"),
        (["missing"], False, "unknown action"),
    ],
)
def test_rejects_invalid_playlist(
    requested: list[str],
    play_all: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_action_names(
            requested,
            play_all=play_all,
            config=load_action_config(),
        )


def test_idle_gap_resumes_and_steps_for_requested_duration() -> None:
    arm = FakeShowcaseArm()
    idle = FakeIdle()

    _run_idle_gap(arm, idle, 1.0)

    assert idle.resume_calls == 1
    assert idle.updates == [0.25] * 4
    assert arm.step_calls == [True] * 4
