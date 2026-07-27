"""Voice CLI hardware gating, API-key, diagnostics, and Ctrl-C cleanup."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from butter_finger.arm import ArmBackend
from butter_finger.config import load_arm_config
from butter_finger.voice.audio import AudioDevice
from examples import voice_chat


class FakeAudio:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.events: list[str] = []
        self.interrupt = interrupt

    def open(self) -> None:
        self.events.append("open")

    def read(self, _sample_rate_hz: int) -> bytes:
        self.events.append("read")
        if self.interrupt:
            raise KeyboardInterrupt
        return b"\x00\x00"

    def write(self, _pcm16: bytes, _sample_rate_hz: int) -> None:
        self.events.append("write")

    def drain(self) -> None:
        self.events.append("drain")

    def reconnect(self) -> None:
        self.events.append("reconnect")

    def close(self) -> None:
        self.events.append("close")


class FakeWake:
    def __init__(self) -> None:
        self.closed = False

    def process(self, _pcm16: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


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
        self.events.append(("move", dict(targets_rad), duration_s))

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.events.append(("home",))

    def disconnect(self) -> None:
        self.events.append(("disconnect",))


def test_real_hardware_requires_confirmation_before_arm_factory() -> None:
    audio = FakeAudio()
    created: list[bool] = []

    with pytest.raises(SystemExit):
        voice_chat.main(
            [],
            audio_factory=lambda _config: audio,
            wake_factory=lambda _config: FakeWake(),
            arm_factory=lambda dry_run: created.append(dry_run) or FakeArm(),
        )

    assert created == []
    assert audio.events == ["open", "close"]


def test_missing_api_key_never_opens_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    created: list[bool] = []

    result = voice_chat.main(
        ["--dry-run"],
        audio_factory=lambda _config: FakeAudio(),
        wake_factory=lambda _config: FakeWake(),
        arm_factory=lambda dry_run: created.append(dry_run) or FakeArm(),
    )

    assert result == 2
    assert created == []


def test_dry_run_ctrl_c_homes_and_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    audio = FakeAudio(interrupt=True)
    wake = FakeWake()
    arm = FakeArm()
    requested: list[bool] = []

    result = voice_chat.main(
        ["--dry-run"],
        audio_factory=lambda _config: audio,
        wake_factory=lambda _config: wake,
        arm_factory=lambda dry_run: requested.append(dry_run) or arm,
    )

    assert result == 0
    assert requested == [True]
    assert arm.events[0] == ("home",)
    assert arm.events[-2:] == [("home",), ("disconnect",)]
    assert audio.events[-1] == "close"
    assert wake.closed


def test_audio_listing_never_builds_audio_or_arm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        voice_chat,
        "query_audio_devices",
        lambda: (AudioDevice(7, "BT HFP", 1, 1, 16000),),
    )
    built: list[str] = []

    result = voice_chat.main(
        ["--list-audio-devices"],
        audio_factory=lambda _config: built.append("audio") or FakeAudio(),
        arm_factory=lambda _dry_run: built.append("arm") or FakeArm(),
    )

    assert result == 0
    assert built == []
    assert "BT HFP" in capsys.readouterr().out
