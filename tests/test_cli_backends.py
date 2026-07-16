"""CLI backend-selection tests with dependency-free fake arms."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from butter_finger import ArmBackend, load_arm_config
from examples import emotion_showcase, run_action


class FakeCLIArm(ArmBackend):
    time_step_s = 0.25

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.validations: list[dict[str, float]] = []
        self.positions = load_arm_config().home_pose

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        self.validations.append(dict(targets_rad))

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
        targets = dict(targets_rad)
        self.events.append(("move", targets, duration_s))
        self.positions.update(targets)

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.events.append(("home",))

    def disconnect(self) -> None:
        self.events.append(("disconnect",))

    def run_for(self, seconds: float) -> None:
        self.events.append(("run_for", seconds))

    def step(self, *, realtime: bool) -> None:
        self.events.append(("step", realtime))


def test_run_action_defaults_to_sim_backend() -> None:
    arm = FakeCLIArm()
    requested: list[str] = []

    result = run_action.main(
        ["base_scan"],
        arm_factory=lambda backend: requested.append(backend) or arm,
    )

    assert result == 0
    assert requested == ["sim"]
    assert ("home",) not in arm.events
    assert arm.events[-2:] == [("run_for", 1.0), ("disconnect",)]


def test_run_action_real_requires_explicit_confirmation() -> None:
    requested: list[str] = []

    with pytest.raises(SystemExit):
        run_action.main(
            ["base_scan", "--backend", "real"],
            arm_factory=lambda backend: requested.append(backend) or FakeCLIArm(),
        )
    assert requested == []


def test_run_action_real_homes_before_action() -> None:
    arm = FakeCLIArm()

    result = run_action.main(
        ["wrist_down", "--backend", "real", "--confirm-hardware"],
        arm_factory=lambda backend: arm,
    )

    assert result == 0
    assert arm.events[0] == ("home",)
    assert arm.events[1][0] == "move"
    assert all(event[0] != "run_for" for event in arm.events)
    assert arm.events[-1] == ("disconnect",)


def test_action_list_does_not_open_a_backend() -> None:
    requested: list[str] = []

    result = run_action.main(
        ["--list"],
        arm_factory=lambda backend: requested.append(backend) or FakeCLIArm(),
    )

    assert result == 0
    assert requested == []


def test_showcase_defaults_to_sim_backend() -> None:
    arm = FakeCLIArm()
    requested: list[str] = []

    result = emotion_showcase.main(
        ["happy", "--idle-seconds", "0"],
        arm_factory=lambda backend: requested.append(backend) or arm,
    )

    assert result == 0
    assert requested == ["sim"]
    assert ("home",) not in arm.events
    assert arm.events[-1] == ("disconnect",)


def test_showcase_real_requires_explicit_confirmation() -> None:
    requested: list[str] = []

    with pytest.raises(SystemExit):
        emotion_showcase.main(
            ["happy", "--backend", "real"],
            arm_factory=lambda backend: requested.append(backend) or FakeCLIArm(),
        )
    assert requested == []


def test_real_showcase_homes_then_waits_once_between_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = FakeCLIArm()
    monkeypatch.setattr(
        emotion_showcase,
        "_wait_real_gap",
        lambda seconds: arm.events.append(("real_gap", seconds)),
    )

    result = emotion_showcase.main(
        [
            "happy",
            "curious",
            "--backend",
            "real",
            "--confirm-hardware",
            "--idle-seconds",
            "0.25",
        ],
        arm_factory=lambda backend: arm,
    )

    assert result == 0
    assert arm.events[0] == ("home",)
    assert arm.events.count(("real_gap", 0.25)) == 1
    gap_index = arm.events.index(("real_gap", 0.25))
    assert any(event[0] == "move" for event in arm.events[1:gap_index])
    assert any(event[0] == "move" for event in arm.events[gap_index + 1 :])
    assert not any(event[0] == "step" for event in arm.events)
    assert arm.events[-1] == ("disconnect",)
