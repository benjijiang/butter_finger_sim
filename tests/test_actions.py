"""Validate named actions, timed motion, and backend-neutral scheduling."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from butter_finger import (
    ActionRunner,
    ArmBackend,
    RaspberryPiArm,
    UnknownActionError,
    load_action_config,
    load_arm_config,
)
from butter_finger.backends.pybullet_arm import PyBulletArm, _smoothstep_trajectory

EXPECTED_ACTIONS = (
    "home",
    "demo_reach",
    "base_scan",
    "reach_and_return",
    "wrist_up",
    "wrist_down",
)


class FakeArm(ArmBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], float | None]] = []
        self.positions = {joint: 0.0 for joint in load_arm_config().joint_order}

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
        self.calls.append((targets, duration_s))
        self.positions.update(targets)

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def go_home(self) -> None:
        self.move_joints({joint: 0.0 for joint in self.positions})

    def disconnect(self) -> None:
        pass


class FakePyBullet:
    POSITION_CONTROL = 1
    GUI = 2
    DIRECT = 3

    def __init__(self) -> None:
        self.states = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.control_calls: list[dict] = []
        self.step_calls = 0

    def setJointMotorControl2(self, **kwargs) -> None:
        self.control_calls.append(kwargs)
        self.states[kwargs["jointIndex"]] = kwargs["targetPosition"]

    def getJointState(self, body_id, joint_index, physicsClientId):
        return (self.states[joint_index], 0.0, (), 0.0)

    def stepSimulation(self, physicsClientId) -> None:
        self.step_calls += 1

    def getConnectionInfo(self, client_id) -> dict[str, int]:
        return {"connectionMethod": self.DIRECT}


def make_fake_pybullet_arm() -> tuple[PyBulletArm, FakePyBullet]:
    arm = object.__new__(PyBulletArm)
    pb = FakePyBullet()
    arm._config = load_arm_config()
    arm._pb = pb
    arm._client = 7
    arm._robot_id = 11
    arm._joint_indices = {
        "base": 0,
        "shoulder": 1,
        "elbow": 2,
        "wrist": 3,
    }
    return arm, pb


def write_actions(tmp_path: Path, actions: dict) -> Path:
    (tmp_path / "actions.yaml").write_text(
        yaml.safe_dump({"actions": actions}),
        encoding="utf-8",
    )
    return tmp_path


def test_loads_expected_actions_and_resolves_poses() -> None:
    config = load_action_config()

    assert config.action_names == EXPECTED_ACTIONS
    assert dict(config.actions["home"].steps[0].targets_rad) == {
        "base": 0.0,
        "shoulder": 0.0,
        "elbow": 0.0,
        "wrist": -1.571,
    }
    assert dict(config.actions["demo_reach"].steps[0].targets_rad) == {
        "base": 0.8,
        "shoulder": -0.6,
        "elbow": -0.9,
        "wrist": 0.4,
    }


def test_configured_development_action_values() -> None:
    actions = load_action_config().actions

    assert [step.targets_rad["base"] for step in actions["base_scan"].steps] == [
        0.0,
        -0.8,
        0.8,
        0.0,
    ]
    assert actions["wrist_up"].steps[0].targets_rad["wrist"] == 0.6
    assert actions["wrist_down"].steps[0].targets_rad["wrist"] == -0.6
    assert len(actions["reach_and_return"].steps) == 3


def test_loaded_action_mappings_are_immutable() -> None:
    config = load_action_config()

    with pytest.raises(TypeError):
        config.actions["new"] = config.actions["home"]
    with pytest.raises(TypeError):
        config.actions["home"].steps[0].targets_rad["base"] = 0.5


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"pose": "missing", "duration_s": 1.0}, "unknown pose"),
        ({"targets_rad": {"gripper": 0.0}, "duration_s": 1.0}, "unknown joint"),
        ({"targets_rad": {"base": 2.0}, "duration_s": 1.0}, "outside"),
        ({"targets_rad": {"base": 0.0}, "duration_s": 0.0}, "greater than zero"),
        ({"targets_rad": {"base": 0.0}, "duration_s": float("inf")}, "finite"),
        (
            {"pose": "sim_home", "targets_rad": {"base": 0.0}, "duration_s": 1.0},
            "exactly one",
        ),
    ],
)
def test_rejects_invalid_action_steps(
    tmp_path: Path,
    step: dict,
    message: str,
) -> None:
    config_dir = write_actions(
        tmp_path,
        {"bad": {"description": "invalid test action", "steps": [step]}},
    )

    with pytest.raises(ValueError, match=message):
        load_action_config(config_dir, arm_config=load_arm_config())


def test_action_runner_forwards_targets_and_durations_in_order() -> None:
    arm = FakeArm()
    runner = ActionRunner(arm, load_action_config())

    runner.run("base_scan")

    assert arm.calls == [
        ({"base": 0.0}, 1.0),
        ({"base": -0.8}, 2.0),
        ({"base": 0.8}, 3.0),
        ({"base": 0.0}, 2.0),
    ]
    assert arm.positions["base"] == 0.0
    assert arm.positions["wrist"] == 0.0


def test_action_runner_rejects_unknown_name_before_motion() -> None:
    arm = FakeArm()
    runner = ActionRunner(arm, load_action_config())

    with pytest.raises(UnknownActionError, match="Unknown action"):
        runner.run("missing")
    assert arm.calls == []


def test_smoothstep_trajectory_has_expected_steps_and_endpoint() -> None:
    trajectory = list(
        _smoothstep_trajectory(
            {"base": 0.0},
            {"base": 1.0},
            duration_s=1.0,
            control_rate_hz=4.0,
        )
    )
    values = [pose["base"] for pose in trajectory]

    assert len(values) == 4
    assert values[-1] == pytest.approx(1.0)
    assert values == sorted(values)
    assert all(0.0 < value <= 1.0 for value in values)


def test_pybullet_move_without_duration_remains_non_blocking() -> None:
    arm, pb = make_fake_pybullet_arm()

    arm.move_joints({"base": 0.5})

    assert pb.step_calls == 0
    assert len(pb.control_calls) == 1
    assert pb.control_calls[0]["targetPosition"] == pytest.approx(0.5)


def test_pybullet_timed_move_steps_and_preserves_unspecified_joints() -> None:
    arm, pb = make_fake_pybullet_arm()

    arm.move_joint("wrist", 0.6, duration_s=0.025)

    assert pb.step_calls == 6
    assert len(pb.control_calls) == 6
    assert pb.control_calls[-1]["targetPosition"] == pytest.approx(0.6)
    assert {call["jointIndex"] for call in pb.control_calls} == {3}
    assert pb.states[0] == 0.0
    assert pb.states[1] == 0.0
    assert pb.states[2] == 0.0


@pytest.mark.parametrize("duration", [True, "1", 0.0, -1.0, float("nan"), float("inf")])
def test_pybullet_rejects_invalid_duration_before_command(duration) -> None:
    arm, pb = make_fake_pybullet_arm()

    with pytest.raises(ValueError, match="duration_s"):
        arm.move_joint("base", 0.5, duration_s=duration)
    assert pb.control_calls == []
    assert pb.step_calls == 0


def test_raspberry_pi_radians_backend_remains_unavailable() -> None:
    with pytest.raises(NotImplementedError, match="no measured PWM-to-angle"):
        RaspberryPiArm()
