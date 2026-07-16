"""Validate named actions, timed motion, and backend-neutral scheduling."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from butter_finger import (
    ActionRunner,
    ArmBackend,
    UnknownActionError,
    load_action_config,
    load_arm_config,
    load_physical_config,
)
from butter_finger.backends.pybullet_arm import PyBulletArm, _smoothstep_trajectory
from examples.go_home import VISIBLE_OFFSET_POSE_RAD
from examples.scripted_motion import (
    BASE_TARGET_RAD,
    REACH_TARGETS_RAD,
    WRIST_TARGET_RAD,
)

EXPECTED_ACTIONS = (
    "home",
    "demo_reach",
    "base_scan",
    "reach_and_return",
    "wrist_up",
    "wrist_down",
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

EMOTION_ACTIONS = EXPECTED_ACTIONS[6:]


class FakeArm(ArmBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], float | None]] = []
        self.validations: list[dict[str, float]] = []
        self.positions = {joint: 0.0 for joint in load_arm_config().joint_order}

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
        "elbow": 0.065,
        "wrist": -1.571,
    }
    assert dict(config.actions["demo_reach"].steps[0].targets_rad) == {
        "base": 0.8,
        "shoulder": -0.6,
        "elbow": -0.9,
        "wrist": 0.15,
    }


def test_configured_development_action_values() -> None:
    actions = load_action_config().actions

    assert [step.targets_rad["base"] for step in actions["base_scan"].steps] == [
        0.0,
        -0.8,
        0.8,
        0.0,
    ]
    assert actions["wrist_up"].steps[0].targets_rad["wrist"] == 0.15
    assert actions["wrist_down"].steps[0].targets_rad["wrist"] == -0.6
    assert len(actions["reach_and_return"].steps) == 3


def test_emotional_actions_begin_and_end_at_idle_ready() -> None:
    config = load_action_config()
    idle_ready = load_arm_config().poses["idle_ready"]

    assert len(EMOTION_ACTIONS) == 20
    for name in EMOTION_ACTIONS:
        action = config.actions[name]
        assert dict(action.steps[0].targets_rad) == idle_ready, name
        assert dict(action.steps[-1].targets_rad) == idle_ready, name


def test_emotional_action_targets_stay_within_simulation_limits() -> None:
    arm_config = load_arm_config()
    actions = load_action_config(arm_config=arm_config).actions

    for name in EMOTION_ACTIONS:
        for step in actions[name].steps:
            for joint, angle in step.targets_rad.items():
                assert arm_config.sim_limits[joint].contains(angle), (name, joint, angle)


def test_all_action_targets_stay_within_physical_calibration() -> None:
    actions = load_action_config().actions
    calibrations = load_physical_config().calibrations

    for name, action in actions.items():
        for step in action.steps:
            for joint, angle in step.targets_rad.items():
                assert calibrations[joint].contains_rad(angle), (name, joint, angle)


def test_example_targets_stay_within_physical_calibration() -> None:
    calibrations = load_physical_config().calibrations
    targets = [
        VISIBLE_OFFSET_POSE_RAD,
        {"base": BASE_TARGET_RAD},
        REACH_TARGETS_RAD,
        {"wrist": WRIST_TARGET_RAD},
    ]

    for target in targets:
        for joint, angle in target.items():
            assert calibrations[joint].contains_rad(angle), (joint, angle)


def test_emotional_actions_have_distinct_timing_signatures() -> None:
    actions = load_action_config().actions

    assert actions["surprised"].steps[1].duration_s < 0.35
    assert min(step.duration_s for step in actions["scared"].steps) <= 0.1
    assert max(step.duration_s for step in actions["thinking"].steps) >= 2.0
    assert max(step.duration_s for step in actions["sad"].steps) >= 2.5

    confused = actions["confused"].steps
    assert confused[1].duration_s != confused[3].duration_s

    sleepy_droops = (
        actions["sleepy"].steps[1].duration_s,
        actions["sleepy"].steps[3].duration_s,
        actions["sleepy"].steps[5].duration_s,
    )
    assert sleepy_droops == tuple(sorted(sleepy_droops))


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
    assert arm.validations == [call[0] for call in arm.calls]


def test_action_runner_rejects_unknown_name_before_motion() -> None:
    arm = FakeArm()
    runner = ActionRunner(arm, load_action_config())

    with pytest.raises(UnknownActionError, match="Unknown action"):
        runner.run("missing")
    assert arm.calls == []


def test_action_runner_prechecks_every_step_before_motion() -> None:
    class RejectingArm(FakeArm):
        def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
            super().validate_targets(targets_rad)
            if targets_rad.get("base") == 0.8:
                raise ValueError("unsafe final target")

    arm = RejectingArm()
    runner = ActionRunner(arm, load_action_config())

    with pytest.raises(ValueError, match="unsafe final target"):
        runner.run("base_scan")
    assert arm.calls == []
    assert arm.validations == [
        {"base": 0.0},
        {"base": -0.8},
        {"base": 0.8},
    ]


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

    arm.move_joint("wrist", 0.15, duration_s=0.025)

    assert pb.step_calls == 6
    assert len(pb.control_calls) == 6
    assert pb.control_calls[-1]["targetPosition"] == pytest.approx(0.15)
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
