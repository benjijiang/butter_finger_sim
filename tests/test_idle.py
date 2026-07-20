"""Validate the simulation-only idle scan configuration and controller."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from butter_finger import (
    ArmBackend,
    IdleConfig,
    IdleController,
    load_arm_config,
    load_idle_config,
)


class FakeArm(ArmBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], float | None]] = []
        self.positions = {joint: 0.0 for joint in load_arm_config().joint_order}

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        pass

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
        pass

    def disconnect(self) -> None:
        pass


def make_idle_config() -> IdleConfig:
    return IdleConfig(
        pose_name="idle_ready",
        pose_rad={
            "base": 0.0,
            "shoulder": -0.3,
            "elbow": -0.5,
            "wrist": -1.0,
        },
        scan_joint="base",
        lower_rad=-0.5,
        upper_rad=0.5,
        half_cycle_s=4.0,
    )


def write_idle(tmp_path: Path, idle: dict) -> None:
    (tmp_path / "idle.yaml").write_text(
        yaml.safe_dump({"idle": idle}),
        encoding="utf-8",
    )


def test_loads_idle_configuration() -> None:
    config = load_idle_config()

    assert config.pose_name == "idle_ready"
    assert config.scan_joint == "base"
    assert config.lower_rad == pytest.approx(-1.5708)
    assert config.upper_rad == pytest.approx(1.5708)
    assert config.half_cycle_s == pytest.approx(5.0)
    assert config.scan_speed_rad_s == pytest.approx(0.62832)
    assert dict(config.pose_rad) == load_arm_config().poses["idle_ready"]

    with pytest.raises(TypeError):
        config.pose_rad["base"] = 0.1


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("pose", "missing", "unknown pose"),
        ("scan_joint", "gripper", "unknown joint"),
        ("lower_rad", -2.0, "outside"),
        ("upper_rad", -0.8, "lower_rad < upper_rad"),
        ("half_cycle_s", 0.0, "greater than zero"),
    ],
)
def test_rejects_invalid_idle_configuration(
    tmp_path: Path,
    key: str,
    value,
    message: str,
) -> None:
    idle = {
        "pose": "idle_ready",
        "scan_joint": "base",
        "lower_rad": -0.55,
        "upper_rad": 0.55,
        "half_cycle_s": 5.0,
    }
    idle[key] = value
    write_idle(tmp_path, idle)

    with pytest.raises(ValueError, match=message):
        load_idle_config(tmp_path, arm_config=load_arm_config())


def test_resume_restores_full_idle_pose_without_blocking() -> None:
    arm = FakeArm()
    config = make_idle_config()
    controller = IdleController(arm, config)

    controller.resume()

    assert arm.calls == [(dict(config.pose_rad), None)]
    assert controller.position_rad == 0.0
    assert controller.direction == 1


def test_update_scans_between_bounds_and_reverses() -> None:
    arm = FakeArm()
    config = make_idle_config()
    controller = IdleController(arm, config)

    controller.update(2.0)
    assert controller.position_rad == pytest.approx(0.5)
    assert controller.direction == -1

    controller.update(4.0)
    assert controller.position_rad == pytest.approx(-0.5)
    assert controller.direction == 1

    for _ in range(30):
        controller.update(0.7)
        assert config.lower_rad <= controller.position_rad <= config.upper_rad


def test_update_preserves_idle_posture_and_is_non_blocking() -> None:
    arm = FakeArm()
    config = make_idle_config()
    controller = IdleController(arm, config)

    controller.update(1.0)

    targets, duration_s = arm.calls[-1]
    assert duration_s is None
    assert targets["base"] == pytest.approx(0.25)
    assert targets["shoulder"] == -0.3
    assert targets["elbow"] == -0.5
    assert targets["wrist"] == -1.0


@pytest.mark.parametrize("dt_s", [True, "1", 0.0, -1.0, float("nan"), float("inf")])
def test_update_rejects_invalid_delta_time(dt_s) -> None:
    arm = FakeArm()
    controller = IdleController(arm, make_idle_config())

    with pytest.raises(ValueError, match="dt_s"):
        controller.update(dt_s)
    assert arm.calls == []
