"""Validate the YAML configuration files and the config loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from butter_finger.config import JOINT_NAMES, load_arm_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="module")
def joints_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "joints.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def poses_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "poses.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def geometry_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "geometry.yaml").read_text(encoding="utf-8"))


def test_confirmed_pwm_port_mapping(joints_cfg: dict) -> None:
    assert joints_cfg["physical"]["pwm_ports"] == {
        "base": 1,
        "shoulder": 3,
        "elbow": 4,
        "wrist": 5,
    }
    assert joints_cfg["physical"]["unused_ports"] == [2, 6]


def test_recorded_home_pwm(joints_cfg: dict) -> None:
    assert joints_cfg["physical"]["home_pwm_us"] == {
        "base": 1500,
        "shoulder": 2200,
        "elbow": 2490,
        "wrist": 1400,
    }


def test_tested_shoulder_pwm_range(joints_cfg: dict) -> None:
    limits = joints_cfg["physical"]["tested_pwm_limits_us"]
    assert limits["shoulder"] == {"min": 1200, "max": 2220}


def test_unknown_pwm_limits_stay_null(joints_cfg: dict) -> None:
    limits = joints_cfg["physical"]["tested_pwm_limits_us"]
    for joint in ("base", "elbow", "wrist"):
        assert limits[joint] == {"min": None, "max": None}, (
            f"{joint} PWM limits are unknown and must remain null"
        )


def test_shoulder_home_warning_present(joints_cfg: dict) -> None:
    warning = joints_cfg["physical"]["warnings"]["shoulder_home"]
    assert "2200" in warning
    assert "2220" in warning


def test_simulation_limits(joints_cfg: dict) -> None:
    limits = joints_cfg["simulation"]["limits_rad"]
    assert set(limits) == set(JOINT_NAMES)
    for joint, entry in limits.items():
        assert entry["lower"] == pytest.approx(-1.5708)
        assert entry["upper"] == pytest.approx(1.5708)


def test_sim_home_pose_is_neutral(poses_cfg: dict) -> None:
    home = poses_cfg["poses"]["sim_home"]
    assert set(home) == set(JOINT_NAMES)
    for joint, angle in home.items():
        assert angle == 0.0, f"sim_home {joint} must be the neutral reference (0 rad)"


def test_geometry_values_are_positive(geometry_cfg: dict) -> None:
    for key in (
        "base_height",
        "upper_arm_length",
        "forearm_length",
        "wrist_length",
        "link_width",
    ):
        assert geometry_cfg[key] > 0
    for name, mass in geometry_cfg["masses_kg"].items():
        assert mass > 0, f"mass of {name} must be positive"


def test_load_arm_config() -> None:
    config = load_arm_config()
    assert config.joint_order == JOINT_NAMES
    assert config.control_rate_hz == 240
    assert config.time_step_s == pytest.approx(1.0 / 240.0)
    assert set(config.home_pose) == set(JOINT_NAMES)
    for joint, angle in config.home_pose.items():
        assert config.sim_limits[joint].contains(angle)
