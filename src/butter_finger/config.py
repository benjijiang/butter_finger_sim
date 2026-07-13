"""Load and validate the YAML configuration files.

Keeps the five configuration concepts separate (see config/joints.yaml):
simulation radians, recorded physical PWM microseconds, verified hardware
limits, temporary simulation limits, and named poses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
URDF_PATH = REPO_ROOT / "models" / "butter_finger_simple.urdf"

# Logical joint names used throughout the API. The URDF joint names are
# "<name>_joint" (see URDF_JOINT_NAMES).
JOINT_NAMES: tuple[str, ...] = ("base", "shoulder", "elbow", "wrist")
URDF_JOINT_NAMES: dict[str, str] = {name: f"{name}_joint" for name in JOINT_NAMES}


@dataclass(frozen=True)
class JointLimits:
    """Temporary simulation limits in radians. NOT physical calibration."""

    lower_rad: float
    upper_rad: float

    def contains(self, position_rad: float) -> bool:
        return self.lower_rad <= position_rad <= self.upper_rad


@dataclass(frozen=True)
class ArmConfig:
    joint_order: tuple[str, ...]
    sim_limits: dict[str, JointLimits]
    poses: dict[str, dict[str, float]]
    max_force_nm: float
    max_velocity_rad_s: float
    control_rate_hz: float

    @property
    def time_step_s(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def home_pose(self) -> dict[str, float]:
        return dict(self.poses["sim_home"])


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_arm_config(config_dir: Path = CONFIG_DIR) -> ArmConfig:
    """Load the simulation-facing configuration.

    Only reads the simulation sections; the recorded physical PWM data is
    historical calibration for the future Raspberry Pi backend and is
    deliberately not exposed here.
    """
    joints_cfg = _load_yaml(config_dir / "joints.yaml")
    poses_cfg = _load_yaml(config_dir / "poses.yaml")

    joint_order = tuple(joints_cfg["joint_order"])
    if set(joint_order) != set(JOINT_NAMES):
        raise ValueError(
            f"joint_order in joints.yaml {joint_order} does not match the "
            f"expected joints {JOINT_NAMES}"
        )

    limits_raw = joints_cfg["simulation"]["limits_rad"]
    sim_limits = {
        name: JointLimits(float(limits_raw[name]["lower"]), float(limits_raw[name]["upper"]))
        for name in joint_order
    }

    poses: dict[str, dict[str, float]] = {}
    for pose_name, pose in poses_cfg["poses"].items():
        poses[pose_name] = {joint: float(angle) for joint, angle in pose.items()}
        for joint, angle in poses[pose_name].items():
            if joint not in sim_limits:
                raise ValueError(f"Pose {pose_name!r} references unknown joint {joint!r}")
            if not sim_limits[joint].contains(angle):
                raise ValueError(
                    f"Pose {pose_name!r} puts {joint!r} at {angle} rad, outside "
                    f"the simulation limits"
                )

    control = joints_cfg["simulation"]["control"]
    return ArmConfig(
        joint_order=joint_order,
        sim_limits=sim_limits,
        poses=poses,
        max_force_nm=float(control["max_force_nm"]),
        max_velocity_rad_s=float(control["max_velocity_rad_s"]),
        control_rate_hz=float(control["control_rate_hz"]),
    )
