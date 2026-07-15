"""Load and validate the YAML configuration files.

Keeps the configuration concepts separate (see config/joints.yaml):
simulation radians, recorded physical PWM microseconds, verified hardware
limits, temporary simulation limits, named poses, and named actions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

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
class PulseLimits:
    """Tested hardware PWM pulse limits in microseconds."""

    min_us: int
    max_us: int

    def contains(self, pulse_us: float) -> bool:
        return self.min_us <= pulse_us <= self.max_us


@dataclass(frozen=True)
class PhysicalConfig:
    """The 'physical' section of config/joints.yaml: recorded hardware data.

    Used only by the PWM hardware layer on the Raspberry Pi; the simulator
    never reads these values.
    """

    joint_order: tuple[str, ...]
    pwm_ports: dict[str, int]
    home_pwm_us: dict[str, int]
    pulse_limits_us: dict[str, PulseLimits]


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


@dataclass(frozen=True)
class ActionStep:
    """One timed action step, expressed only in simulation radians."""

    targets_rad: Mapping[str, float]
    duration_s: float


@dataclass(frozen=True)
class ArmAction:
    """A named sequence of one or more timed joint targets."""

    name: str
    description: str
    steps: tuple[ActionStep, ...]


@dataclass(frozen=True)
class ActionConfig:
    """Validated named actions. These are not approved for real hardware."""

    actions: Mapping[str, ArmAction]

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(self.actions)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


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


def load_action_config(
    config_dir: Path = CONFIG_DIR,
    arm_config: ArmConfig | None = None,
) -> ActionConfig:
    """Load simulation-only named actions and resolve named pose references."""
    arm_config = arm_config if arm_config is not None else load_arm_config(config_dir)
    actions_cfg = _load_yaml(config_dir / "actions.yaml")
    if not isinstance(actions_cfg, dict) or not isinstance(actions_cfg.get("actions"), dict):
        raise ValueError("actions.yaml must contain an 'actions' mapping")

    actions_raw = actions_cfg["actions"]
    if not actions_raw:
        raise ValueError("actions.yaml must define at least one action")

    actions: dict[str, ArmAction] = {}
    for action_name, action_raw in actions_raw.items():
        if not isinstance(action_name, str) or not action_name:
            raise ValueError("Action names must be non-empty strings")
        if not isinstance(action_raw, dict):
            raise ValueError(f"Action {action_name!r} must be a mapping")

        unknown_action_keys = set(action_raw) - {"description", "steps"}
        if unknown_action_keys:
            raise ValueError(
                f"Action {action_name!r} has unknown keys "
                f"{sorted(unknown_action_keys)}"
            )

        description = action_raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"Action {action_name!r} description must be a string")

        steps_raw = action_raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError(f"Action {action_name!r} must contain at least one step")

        steps: list[ActionStep] = []
        for index, step_raw in enumerate(steps_raw, start=1):
            label = f"Action {action_name!r} step {index}"
            if not isinstance(step_raw, dict):
                raise ValueError(f"{label} must be a mapping")

            unknown_step_keys = set(step_raw) - {"pose", "targets_rad", "duration_s"}
            if unknown_step_keys:
                raise ValueError(f"{label} has unknown keys {sorted(unknown_step_keys)}")

            has_pose = "pose" in step_raw
            has_targets = "targets_rad" in step_raw
            if has_pose == has_targets:
                raise ValueError(
                    f"{label} must define exactly one of 'pose' or 'targets_rad'"
                )

            duration_s = _finite_float(step_raw.get("duration_s"), f"{label} duration_s")
            if duration_s <= 0:
                raise ValueError(f"{label} duration_s must be greater than zero")

            if has_pose:
                pose_name = step_raw["pose"]
                if not isinstance(pose_name, str) or pose_name not in arm_config.poses:
                    raise ValueError(f"{label} references unknown pose {pose_name!r}")
                targets = dict(arm_config.poses[pose_name])
            else:
                targets_raw = step_raw["targets_rad"]
                if not isinstance(targets_raw, dict) or not targets_raw:
                    raise ValueError(f"{label} targets_rad must be a non-empty mapping")
                targets = {}
                for joint, raw_angle in targets_raw.items():
                    if joint not in arm_config.sim_limits:
                        raise ValueError(f"{label} references unknown joint {joint!r}")
                    angle = _finite_float(raw_angle, f"{label} target for {joint!r}")
                    if not arm_config.sim_limits[joint].contains(angle):
                        raise ValueError(
                            f"{label} puts {joint!r} at {angle} rad, outside "
                            "the simulation limits"
                        )
                    targets[joint] = angle

            steps.append(
                ActionStep(
                    targets_rad=MappingProxyType(targets),
                    duration_s=duration_s,
                )
            )

        actions[action_name] = ArmAction(
            name=action_name,
            description=description,
            steps=tuple(steps),
        )

    return ActionConfig(actions=MappingProxyType(actions))


def load_physical_config(config_dir: Path = CONFIG_DIR) -> PhysicalConfig:
    """Load the recorded physical hardware data (PWM microseconds).

    This is the tested calibration record used by the PWM hardware layer
    (backends/pwm_robot_arm.py) on the Raspberry Pi. It is validated so a
    home pose outside the tested pulse limits fails at load time.
    """
    joints_cfg = _load_yaml(config_dir / "joints.yaml")
    physical = joints_cfg["physical"]

    joint_order = tuple(joints_cfg["joint_order"])
    if set(joint_order) != set(JOINT_NAMES):
        raise ValueError(
            f"joint_order in joints.yaml {joint_order} does not match the "
            f"expected joints {JOINT_NAMES}"
        )

    pwm_ports = {name: int(physical["pwm_ports"][name]) for name in joint_order}
    home_pwm_us = {name: int(physical["home_pwm_us"][name]) for name in joint_order}
    pulse_limits_us = {
        name: PulseLimits(
            int(physical["tested_pwm_limits_us"][name]["min"]),
            int(physical["tested_pwm_limits_us"][name]["max"]),
        )
        for name in joint_order
    }

    for name in joint_order:
        limits = pulse_limits_us[name]
        if limits.min_us >= limits.max_us:
            raise ValueError(f"{name}: invalid pulse limits {limits}")
        if not limits.contains(home_pwm_us[name]):
            raise ValueError(
                f"{name}: home pulse {home_pwm_us[name]} us is outside the "
                f"tested range [{limits.min_us}, {limits.max_us}] us"
            )

    return PhysicalConfig(
        joint_order=joint_order,
        pwm_ports=pwm_ports,
        home_pwm_us=home_pwm_us,
        pulse_limits_us=pulse_limits_us,
    )
