"""Validated Stage 3 link/control configuration and profile fingerprint."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from butter_finger.config import (
    CONFIG_DIR,
    JOINT_NAMES,
    ArmConfig,
    load_arm_config,
)

REMOTE_TELEOPERATION_CONFIG_PATH = CONFIG_DIR / "remote_teleoperation.yaml"


@dataclass(frozen=True)
class RemoteTeleoperationConfig:
    protocol_version: int
    bind_host: str
    port: int
    publish_rate_hz: float
    mac_control_timeout_s: float
    watchdog_timeout_s: float
    connect_timeout_s: float
    socket_timeout_s: float
    status_timeout_s: float
    max_frame_bytes: int
    startup_pose: str
    startup_duration_s: float
    stream_duration_s: float
    max_slew_dt_s: float
    command_epsilon_rad: float
    command_envelope: str
    mac_joint_rate_limits_rad_s: dict[str, float]
    pi_joint_rate_limits_rad_s: dict[str, float]

    @property
    def publish_period_s(self) -> float:
        return 1.0 / self.publish_rate_hz


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _joint_rates(value: Any, label: str) -> dict[str, float]:
    raw = _mapping(value, label)
    if set(raw) != set(JOINT_NAMES):
        raise ValueError(f"{label} must contain exactly {list(JOINT_NAMES)}")
    return {
        joint: _number(raw[joint], f"{label}.{joint}") for joint in JOINT_NAMES
    }


def load_remote_teleoperation_config(
    path: Path = REMOTE_TELEOPERATION_CONFIG_PATH,
) -> RemoteTeleoperationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "remote_teleoperation.yaml")
    remote = _mapping(root.get("remote_teleoperation"), "remote_teleoperation")
    endpoint = _mapping(remote.get("endpoint"), "remote_teleoperation.endpoint")
    link = _mapping(remote.get("link"), "remote_teleoperation.link")
    control = _mapping(remote.get("control"), "remote_teleoperation.control")

    protocol_version = _positive_int(
        remote.get("protocol_version"), "remote_teleoperation.protocol_version"
    )
    if protocol_version != 1:
        raise ValueError("only remote teleoperation protocol_version 1 is supported")

    port = _positive_int(endpoint.get("port"), "remote_teleoperation.endpoint.port")
    if port > 65535:
        raise ValueError("remote_teleoperation.endpoint.port must be <= 65535")

    max_frame_bytes = _positive_int(
        link.get("max_frame_bytes"),
        "remote_teleoperation.link.max_frame_bytes",
    )
    if max_frame_bytes < 512:
        raise ValueError("remote teleoperation max_frame_bytes must be at least 512")

    envelope = _string(
        control.get("command_envelope"),
        "remote_teleoperation.control.command_envelope",
    )
    if envelope != "calibrated":
        raise ValueError("Stage 3 v1 command_envelope must be 'calibrated'")

    config = RemoteTeleoperationConfig(
        protocol_version=protocol_version,
        bind_host=_string(
            endpoint.get("bind_host"), "remote_teleoperation.endpoint.bind_host"
        ),
        port=port,
        publish_rate_hz=_number(
            link.get("publish_rate_hz"),
            "remote_teleoperation.link.publish_rate_hz",
        ),
        mac_control_timeout_s=_number(
            link.get("mac_control_timeout_s"),
            "remote_teleoperation.link.mac_control_timeout_s",
        ),
        watchdog_timeout_s=_number(
            link.get("watchdog_timeout_s"),
            "remote_teleoperation.link.watchdog_timeout_s",
        ),
        connect_timeout_s=_number(
            link.get("connect_timeout_s"),
            "remote_teleoperation.link.connect_timeout_s",
        ),
        socket_timeout_s=_number(
            link.get("socket_timeout_s"),
            "remote_teleoperation.link.socket_timeout_s",
        ),
        status_timeout_s=_number(
            link.get("status_timeout_s"),
            "remote_teleoperation.link.status_timeout_s",
        ),
        max_frame_bytes=max_frame_bytes,
        startup_pose=_string(
            control.get("startup_pose"),
            "remote_teleoperation.control.startup_pose",
        ),
        startup_duration_s=_number(
            control.get("startup_duration_s"),
            "remote_teleoperation.control.startup_duration_s",
        ),
        stream_duration_s=_number(
            control.get("stream_duration_s"),
            "remote_teleoperation.control.stream_duration_s",
        ),
        max_slew_dt_s=_number(
            control.get("max_slew_dt_s"),
            "remote_teleoperation.control.max_slew_dt_s",
        ),
        command_epsilon_rad=_number(
            control.get("command_epsilon_rad"),
            "remote_teleoperation.control.command_epsilon_rad",
            allow_zero=True,
        ),
        command_envelope=envelope,
        mac_joint_rate_limits_rad_s=_joint_rates(
            control.get("mac_joint_rate_limits_rad_s"),
            "remote_teleoperation.control.mac_joint_rate_limits_rad_s",
        ),
        pi_joint_rate_limits_rad_s=_joint_rates(
            control.get("pi_joint_rate_limits_rad_s"),
            "remote_teleoperation.control.pi_joint_rate_limits_rad_s",
        ),
    )
    if config.watchdog_timeout_s <= config.publish_period_s:
        raise ValueError("watchdog_timeout_s must exceed one publish period")
    if config.mac_control_timeout_s >= config.watchdog_timeout_s:
        raise ValueError("mac_control_timeout_s must be lower than Pi watchdog_timeout_s")
    if config.status_timeout_s <= config.watchdog_timeout_s:
        raise ValueError("status_timeout_s must exceed watchdog_timeout_s")
    for joint in JOINT_NAMES:
        if (
            config.mac_joint_rate_limits_rad_s[joint]
            >= config.pi_joint_rate_limits_rad_s[joint]
        ):
            raise ValueError(
                "each Mac joint rate must be lower than the Pi safety rate"
            )
    return config


def compute_profile_sha256(
    config: RemoteTeleoperationConfig,
    arm_config: ArmConfig | None = None,
) -> str:
    """Fingerprint protocol/control settings and the shared radian model."""
    arm = arm_config if arm_config is not None else load_arm_config()
    if config.startup_pose not in arm.poses:
        raise ValueError(f"unknown Stage 3 startup pose {config.startup_pose!r}")
    profile = {
        "protocol_version": config.protocol_version,
        "joint_order": list(arm.joint_order),
        "limits_rad": {
            joint: [
                arm.sim_limits[joint].lower_rad,
                arm.sim_limits[joint].upper_rad,
            ]
            for joint in arm.joint_order
        },
        "startup_pose": config.startup_pose,
        "startup_joints_rad": {
            joint: arm.poses[config.startup_pose][joint]
            for joint in arm.joint_order
        },
        "publish_rate_hz": config.publish_rate_hz,
        "mac_control_timeout_s": config.mac_control_timeout_s,
        "watchdog_timeout_s": config.watchdog_timeout_s,
        "connect_timeout_s": config.connect_timeout_s,
        "socket_timeout_s": config.socket_timeout_s,
        "status_timeout_s": config.status_timeout_s,
        "max_frame_bytes": config.max_frame_bytes,
        "startup_duration_s": config.startup_duration_s,
        "stream_duration_s": config.stream_duration_s,
        "max_slew_dt_s": config.max_slew_dt_s,
        "command_epsilon_rad": config.command_epsilon_rad,
        "command_envelope": config.command_envelope,
        "mac_rates": config.mac_joint_rate_limits_rad_s,
        "pi_rates": config.pi_joint_rate_limits_rad_s,
    }
    canonical = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
