"""Validated loader for config/teleoperation.yaml."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from butter_finger.config import CONFIG_DIR
from butter_finger.teleoperation.types import VirtualEETarget

TELEOPERATION_CONFIG_PATH = CONFIG_DIR / "teleoperation.yaml"


@dataclass(frozen=True)
class RangeLimit:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("workspace limits must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("workspace limits must satisfy min < max")

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, value))

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class WebcamConfig:
    width: int
    height: int
    fps: int
    mirror: bool


@dataclass(frozen=True)
class HandTrackingConfig:
    num_hands: int
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float
    max_result_age_s: float
    owner_match_distance: float


@dataclass(frozen=True)
class ClutchConfig:
    engage_ratio: float
    release_ratio: float
    debounce_frames: int
    lost_timeout_s: float


@dataclass(frozen=True)
class MappingConfig:
    gain_x_m: float
    gain_y_m: float
    gain_z_m: float
    gain_pitch: float


@dataclass(frozen=True)
class WorkspaceConfig:
    x_m: RangeLimit
    y_m: RangeLimit
    z_m: RangeLimit
    pitch_rad: RangeLimit
    initial_target: VirtualEETarget


@dataclass(frozen=True)
class TeleoperationConfig:
    webcam: WebcamConfig
    hand_tracking: HandTrackingConfig
    clutch: ClutchConfig
    mapping: MappingConfig
    filter_cutoff_hz: float
    workspace: WorkspaceConfig


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _probability(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _range(value: Any, label: str) -> RangeLimit:
    raw = _mapping(value, label)
    return RangeLimit(
        _number(raw.get("min"), f"{label}.min"),
        _number(raw.get("max"), f"{label}.max"),
    )


def load_teleoperation_config(
    path: Path = TELEOPERATION_CONFIG_PATH,
) -> TeleoperationConfig:
    """Load Stage 0/1 knobs without importing OpenCV or MediaPipe."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "teleoperation.yaml")
    teleop = _mapping(root.get("teleoperation"), "teleoperation")
    webcam_raw = _mapping(teleop.get("webcam"), "teleoperation.webcam")
    tracking_raw = _mapping(
        teleop.get("hand_tracking"), "teleoperation.hand_tracking"
    )
    clutch_raw = _mapping(teleop.get("clutch"), "teleoperation.clutch")
    mapping_raw = _mapping(teleop.get("mapping"), "teleoperation.mapping")
    filter_raw = _mapping(teleop.get("filter"), "teleoperation.filter")
    workspace_raw = _mapping(
        teleop.get("provisional_workspace"),
        "teleoperation.provisional_workspace",
    )
    initial_raw = _mapping(
        workspace_raw.get("initial_target"),
        "teleoperation.provisional_workspace.initial_target",
    )

    mirror = webcam_raw.get("mirror")
    if not isinstance(mirror, bool):
        raise ValueError("teleoperation.webcam.mirror must be a boolean")
    webcam = WebcamConfig(
        width=_positive_int(webcam_raw.get("width"), "teleoperation.webcam.width"),
        height=_positive_int(webcam_raw.get("height"), "teleoperation.webcam.height"),
        fps=_positive_int(webcam_raw.get("fps"), "teleoperation.webcam.fps"),
        mirror=mirror,
    )
    tracking = HandTrackingConfig(
        num_hands=_positive_int(
            tracking_raw.get("num_hands"), "teleoperation.hand_tracking.num_hands"
        ),
        min_detection_confidence=_probability(
            tracking_raw.get("min_detection_confidence"),
            "teleoperation.hand_tracking.min_detection_confidence",
        ),
        min_presence_confidence=_probability(
            tracking_raw.get("min_presence_confidence"),
            "teleoperation.hand_tracking.min_presence_confidence",
        ),
        min_tracking_confidence=_probability(
            tracking_raw.get("min_tracking_confidence"),
            "teleoperation.hand_tracking.min_tracking_confidence",
        ),
        max_result_age_s=_positive(
            tracking_raw.get("max_result_age_s"),
            "teleoperation.hand_tracking.max_result_age_s",
        ),
        owner_match_distance=_positive(
            tracking_raw.get("owner_match_distance"),
            "teleoperation.hand_tracking.owner_match_distance",
        ),
    )
    if tracking.num_hands > 2:
        raise ValueError("teleoperation.hand_tracking.num_hands must be at most 2")
    clutch = ClutchConfig(
        engage_ratio=_positive(
            clutch_raw.get("engage_ratio"), "teleoperation.clutch.engage_ratio"
        ),
        release_ratio=_positive(
            clutch_raw.get("release_ratio"), "teleoperation.clutch.release_ratio"
        ),
        debounce_frames=_positive_int(
            clutch_raw.get("debounce_frames"),
            "teleoperation.clutch.debounce_frames",
        ),
        lost_timeout_s=_positive(
            clutch_raw.get("lost_timeout_s"),
            "teleoperation.clutch.lost_timeout_s",
        ),
    )
    if clutch.engage_ratio >= clutch.release_ratio:
        raise ValueError("clutch engage_ratio must be less than release_ratio")

    mapping = MappingConfig(
        gain_x_m=_positive(mapping_raw.get("gain_x_m"), "mapping.gain_x_m"),
        gain_y_m=_positive(mapping_raw.get("gain_y_m"), "mapping.gain_y_m"),
        gain_z_m=_positive(mapping_raw.get("gain_z_m"), "mapping.gain_z_m"),
        gain_pitch=_positive(mapping_raw.get("gain_pitch"), "mapping.gain_pitch"),
    )
    workspace = WorkspaceConfig(
        x_m=_range(workspace_raw.get("x_m"), "workspace.x_m"),
        y_m=_range(workspace_raw.get("y_m"), "workspace.y_m"),
        z_m=_range(workspace_raw.get("z_m"), "workspace.z_m"),
        pitch_rad=_range(workspace_raw.get("pitch_rad"), "workspace.pitch_rad"),
        initial_target=VirtualEETarget(
            x_m=_number(initial_raw.get("x_m"), "initial_target.x_m"),
            y_m=_number(initial_raw.get("y_m"), "initial_target.y_m"),
            z_m=_number(initial_raw.get("z_m"), "initial_target.z_m"),
            pitch_rad=_number(
                initial_raw.get("pitch_rad"), "initial_target.pitch_rad"
            ),
        ),
    )
    for axis, value, limit in (
        ("x_m", workspace.initial_target.x_m, workspace.x_m),
        ("y_m", workspace.initial_target.y_m, workspace.y_m),
        ("z_m", workspace.initial_target.z_m, workspace.z_m),
        ("pitch_rad", workspace.initial_target.pitch_rad, workspace.pitch_rad),
    ):
        if not limit.contains(value):
            raise ValueError(f"initial_target.{axis} is outside workspace.{axis}")

    return TeleoperationConfig(
        webcam=webcam,
        hand_tracking=tracking,
        clutch=clutch,
        mapping=mapping,
        filter_cutoff_hz=_positive(
            filter_raw.get("cutoff_hz"), "teleoperation.filter.cutoff_hz"
        ),
        workspace=workspace,
    )
