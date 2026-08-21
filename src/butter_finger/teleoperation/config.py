"""Validated loader for config/teleoperation.yaml."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from butter_finger.config import CONFIG_DIR, JOINT_NAMES
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
class IKConfig:
    """Pure-math Stage 2 solver and provisional dry-run slew parameters."""

    anchor_pose: str
    end_effector: str
    position_tolerance_m: float
    pitch_tolerance_rad: float
    orientation_weight_m_per_rad: float
    min_forward_component: float
    max_iterations: int
    finite_difference_step_rad: float
    max_iteration_step_rad: float
    initial_damping: float
    min_damping: float
    max_damping: float
    max_backtracking_steps: int
    max_solution_jump_rad: float
    max_slew_dt_s: float
    joint_rate_limits_rad_s: dict[str, float]


@dataclass(frozen=True)
class TeleoperationConfig:
    webcam: WebcamConfig
    hand_tracking: HandTrackingConfig
    clutch: ClutchConfig
    mapping: MappingConfig
    filter_cutoff_hz: float
    workspace: WorkspaceConfig
    ik: IKConfig | None = None


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


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
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


def _load_ik_config(teleop: dict[str, Any]) -> IKConfig:
    """Load Stage 2-only values; Stage 0/1 callers may skip this section."""
    ik_raw = _mapping(teleop.get("ik"), "teleoperation.ik")
    rates_raw = _mapping(
        ik_raw.get("joint_rate_limits_rad_s"),
        "teleoperation.ik.joint_rate_limits_rad_s",
    )
    if set(rates_raw) != set(JOINT_NAMES):
        raise ValueError(
            "teleoperation.ik.joint_rate_limits_rad_s must contain exactly "
            f"{list(JOINT_NAMES)}"
        )

    min_forward_component = _number(
        ik_raw.get("min_forward_component"),
        "teleoperation.ik.min_forward_component",
    )
    if not 0.0 < min_forward_component < 1.0:
        raise ValueError(
            "teleoperation.ik.min_forward_component must be between 0 and 1"
        )
    min_damping = _positive(
        ik_raw.get("min_damping"), "teleoperation.ik.min_damping"
    )
    initial_damping = _positive(
        ik_raw.get("initial_damping"), "teleoperation.ik.initial_damping"
    )
    max_damping = _positive(
        ik_raw.get("max_damping"), "teleoperation.ik.max_damping"
    )
    if not min_damping <= initial_damping <= max_damping:
        raise ValueError(
            "teleoperation.ik damping must satisfy "
            "min_damping <= initial_damping <= max_damping"
        )

    return IKConfig(
        anchor_pose=_nonempty_string(
            ik_raw.get("anchor_pose"), "teleoperation.ik.anchor_pose"
        ),
        end_effector=_nonempty_string(
            ik_raw.get("end_effector"), "teleoperation.ik.end_effector"
        ),
        position_tolerance_m=_positive(
            ik_raw.get("position_tolerance_m"),
            "teleoperation.ik.position_tolerance_m",
        ),
        pitch_tolerance_rad=_positive(
            ik_raw.get("pitch_tolerance_rad"),
            "teleoperation.ik.pitch_tolerance_rad",
        ),
        orientation_weight_m_per_rad=_positive(
            ik_raw.get("orientation_weight_m_per_rad"),
            "teleoperation.ik.orientation_weight_m_per_rad",
        ),
        min_forward_component=min_forward_component,
        max_iterations=_positive_int(
            ik_raw.get("max_iterations"), "teleoperation.ik.max_iterations"
        ),
        finite_difference_step_rad=_positive(
            ik_raw.get("finite_difference_step_rad"),
            "teleoperation.ik.finite_difference_step_rad",
        ),
        max_iteration_step_rad=_positive(
            ik_raw.get("max_iteration_step_rad"),
            "teleoperation.ik.max_iteration_step_rad",
        ),
        initial_damping=initial_damping,
        min_damping=min_damping,
        max_damping=max_damping,
        max_backtracking_steps=_positive_int(
            ik_raw.get("max_backtracking_steps"),
            "teleoperation.ik.max_backtracking_steps",
        ),
        max_solution_jump_rad=_positive(
            ik_raw.get("max_solution_jump_rad"),
            "teleoperation.ik.max_solution_jump_rad",
        ),
        max_slew_dt_s=_positive(
            ik_raw.get("max_slew_dt_s"), "teleoperation.ik.max_slew_dt_s"
        ),
        joint_rate_limits_rad_s={
            joint: _positive(
                rates_raw.get(joint),
                f"teleoperation.ik.joint_rate_limits_rad_s.{joint}",
            )
            for joint in JOINT_NAMES
        },
    )


def load_teleoperation_config(
    path: Path = TELEOPERATION_CONFIG_PATH,
    *,
    include_ik: bool | None = None,
) -> TeleoperationConfig:
    """Load teleoperation input and optionally require Stage 2 IK settings.

    The default auto-loads IK when the section exists while retaining support
    for legacy Stage 0/1 files. Passing ``True`` requires IK; passing ``False``
    deliberately skips it.
    """
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

    ik = (
        _load_ik_config(teleop)
        if include_ik is True or (include_ik is None and "ik" in teleop)
        else None
    )

    return TeleoperationConfig(
        webcam=webcam,
        hand_tracking=tracking,
        clutch=clutch,
        mapping=mapping,
        filter_cutoff_hz=_positive(
            filter_raw.get("cutoff_hz"), "teleoperation.filter.cutoff_hz"
        ),
        workspace=workspace,
        ik=ik,
    )
