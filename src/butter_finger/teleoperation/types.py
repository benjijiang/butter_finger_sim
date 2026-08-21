"""Dependency-free data types for hand-teleoperation stages."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

Landmark3D = tuple[float, float, float]


class TrackingState(str, Enum):
    """Operator-input state. None of these states commands an arm."""

    NO_HAND = "NO_HAND"
    HOVER = "HOVER"
    CLUTCHED = "CLUTCHED"
    HOLD = "HOLD"
    LOST = "LOST"


@dataclass(frozen=True)
class HandObservation:
    """One MediaPipe hand result in logical (optionally mirrored) image space."""

    timestamp_s: float
    hand_id: str
    handedness: str
    confidence: float
    image_landmarks: tuple[Landmark3D, ...]
    world_landmarks: tuple[Landmark3D, ...]


@dataclass(frozen=True)
class HandFeatures:
    """Retargeting features extracted from one 21-landmark observation."""

    timestamp_s: float
    hand_id: str
    handedness: str
    confidence: float
    palm_u: float
    palm_v: float
    palm_scale: float
    palm_pitch_rad: float
    pinch_ratio: float


@dataclass(frozen=True)
class VirtualEETarget:
    """Stage 1 virtual target; meters and radians, never sent to an arm."""

    x_m: float
    y_m: float
    z_m: float
    pitch_rad: float
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class EndEffectorPose:
    """Camera base-link position and signed fixed-task-frame pitch."""

    x_m: float
    y_m: float
    z_m: float
    pitch_rad: float


class IKStatus(str, Enum):
    """Outcome of solving one Stage 2 target.

    ``UNREACHABLE`` is reserved for conservative kinematic rejection;
    invalid inputs and ordinary solver non-convergence are numerical failures.
    """

    SOLVED = "SOLVED"
    UNREACHABLE = "UNREACHABLE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True)
class IKResult:
    """Raw inverse-kinematics result before dry-run slew limiting."""

    status: IKStatus
    target_pose: EndEffectorPose
    achieved_pose: EndEffectorPose
    joints_rad: dict[str, float] | None
    position_error_m: float
    pitch_error_rad: float
    iterations: int
    active_limits: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        solved = self.status is IKStatus.SOLVED
        has_candidate = isinstance(self.joints_rad, dict) and bool(self.joints_rad)
        if solved != has_candidate:
            raise ValueError("joints_rad must be non-empty exactly when IK is SOLVED")
        if solved and self.joints_rad is not None and any(
            not math.isfinite(value) for value in self.joints_rad.values()
        ):
            raise ValueError("SOLVED joint candidates must be finite")


@dataclass(frozen=True)
class TargetUpdate:
    """Observable result of one Stage 1 controller update.

    ``motion_eligible`` means this frame processed the owning hand without a
    release debounce; it is an input-validity gate, not a hardware safety flag.
    """

    state: TrackingState
    target: VirtualEETarget
    raw_target: VirtualEETarget
    owner_hand_id: str | None = None
    owner_handedness: str | None = None
    pinch_ratio: float | None = None
    clamped_axes: tuple[str, ...] = field(default_factory=tuple)
    motion_eligible: bool = False


@dataclass(frozen=True)
class HandTrackingResult:
    """Newest asynchronous landmarker result and the frame it describes."""

    timestamp_ms: int
    observations: tuple[HandObservation, ...]
    frame_rgb: object


@dataclass(frozen=True)
class DryRunStep:
    """One Stage 2 diagnostic step; no value here is sent to hardware."""

    target_update: TargetUpdate
    desired_pose: EndEffectorPose
    ik_result: IKResult | None
    output_joints_rad: dict[str, float]
    output_pose: EndEffectorPose
    output_position_error_m: float
    output_pitch_error_rad: float
    moved: bool
    slewing: bool
    hold_reason: str | None = None
