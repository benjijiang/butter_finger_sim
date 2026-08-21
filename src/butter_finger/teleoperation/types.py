"""Dependency-free data types for Stage 0/1 hand teleoperation."""
from __future__ import annotations

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
class TargetUpdate:
    """Observable result of one Stage 1 controller update."""

    state: TrackingState
    target: VirtualEETarget
    raw_target: VirtualEETarget
    owner_hand_id: str | None = None
    owner_handedness: str | None = None
    pinch_ratio: float | None = None
    clamped_axes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HandTrackingResult:
    """Newest asynchronous landmarker result and the frame it describes."""

    timestamp_ms: int
    observations: tuple[HandObservation, ...]
    frame_rgb: object
