"""Pure landmark math used by both Stage 0 and Stage 1."""
from __future__ import annotations

import math

from butter_finger.teleoperation.types import HandFeatures, HandObservation

PALM_INDICES = (0, 5, 9, 13, 17)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17


def _finite_landmarks(points: tuple[tuple[float, float, float], ...], label: str) -> None:
    if len(points) != 21:
        raise ValueError(f"{label} must contain exactly 21 landmarks")
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError(f"{label} must contain only finite coordinates")


def _distance_2d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _subtract(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def extract_hand_features(observation: HandObservation) -> HandFeatures:
    """Convert 21 image/world landmarks into retargeting features."""
    image = observation.image_landmarks
    world = observation.world_landmarks
    _finite_landmarks(image, "image_landmarks")
    _finite_landmarks(world, "world_landmarks")
    if not math.isfinite(observation.timestamp_s):
        raise ValueError("observation timestamp must be finite")
    if not math.isfinite(observation.confidence):
        raise ValueError("observation confidence must be finite")

    palm_scale = _distance_2d(image[INDEX_MCP], image[PINKY_MCP])
    if palm_scale <= 1e-9:
        raise ValueError("palm scale is zero")
    pinch_ratio = _distance_2d(image[THUMB_TIP], image[INDEX_TIP]) / palm_scale

    palm_u = sum(image[index][0] for index in PALM_INDICES) / len(PALM_INDICES)
    palm_v = sum(image[index][1] for index in PALM_INDICES) / len(PALM_INDICES)

    index_axis = _subtract(world[INDEX_MCP], world[WRIST])
    pinky_axis = _subtract(world[PINKY_MCP], world[WRIST])
    normal = _cross(index_axis, pinky_axis)
    if observation.handedness.strip().lower() == "left":
        normal = tuple(-value for value in normal)
    norm = math.sqrt(sum(value * value for value in normal))
    if norm <= 1e-12:
        raise ValueError("palm plane is degenerate")
    _, ny, nz = (value / norm for value in normal)
    palm_pitch = math.atan2(-ny, -nz)

    return HandFeatures(
        timestamp_s=observation.timestamp_s,
        hand_id=observation.hand_id,
        handedness=observation.handedness,
        confidence=observation.confidence,
        palm_u=palm_u,
        palm_v=palm_v,
        palm_scale=palm_scale,
        palm_pitch_rad=palm_pitch,
        pinch_ratio=pinch_ratio,
    )
