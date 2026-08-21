"""Camera-only Stage 0/1 hand tracking and virtual-target retargeting.

This package deliberately imports no arm backend. MediaPipe and OpenCV are
loaded only by the runtime adapters that need them.
"""
from __future__ import annotations

from butter_finger.teleoperation.buffering import LatestResultBuffer
from butter_finger.teleoperation.config import (
    TeleoperationConfig,
    load_teleoperation_config,
)
from butter_finger.teleoperation.features import extract_hand_features
from butter_finger.teleoperation.retargeting import (
    ExponentialTargetFilter,
    VirtualTargetController,
)
from butter_finger.teleoperation.types import (
    HandFeatures,
    HandObservation,
    HandTrackingResult,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)

__all__ = [
    "ExponentialTargetFilter",
    "HandFeatures",
    "HandObservation",
    "HandTrackingResult",
    "LatestResultBuffer",
    "TargetUpdate",
    "TeleoperationConfig",
    "TrackingState",
    "VirtualEETarget",
    "VirtualTargetController",
    "extract_hand_features",
    "load_teleoperation_config",
]
