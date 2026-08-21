"""Hand tracking, virtual retargeting, and pure-math Stage 2 IK dry run.

This package deliberately imports no arm backend. Stage 2 stops at diagnostic
joint-angle candidates; MediaPipe and OpenCV are loaded only by the runtime
adapters that need them.
"""
from __future__ import annotations

from butter_finger.teleoperation.buffering import LatestResultBuffer
from butter_finger.teleoperation.config import (
    IKConfig,
    TeleoperationConfig,
    load_teleoperation_config,
)
from butter_finger.teleoperation.dry_run import DryRunTeleoperationController
from butter_finger.teleoperation.features import extract_hand_features
from butter_finger.teleoperation.kinematics import IKSolver, KinematicModel
from butter_finger.teleoperation.retargeting import (
    ExponentialTargetFilter,
    VirtualTargetController,
)
from butter_finger.teleoperation.types import (
    DryRunStep,
    EndEffectorPose,
    HandFeatures,
    HandObservation,
    HandTrackingResult,
    IKResult,
    IKStatus,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)

__all__ = [
    "DryRunStep",
    "DryRunTeleoperationController",
    "EndEffectorPose",
    "ExponentialTargetFilter",
    "HandFeatures",
    "HandObservation",
    "HandTrackingResult",
    "IKConfig",
    "IKResult",
    "IKSolver",
    "IKStatus",
    "KinematicModel",
    "LatestResultBuffer",
    "TargetUpdate",
    "TeleoperationConfig",
    "TrackingState",
    "VirtualEETarget",
    "VirtualTargetController",
    "extract_hand_features",
    "load_teleoperation_config",
]
