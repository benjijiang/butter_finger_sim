"""Face detection and visual-servo tracking for the simulated arm.

Adds an "eyes" layer on top of the radians-only ArmBackend: read a camera
frame (the simulated wrist camera via ``PyBulletArm.capture_rgb()`` or a real
camera via OpenCV), detect the largest face, and drive the arm to keep that
face centered. Simulation / perception-development feature only — it commands
PyBulletArm and never talks to real servo hardware directly.

OpenCV is imported lazily inside the classes that need it, so importing this
package stays dependency-light and the test suite needs no camera.
"""
from __future__ import annotations

from butter_finger.perception.attention import FaceFollower, FollowStatus
from butter_finger.perception.config import TrackingConfig, load_tracking_config
from butter_finger.perception.detection import (
    Detection,
    FaceDetector,
    HaarFaceDetector,
    ScriptedFaceDetector,
)
from butter_finger.perception.tracker import FaceTracker, TrackerStep

__all__ = [
    "Detection",
    "FaceDetector",
    "FaceFollower",
    "FaceTracker",
    "FollowStatus",
    "HaarFaceDetector",
    "ScriptedFaceDetector",
    "TrackerStep",
    "TrackingConfig",
    "load_tracking_config",
]
