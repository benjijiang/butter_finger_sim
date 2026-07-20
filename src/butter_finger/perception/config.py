"""Load and validate config/tracking.yaml (face-tracking control knobs).

Kept in the perception package, separate from the core ``butter_finger.config``
loaders, so this simulation-only feature adds no surface to the shared config
module. All values are development-tuning knobs, not physical calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from butter_finger.config import CONFIG_DIR, JOINT_NAMES

TRACKING_CONFIG_PATH = CONFIG_DIR / "tracking.yaml"


@dataclass(frozen=True)
class TrackingConfig:
    """Proportional visual-servo knobs. Signs/gains are NOT hardware-safe.

    The tracker drives three joints: ``pan_joint`` (yaw) follows horizontal
    face error, ``tilt_joint`` (pitch) follows vertical error, and
    ``distance_joint`` adjusts stand-off from the apparent face size.
    """

    pan_joint: str
    tilt_joint: str
    distance_joint: str
    start_pose: str
    gain_pan: float
    gain_tilt: float
    gain_distance: float
    sign_pan: float
    sign_tilt: float
    sign_distance: float
    deadband_xy: float
    deadband_distance: float
    max_step_rad: float
    target_face_fraction: float


def _num(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"tracking.{label} must be a number")
    return float(value)


def load_tracking_config(path: Path = TRACKING_CONFIG_PATH) -> TrackingConfig:
    """Load config/tracking.yaml into a validated TrackingConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("tracking"), dict):
        raise ValueError("tracking.yaml must contain a 'tracking' mapping")
    trk = raw["tracking"]

    config = TrackingConfig(
        pan_joint=str(trk["pan_joint"]),
        tilt_joint=str(trk["tilt_joint"]),
        distance_joint=str(trk["distance_joint"]),
        start_pose=str(trk["start_pose"]),
        gain_pan=_num(trk["gain_pan"], "gain_pan"),
        gain_tilt=_num(trk["gain_tilt"], "gain_tilt"),
        gain_distance=_num(trk["gain_distance"], "gain_distance"),
        sign_pan=_num(trk["sign_pan"], "sign_pan"),
        sign_tilt=_num(trk["sign_tilt"], "sign_tilt"),
        sign_distance=_num(trk["sign_distance"], "sign_distance"),
        deadband_xy=_num(trk["deadband_xy"], "deadband_xy"),
        deadband_distance=_num(trk["deadband_distance"], "deadband_distance"),
        max_step_rad=_num(trk["max_step_rad"], "max_step_rad"),
        target_face_fraction=_num(trk["target_face_fraction"], "target_face_fraction"),
    )

    for role, joint in (
        ("pan_joint", config.pan_joint),
        ("tilt_joint", config.tilt_joint),
        ("distance_joint", config.distance_joint),
    ):
        if joint not in JOINT_NAMES:
            raise ValueError(
                f"tracking.{role} {joint!r} is not one of the arm joints {JOINT_NAMES}"
            )
    if config.max_step_rad <= 0:
        raise ValueError("tracking.max_step_rad must be positive")
    if not 0.0 < config.target_face_fraction < 1.0:
        raise ValueError("tracking.target_face_fraction must be in (0, 1)")
    if config.deadband_xy < 0 or config.deadband_distance < 0:
        raise ValueError("tracking deadbands must be non-negative")

    return config
