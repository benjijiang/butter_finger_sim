#!/usr/bin/env python3
"""REAL HARDWARE: the physical arm follows a face with the Hiwonder camera.

Run ONLY on the Raspberry Pi 5 connected to the RasAdapter5A, with the
Hiwonder USB camera plugged in. This is the real-hardware counterpart to
examples/face_tracking.py: the SAME perception and control code
(FaceFollower / FaceTracker / IdleController / HaarFaceDetector /
WebcamSource) driving RaspberryPiArm instead of PyBulletArm.

    python examples/real_face_tracking.py --confirm-hardware

The arm pans with the base, tilts with the wrist, and adjusts stand-off with
the shoulder from the apparent face size. When no face is seen for a short
grace period it hands off to the idle base scan to look for one, then
reacquires.

Safety:
  * --confirm-hardware is REQUIRED before anything moves (see CLAUDE.md).
  * The arm eases to its exact recorded home, then to the tracking start
    pose, before the tracking loop begins.
  * Ctrl-C returns the arm home and releases the camera cleanly.

Prerequisites on the Pi:
  * pip install opencv-python  (only the sim machine had it before)
  * the Hiwonder camera enumerated as /dev/video0 (see --camera-index)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import (
    BackendUnavailableError,
    IdleController,
    RaspberryPiArm,
)
from butter_finger.config import load_arm_config
from butter_finger.perception import (
    FaceFollower,
    FaceTracker,
    HaarFaceDetector,
    load_tracking_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required acknowledgement that this moves the real arm",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="camera device index (/dev/video<index>); default 0",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help="control loop rate in Hz (default 30)",
    )
    args = parser.parse_args()

    if not args.confirm_hardware:
        print(
            "Refusing to move the real arm without --confirm-hardware.",
            file=sys.stderr,
        )
        return 2
    if args.rate_hz <= 0:
        print("--rate-hz must be greater than zero.", file=sys.stderr)
        return 2

    # ArmConfig (radians limits + named poses) for the tracker. This is NOT
    # RaspberryPiArm.config, which is a PhysicalConfig (calibrations/PWM).
    arm_config = load_arm_config()
    tracking_cfg = load_tracking_config()

    try:
        arm = RaspberryPiArm()
    except BackendUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Open the Hiwonder camera and build the (hardware-agnostic) detector.
    # Native stream is 640x480; see config/camera.yaml.
    try:
        from butter_finger.perception.sources import WebcamSource

        source = WebcamSource(args.camera_index, width=640, height=480)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        arm.disconnect()
        return 1

    try:
        detector = HaarFaceDetector()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        source.close()
        arm.disconnect()
        return 1

    tracker = FaceTracker(arm, arm_config, tracking_cfg)
    follower = FaceFollower(tracker, IdleController(arm))
    dt = 1.0 / args.rate_hz

    print("Butter Finger REAL face tracking")
    print(f"  camera=/dev/video{args.camera_index}  rate={args.rate_hz:g} Hz")
    print("  Homing the arm safely, then tracking. Press Ctrl-C to stop.")

    try:
        # Ease to the exact recorded home first, then to the tracking start
        # pose. go_home() also seeds every joint so get_joint_positions()
        # (used by the tracker) is available.
        arm.go_home()
        arm.move_joints(tracker.start_pose, duration_s=2.0)

        while True:
            frame = source.read()
            detection = detector.detect(frame) if frame is not None else None
            # Track a visible face, or idle-scan the base to look for one.
            follower.update(dt, detection)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopping; returning the arm home.")
    finally:
        source.close()
        try:
            arm.go_home()
        finally:
            arm.disconnect()

    print("Real face tracking stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
