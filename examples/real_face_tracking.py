#!/usr/bin/env python3
"""REAL HARDWARE: the physical arm follows a face with the Hiwonder camera.

Run ONLY on the Raspberry Pi 5 connected to the RasAdapter5A, with the
Hiwonder USB camera plugged in. This is the real-hardware counterpart to
examples/face_tracking.py: the SAME perception and control code
(FaceFollower / FaceTracker / IdleController / HaarFaceDetector /
WebcamSource) driving RaspberryPiArm instead of PyBulletArm.

    python examples/real_face_tracking.py --dry-run          # nothing moves
    python examples/real_face_tracking.py --confirm-hardware

The arm pans with the base, tilts with the wrist, and adjusts stand-off with
the shoulder from the apparent face size. When no face is seen for a short
grace period it hands off to the idle base scan to look for one, then
reacquires.

Three things differ from the simulation, all of them measured on the Pi with
examples/diagnose_face_camera.py:

  * The camera is mounted a quarter-turn, so webcam frames are rotated by
    config/camera.yaml's `rotate_clockwise_deg` before detection, exactly as
    PyBulletArm.capture_rgb() does. Without this the Haar cascade sees
    sideways faces and detects nothing (measured: 0/120 frames unrotated,
    118/120 rotated).
  * The loop uses its measured wall-clock period, and slews at a rate in
    rad/s rather than config/tracking.yaml's per-step limit, which was tuned
    against a 240 Hz simulation and is far too fast at ~35 Hz.
  * Streamed commands use a short board interpolation window (see
    RaspberryPiArm's stream_duration_s) instead of the one-second default.

VERIFY THE RESPONSE SIGNS FIRST. config/tracking.yaml's sign_pan/sign_tilt/
sign_distance were guessed for simulation and are explicitly not trusted on
hardware. Run --dry-run, move your face, and check that the printed target
deltas move the camera TOWARD your face. If a joint goes the wrong way, flip
its sign with --flip-pan / --flip-tilt / --flip-distance, then make the
change permanent in config/tracking.yaml.

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
import dataclasses
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
from butter_finger.config import load_arm_config, load_camera_config
from butter_finger.perception import (
    FaceFollower,
    FaceTracker,
    HaarFaceDetector,
    load_tracking_config,
)

# Hardware slew limit. config/tracking.yaml's max_step_rad is per control
# step, and at the simulation's 240 Hz it allows several rad/s. This caps the
# real arm's commanded speed in rad/s; the per-step limit is derived from the
# measured loop period below.
DEFAULT_MAX_RATE_RAD_S = 0.35

# Board interpolation window for streamed targets: a little longer than one
# loop period, so motion stays continuous without commands piling up.
STREAM_DURATION_MARGIN = 1.5


class DryRunArm:
    """Wraps RaspberryPiArm and prints commands instead of sending them.

    Joint estimates advance as if the moves happened, so the control law runs
    exactly as it would on hardware — only the PWM writes are suppressed.
    """

    def __init__(self, positions_rad: dict[str, float]) -> None:
        self._positions = dict(positions_rad)
        self.last_targets: dict[str, float] = {}

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._positions)

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def move_joints(self, targets_rad, *, duration_s=None) -> None:
        self.last_targets = {joint: float(v) for joint, v in targets_rad.items()}
        self._positions.update(self.last_targets)

    def move_joint(self, joint, position_rad, *, duration_s=None) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def go_home(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required acknowledgement that this moves the real arm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the camera and control law but never command the arm; "
        "prints the joint deltas it would send, for verifying signs",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="camera device index (/dev/video<index>); default 0",
    )
    parser.add_argument(
        "--target-face-fraction",
        type=float,
        default=None,
        help="face box width as a fraction of image width to hold; overrides "
        "config/tracking.yaml. Measure yours with diagnose_face_camera.py",
    )
    parser.add_argument(
        "--max-rate-rad-s",
        type=float,
        default=DEFAULT_MAX_RATE_RAD_S,
        help=f"joint slew limit in rad/s (default {DEFAULT_MAX_RATE_RAD_S})",
    )
    parser.add_argument("--flip-pan", action="store_true", help="invert sign_pan")
    parser.add_argument("--flip-tilt", action="store_true", help="invert sign_tilt")
    parser.add_argument(
        "--flip-distance", action="store_true", help="invert sign_distance"
    )
    args = parser.parse_args()

    if not args.confirm_hardware and not args.dry_run:
        print(
            "Refusing to move the real arm without --confirm-hardware.\n"
            "Run with --dry-run first to verify the response signs.",
            file=sys.stderr,
        )
        return 2
    if args.max_rate_rad_s <= 0:
        print("--max-rate-rad-s must be greater than zero.", file=sys.stderr)
        return 2
    if args.target_face_fraction is not None and not (
        0.0 < args.target_face_fraction < 1.0
    ):
        print("--target-face-fraction must be in (0, 1).", file=sys.stderr)
        return 2

    # ArmConfig (radians limits + named poses) for the tracker. This is NOT
    # RaspberryPiArm.config, which is a PhysicalConfig (calibrations/PWM).
    arm_config = load_arm_config()
    camera_cfg = load_camera_config()
    tracking_cfg = load_tracking_config()

    overrides: dict[str, float] = {}
    if args.target_face_fraction is not None:
        overrides["target_face_fraction"] = args.target_face_fraction
    if args.flip_pan:
        overrides["sign_pan"] = -tracking_cfg.sign_pan
    if args.flip_tilt:
        overrides["sign_tilt"] = -tracking_cfg.sign_tilt
    if args.flip_distance:
        overrides["sign_distance"] = -tracking_cfg.sign_distance
    if overrides:
        tracking_cfg = dataclasses.replace(tracking_cfg, **overrides)

    # Open the Hiwonder camera and build the (hardware-agnostic) detector.
    # Rotating here is what makes real frames match the sim's upright output.
    try:
        from butter_finger.perception.sources import WebcamSource

        source = WebcamSource(
            args.camera_index,
            width=camera_cfg.native_width,
            height=camera_cfg.native_height,
            rotate_clockwise_deg=camera_cfg.rotate_clockwise_deg,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        detector = HaarFaceDetector()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        source.close()
        return 1

    # Measure the real loop period before deciding how fast to slew and how
    # long the board should take over each streamed command.
    loop_period_s = measure_loop_period(source, detector)
    if loop_period_s is None:
        print("ERROR: the camera returned no frames.", file=sys.stderr)
        source.close()
        return 1

    max_step_rad = args.max_rate_rad_s * loop_period_s
    tracking_cfg = dataclasses.replace(tracking_cfg, max_step_rad=max_step_rad)
    stream_duration_s = loop_period_s * STREAM_DURATION_MARGIN

    arm: object
    if args.dry_run:
        # Seeded with zeros; the loop's move to the start pose overwrites them
        # before the tracker ever reads a position.
        arm = DryRunArm({joint: 0.0 for joint in arm_config.sim_limits})
    else:
        try:
            # verbose=False: at loop rate the per-pulse prints would flood the
            # terminal and, over SSH, slow the loop down.
            from butter_finger.backends.pwm_robot_arm import PWMRobotArm

            arm = RaspberryPiArm(
                pwm_arm=PWMRobotArm(verbose=False),
                stream_duration_s=stream_duration_s,
            )
        except BackendUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            source.close()
            return 1

    tracker = FaceTracker(arm, arm_config, tracking_cfg)
    follower = FaceFollower(tracker, IdleController(arm))

    mode = "DRY RUN (nothing moves)" if args.dry_run else "REAL"
    print(f"\nButter Finger {mode} face tracking")
    print(f"  camera=/dev/video{args.camera_index}  "
          f"rotated {camera_cfg.rotate_clockwise_deg} deg clockwise")
    print(f"  measured loop period {loop_period_s * 1000:.0f} ms "
          f"({1.0 / loop_period_s:.1f} Hz)")
    print(f"  slew {args.max_rate_rad_s:g} rad/s -> {max_step_rad:.4f} rad/step; "
          f"board window {stream_duration_s * 1000:.0f} ms")
    print(f"  signs: pan {tracking_cfg.sign_pan:+g}  tilt {tracking_cfg.sign_tilt:+g}"
          f"  distance {tracking_cfg.sign_distance:+g}")
    print(f"  holding face width at {tracking_cfg.target_face_fraction:.2f} of image")
    if args.dry_run:
        print("\n  Move your face and check each joint moves TOWARD it.")
        print("  Wrong direction? Re-run with --flip-pan / --flip-tilt / "
              "--flip-distance.")
    else:
        print("\n  Homing the arm safely, then tracking.")
    print("  Press Ctrl-C to stop.\n")

    try:
        # Ease to the exact recorded home first, then to the tracking start
        # pose. go_home() also seeds every joint so get_joint_positions()
        # (used by the tracker) is available.
        arm.go_home()
        arm.move_joints(tracker.start_pose, duration_s=2.0)

        last_report = 0.0
        previous = time.monotonic()
        while True:
            frame = source.read()
            detection = detector.detect(frame) if frame is not None else None

            now = time.monotonic()
            dt = now - previous
            previous = now
            if dt <= 0:
                continue

            # Track a visible face, or idle-scan the base to look for one.
            status = follower.update(dt, detection)

            if now - last_report >= 0.5:
                last_report = now
                report(status, detection, arm)
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


def measure_loop_period(source, detector, frames: int = 20) -> float | None:
    """Time capture + detect so the loop can use its true period."""
    print("Measuring the capture + detect period...")
    durations: list[float] = []
    for _ in range(frames):
        start = time.perf_counter()
        frame = source.read()
        if frame is None:
            continue
        detector.detect(frame)
        durations.append(time.perf_counter() - start)
    if not durations:
        return None
    return sum(durations) / len(durations)


def report(status, detection, arm) -> None:
    """One periodic status line: what was seen and what was commanded."""
    if not status.detected:
        position = status.idle_position_rad
        where = f" base={position:+.3f}" if position is not None else ""
        print(f"[{status.state:8}] no face{where}")
        return

    step = status.tracker_step
    targets = " ".join(
        f"{joint}={angle:+.3f}" for joint, angle in sorted(step.targets.items())
    )
    print(
        f"[{status.state:8}] face err x={step.error_x:+.2f} y={step.error_y:+.2f} "
        f"size={step.error_size:+.2f} -> {targets}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
