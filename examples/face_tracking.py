#!/usr/bin/env python3
"""Camera face tracking: the simulated arm follows a face with its wrist camera.

Run on the simulation machine (Linux primary, Mac fallback), inside the
project's .venv:

    python examples/face_tracking.py                     # webcam + Haar (default)
    python examples/face_tracking.py --source sim --detector scripted
    python examples/face_tracking.py --show              # also show the camera window

Modes:
  --source   webcam  : a local camera via OpenCV/V4L2 (real faces).   [default]
             sim      : render from the arm's wrist camera_link (capture_rgb).
  --detector haar     : OpenCV Haar frontal-face detector.            [default]
             scripted : a synthetic face that orbits the frame (no camera).

The arm pans with the base, tilts with the wrist, and adjusts stand-off with
the shoulder based on apparent face size. This is a SIMULATION feature: it
drives PyBulletArm only and never commands real hardware.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, IdleController, PyBulletArm
from butter_finger.perception import (
    Detection,
    FaceFollower,
    FaceTracker,
    HaarFaceDetector,
    load_tracking_config,
)


def orbiting_detection(step: int, width: int, height: int, target_fraction: float) -> Detection:
    """A synthetic face that circles the frame, for the no-camera demo."""
    t = step / 60.0
    cx = width / 2 + 0.35 * width * math.cos(t)
    cy = height / 2 + 0.25 * height * math.sin(0.7 * t)
    size = target_fraction * width * (1.0 + 0.25 * math.sin(0.3 * t))
    return Detection.from_xywh(cx - size / 2, cy - size / 2, size, size, width, height)


def scripted_detection(
    step: int, width: int, height: int, target_fraction: float
) -> Detection | None:
    """Orbiting face that vanishes for a stretch each cycle.

    With no camera this still exercises the whole loop: the arm tracks the
    face, then (when it disappears) hands off to the idle base scan, then
    reacquires when the face returns.
    """
    if step % 1600 >= 1100:  # ~2 s of no face at 240 Hz -> idle scan
        return None
    return orbiting_detection(step, width, height, target_fraction)


def draw_and_show(cv2, frame_rgb, detection) -> bool:
    """Show the camera frame with the detection box. Returns False to quit."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if detection is not None:
        x = int(detection.cx - detection.w / 2)
        y = int(detection.cy - detection.h / 2)
        cv2.rectangle(bgr, (x, y), (x + int(detection.w), y + int(detection.h)), (0, 255, 0), 2)
    cv2.imshow("Butter Finger face tracking (press q to quit)", bgr)
    return cv2.waitKey(1) & 0xFF != ord("q")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["webcam", "sim"], default="webcam")
    parser.add_argument("--detector", choices=["haar", "scripted"], default="haar")
    parser.add_argument("--camera-index", type=int, default=0, help="camera device index")
    parser.add_argument("--show", action="store_true", help="show the camera window (needs a GUI)")
    args = parser.parse_args()

    tracking_cfg = load_tracking_config()

    try:
        arm = PyBulletArm(gui=True)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with arm:
        tracker = FaceTracker(arm, arm.config, tracking_cfg)
        follower = FaceFollower(tracker, IdleController(arm))
        dt = arm.time_step_s

        # Home to the tracking start pose, then hold briefly.
        arm.reset_joints(tracker.start_pose)
        arm.run_for(0.5)

        # The simulated camera's output frame size (portrait after rotation).
        frame_w = arm.camera_config.output_width
        frame_h = arm.camera_config.output_height

        # scripted synthesizes its own faces => no detector object, no source.
        scripted = args.detector == "scripted"
        detector = None if scripted else HaarFaceDetector()
        source = None
        show = args.show and not scripted
        if not scripted:
            if args.source == "webcam":
                from butter_finger.perception.sources import WebcamSource

                try:
                    source = WebcamSource(args.camera_index, width=frame_w, height=frame_h)
                except RuntimeError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1
            else:  # sim
                from butter_finger.perception.sources import SimCameraSource

                source = SimCameraSource(arm)

        cv2 = None
        if show:
            from butter_finger.perception.detection import _import_cv2

            cv2 = _import_cv2()

        print("Butter Finger face tracking")
        print(f"  source={args.source}  detector={args.detector}  frame={frame_w}x{frame_h}")
        print("  Close the PyBullet window (or press q in the camera window) to exit.")

        step_i = 0
        try:
            while arm.is_connected():
                if scripted:
                    detection = scripted_detection(
                        step_i, frame_w, frame_h, tracking_cfg.target_face_fraction
                    )
                else:
                    frame = source.read()
                    detection = detector.detect(frame) if frame is not None else None
                    if cv2 is not None and frame is not None:
                        if not draw_and_show(cv2, frame, detection):
                            break

                # Track a visible face, or idle-scan the base to look for one.
                follower.update(dt, detection)
                arm.step(realtime=True)
                step_i += 1
        except arm.pb.error:
            pass  # GUI window was closed mid-step
        finally:
            if source is not None:
                source.close()
            if cv2 is not None:
                cv2.destroyAllWindows()

    print("Face tracking stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
