#!/usr/bin/env python3
"""CAMERA-ONLY diagnostic: why does real face tracking not see a face?

This script NEVER touches the arm — no SDK, no PWM, nothing moves. It only
opens the Hiwonder USB camera and answers three questions:

  1. Orientation. The recorded camera mounting says the top of the NATIVE
     image points along wrist-local -X, which is why PyBulletArm rotates its
     render 90 degrees clockwise (config/camera.yaml `output`). WebcamSource
     does NOT apply that rotation, so real frames reach the Haar cascade
     rotated by a quarter turn — and a frontal-face cascade does not detect
     sideways faces. This runs the detector at all four rotations and reports
     which one actually finds your face.

  2. Apparent size ("focal length"). It reports the detected box width as a
     fraction of image width against tracking.target_face_fraction, so you can
     see whether the wide-angle lens makes your face far smaller than the
     distance controller expects.

  3. Loop timing. It reports how long a capture + detect actually takes, which
     is the real dt the control loop should be using.

Run on the Pi:

    python examples/diagnose_face_camera.py
    python examples/diagnose_face_camera.py --frames 120 --save /tmp/bf_cam

Sit in front of the camera, roughly where you would while talking to the arm,
for the whole run.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.camera import rotate_rgb_clockwise
from butter_finger.config import load_camera_config
from butter_finger.perception import HaarFaceDetector, load_tracking_config
from butter_finger.perception.detection import Detection
from butter_finger.perception.sources import WebcamSource

ROTATIONS = (0, 90, 180, 270)


def describe_position(detection: Detection) -> str:
    """Where in the frame the face sits, as normalized -1..1 errors."""
    error_x = (detection.cx - detection.image_width / 2.0) / (detection.image_width / 2.0)
    error_y = (detection.cy - detection.image_height / 2.0) / (detection.image_height / 2.0)
    return f"x={error_x:+.2f} y={error_y:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="how many frames to sample (default 60)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="directory to write sample PNGs to (checks focus and exposure)",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        print("--frames must be positive", file=sys.stderr)
        return 2

    camera_cfg = load_camera_config()
    tracking_cfg = load_tracking_config()

    try:
        source = WebcamSource(
            args.camera_index,
            width=camera_cfg.native_width,
            height=camera_cfg.native_height,
        )
        detector = HaarFaceDetector()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Butter Finger camera diagnostic (the arm will NOT move)")
    print(f"  camera=/dev/video{args.camera_index}  frames={args.frames}")
    print(f"  config native={camera_cfg.native_width}x{camera_cfg.native_height}, "
          f"sim output rotates {camera_cfg.rotate_clockwise_deg} deg clockwise")
    print("  Sit in front of the camera now.\n")

    hits: dict[int, list[Detection]] = {degrees: [] for degrees in ROTATIONS}
    read_times_s: list[float] = []
    detect_times_s: list[float] = []
    frames_read = 0
    first_frame = None
    samples: dict[int, object] = {}

    for _ in range(args.frames):
        start = time.perf_counter()
        frame = source.read()
        read_times_s.append(time.perf_counter() - start)
        if frame is None:
            continue
        frames_read += 1
        if first_frame is None:
            first_frame = frame

        for degrees in ROTATIONS:
            rotated = frame if degrees == 0 else rotate_rgb_clockwise(frame, degrees)
            start = time.perf_counter()
            detection = detector.detect(rotated)
            if degrees == 0:
                detect_times_s.append(time.perf_counter() - start)
            if detection is not None:
                hits[degrees].append(detection)
                samples.setdefault(degrees, rotated)

    source.close()

    if frames_read == 0:
        print("No frames were read at all. The camera opened but returns nothing:")
        print("  * check `v4l2-ctl --list-devices` and try another --camera-index")
        print("  * another process may already hold the camera")
        return 1

    shape = getattr(first_frame, "shape", None)
    print(f"Frames read: {frames_read}/{args.frames}   raw frame shape: {shape}")
    print(f"Capture: {statistics.mean(read_times_s) * 1000:.0f} ms/frame   "
          f"Haar detect: {statistics.mean(detect_times_s) * 1000:.0f} ms/frame")
    real_dt = statistics.mean(read_times_s) + statistics.mean(detect_times_s)
    print(f"  => a real capture+detect loop runs at about {1.0 / real_dt:.1f} Hz "
          f"(dt={real_dt:.3f} s), not the 30 Hz the example assumes.\n")

    print("Detection rate by clockwise rotation applied before detecting:")
    for degrees in ROTATIONS:
        found = hits[degrees]
        rate = len(found) / frames_read
        line = f"  {degrees:3d} deg : {len(found):3d}/{frames_read} frames ({rate:5.1%})"
        if found:
            fractions = [det.width_fraction for det in found]
            line += (
                f"   face width = {statistics.median(fractions):.3f} of image "
                f"(target {tracking_cfg.target_face_fraction:.2f})"
                f"   last {describe_position(found[-1])}"
            )
        print(line)

    best = max(ROTATIONS, key=lambda degrees: len(hits[degrees]))
    print()
    if not hits[best]:
        print("No face found at ANY rotation. That points away from orientation:")
        print("  * check the saved PNG for focus, exposure, and framing (--save)")
        print("  * a wide-angle lens far away can make the face smaller than the")
        print("    detector's 30x30 px minimum")
        print("  * make sure the camera is not pointing at the ceiling from the")
        print("    arm's home pose")
    elif best == 0:
        print("Best rotation is 0: the raw webcam frame is already upright, so")
        print("orientation is NOT the problem. Look at the control loop instead.")
    else:
        print(f"Best rotation is {best} deg clockwise, matching config/camera.yaml's")
        print(f"output rotation of {camera_cfg.rotate_clockwise_deg} deg. The real")
        print("path (WebcamSource) does not apply it, so the detector is being fed")
        print("sideways faces and the pan/tilt image axes are swapped.")

    if args.save is not None:
        try:
            import cv2
        except ImportError:
            print("\n(--save needs cv2, which is already required here)")
            return 0
        args.save.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save / "raw.png"), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for degrees, image in samples.items():
            cv2.imwrite(
                str(args.save / f"hit_{degrees}deg.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        print(f"\nWrote sample images to {args.save} — check raw.png for focus "
              "and exposure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
