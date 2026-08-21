"""Shared webcam application loop for the Stage 0 and Stage 1 CLIs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from butter_finger.teleoperation.config import (
    TELEOPERATION_CONFIG_PATH,
    TeleoperationConfig,
    load_teleoperation_config,
)
from butter_finger.teleoperation.mediapipe_tracker import (
    DEFAULT_MODEL_PATH,
    MediaPipeHandTracker,
)
from butter_finger.teleoperation.retargeting import VirtualTargetController
from butter_finger.teleoperation.types import HandTrackingResult, TargetUpdate
from butter_finger.teleoperation.visualization import (
    draw_hands,
    draw_runtime_status,
    draw_target_status,
)


class _RateMeter:
    def __init__(self, window: int = 60) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, timestamp_s: float) -> None:
        self._times.append(timestamp_s)

    @property
    def fps(self) -> float:
        if len(self._times) < 2 or self._times[-1] <= self._times[0]:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def build_parser(stage: int) -> argparse.ArgumentParser:
    title = "21-landmark viewer" if stage == 0 else "virtual EE target viewer"
    parser = argparse.ArgumentParser(
        description=(
            f"Butter Finger Stage {stage} {title}. Camera-only: this program "
            "does not construct an arm backend and cannot move the robot."
        )
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", type=Path, default=TELEOPERATION_CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--no-mirror", action="store_true", help="disable selfie-view mirroring"
    )
    if stage == 1:
        parser.add_argument(
            "--log-jsonl",
            type=Path,
            default=None,
            help="write one JSON object per fresh hand result",
        )
    return parser


def run_viewer(
    argv: list[str] | None,
    *,
    stage: int,
    source_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    cv2_module: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run Stage 0 or 1, with injectable boundaries for dependency-free tests."""
    if stage not in (0, 1):
        raise ValueError("stage must be 0 or 1")
    args = build_parser(stage).parse_args(argv)
    try:
        config = load_teleoperation_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not load teleoperation config: {exc}", file=sys.stderr)
        return 1

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError:
            print(
                "ERROR: OpenCV is not installed. Install with "
                "python -m pip install -e '.[teleop]'",
                file=sys.stderr,
            )
            return 1
    if tracker_factory is None and not args.model_path.is_file():
        print(
            f"ERROR: MediaPipe hand model not found at {args.model_path}.\n"
            "Run: python scripts/download_hand_landmarker.py\n"
            "or pass --model-path /path/to/hand_landmarker.task.",
            file=sys.stderr,
        )
        return 1
    if source_factory is None:
        from butter_finger.perception.sources import WebcamSource

        source_factory = WebcamSource
    if tracker_factory is None:
        tracker_factory = MediaPipeHandTracker

    webcam = config.webcam
    source = None
    tracker = None
    try:
        source = source_factory(
            args.camera_index,
            width=webcam.width,
            height=webcam.height,
            fps=webcam.fps,
        )
        tracker = tracker_factory(args.model_path, config.hand_tracking)
    except (OSError, RuntimeError, ValueError) as exc:
        if source is not None:
            source.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    controller = VirtualTargetController(config) if stage == 1 else None
    log_handle = None
    if stage == 1 and args.log_jsonl is not None:
        try:
            log_handle = args.log_jsonl.open("a", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not open JSONL log: {exc}", file=sys.stderr)
            source.close()
            tracker.close()
            return 1

    title = (
        "Butter Finger Stage 0 - hand landmarks (q/Esc to quit)"
        if stage == 0
        else "Butter Finger Stage 1 - virtual target ONLY (q/Esc to quit)"
    )
    print(f"Butter Finger Stage {stage}: CAMERA ONLY; robot motion is disabled.")
    print(
        f"  camera={args.camera_index} {webcam.width}x{webcam.height}@{webcam.fps} "
        f"mirror={webcam.mirror and not args.no_mirror}"
    )

    meter = _RateMeter()
    last_result: HandTrackingResult | None = None
    last_update: TargetUpdate | None = None
    consecutive_read_failures = 0
    max_age_ms = round(config.hand_tracking.max_result_age_s * 1000)
    try:
        while True:
            frame = source.read()
            now_s = clock()
            now_ms = int(now_s * 1000)
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    print("ERROR: webcam returned no frames.", file=sys.stderr)
                    return 1
                continue
            consecutive_read_failures = 0
            meter.tick(now_s)
            logical_frame = np.ascontiguousarray(frame)
            if webcam.mirror and not args.no_mirror:
                logical_frame = np.ascontiguousarray(logical_frame[:, ::-1])
            tracker.submit(logical_frame, now_ms)
            fresh = tracker.take_latest(now_ms=now_ms, max_age_ms=max_age_ms)
            if fresh is not None:
                last_result = fresh
            observations = fresh.observations if fresh is not None else ()

            if controller is not None:
                last_update = controller.update(observations, now_s)
                if fresh is not None and log_handle is not None:
                    _write_log(log_handle, fresh, last_update)

            retained_fresh = (
                last_result is not None
                and now_ms - last_result.timestamp_ms <= max_age_ms
            )
            display_result = last_result if retained_fresh else None
            display_frame = (
                np.asarray(display_result.frame_rgb)
                if display_result is not None
                else logical_frame
            )
            display_observations = (
                display_result.observations if display_result is not None else ()
            )
            display = draw_hands(cv2_module, display_frame, display_observations)
            latency_ms = (
                now_ms - display_result.timestamp_ms
                if display_result is not None
                else None
            )
            draw_runtime_status(
                cv2_module,
                display,
                capture_fps=meter.fps,
                result_fps=tracker.result_fps,
                latency_ms=latency_ms,
                dropped=tracker.dropped_count,
                error=tracker.last_error,
            )
            if last_update is not None:
                draw_target_status(cv2_module, display, last_update)
            cv2_module.imshow(title, display)
            key = cv2_module.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if log_handle is not None:
            log_handle.close()
        source.close()
        tracker.close()
        cv2_module.destroyAllWindows()
    return 0


def _write_log(handle, result: HandTrackingResult, update: TargetUpdate) -> None:
    record = {
        "timestamp_ms": result.timestamp_ms,
        "state": update.state.value,
        "owner_hand_id": update.owner_hand_id,
        "owner_handedness": update.owner_handedness,
        "pinch_ratio": update.pinch_ratio,
        "hand_count": len(result.observations),
        "raw_target": {
            "x_m": update.raw_target.x_m,
            "y_m": update.raw_target.y_m,
            "z_m": update.raw_target.z_m,
            "pitch_rad": update.raw_target.pitch_rad,
        },
        "filtered_target": {
            "x_m": update.target.x_m,
            "y_m": update.target.y_m,
            "z_m": update.target.z_m,
            "pitch_rad": update.target.pitch_rad,
        },
        "clamped_axes": list(update.clamped_axes),
    }
    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    handle.flush()


def load_config_for_tests(path: Path) -> TeleoperationConfig:
    """Small explicit seam used by CLI tests without importing app dependencies."""
    return load_teleoperation_config(path)
