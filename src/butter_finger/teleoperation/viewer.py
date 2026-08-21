"""Shared webcam application loop for the Stage 0/1/2 CLIs."""
from __future__ import annotations

import argparse
import json
import math
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
from butter_finger.teleoperation.types import (
    DryRunStep,
    EndEffectorPose,
    HandTrackingResult,
    TargetUpdate,
    VirtualEETarget,
)
from butter_finger.teleoperation.visualization import (
    draw_dry_run_status,
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
    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1, or 2")
    titles = {
        0: "21-landmark viewer",
        1: "virtual EE target viewer",
        2: "IK dry-run viewer",
    }
    parser = argparse.ArgumentParser(
        description=(
            f"Butter Finger Stage {stage} {titles[stage]}. Camera-only: this "
            "program does not construct an arm backend and cannot move the robot."
        )
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", type=Path, default=TELEOPERATION_CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--no-mirror", action="store_true", help="disable selfie-view mirroring"
    )
    if stage == 2:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="required acknowledgement that no robot commands will be sent",
        )
    if stage in (1, 2):
        parser.add_argument(
            "--log-jsonl",
            type=Path,
            default=None,
            help=(
                "write JSON diagnostics for fresh results and safety hold "
                "transitions"
            ),
        )
    return parser


def run_viewer(
    argv: list[str] | None,
    *,
    stage: int,
    source_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    controller_factory: Callable[[TeleoperationConfig], Any] | None = None,
    cv2_module: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run a hand-teleoperation stage with injectable runtime boundaries."""
    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1, or 2")
    args = build_parser(stage).parse_args(argv)
    if stage == 2 and not args.dry_run:
        print(
            "ERROR: Stage 2 only supports dry-run diagnostics; pass --dry-run. "
            "Robot control is not implemented.",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_teleoperation_config(args.config, include_ik=stage == 2)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not load teleoperation config: {exc}", file=sys.stderr)
        return 1

    dry_run_controller = None
    if stage == 2:
        if controller_factory is None:
            # Keep Stage 0/1 imports and their proven runtime path independent
            # of the Stage 2 kinematics implementation.
            from butter_finger.teleoperation.dry_run import (
                DryRunTeleoperationController,
            )

            controller_factory = DryRunTeleoperationController
        try:
            # Construction validates geometry, limits, the named anchor pose,
            # and solver settings before the webcam is opened.
            dry_run_controller = controller_factory(config)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: could not initialize IK dry-run: {exc}", file=sys.stderr)
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
        _close_runtime_resources(source=source)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    target_controller = VirtualTargetController(config) if stage in (1, 2) else None
    log_handle = None
    if stage in (1, 2) and args.log_jsonl is not None:
        try:
            log_handle = args.log_jsonl.open("a", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not open JSONL log: {exc}", file=sys.stderr)
            _close_runtime_resources(source=source, tracker=tracker)
            return 1

    titles = {
        0: "Butter Finger Stage 0 - hand landmarks (q/Esc to quit)",
        1: "Butter Finger Stage 1 - virtual target ONLY (q/Esc to quit)",
        2: "Butter Finger Stage 2 - IK DRY RUN ONLY (q/Esc to quit)",
    }
    title = titles[stage]
    mode = "CAMERA ONLY" if stage < 2 else "DRY RUN ONLY"
    print(f"Butter Finger Stage {stage}: {mode}; robot motion is disabled.")
    print(
        f"  camera={args.camera_index} {webcam.width}x{webcam.height}@{webcam.fps} "
        f"mirror={webcam.mirror and not args.no_mirror}"
    )

    meter = _RateMeter()
    last_result: HandTrackingResult | None = None
    last_update: TargetUpdate | None = None
    last_dry_run_step: DryRunStep | None = None
    last_logged_hold_reason: str | None = None
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

            if target_controller is not None:
                last_update = target_controller.update(observations, now_s)
                if stage == 2:
                    assert dry_run_controller is not None
                    last_dry_run_step = dry_run_controller.step(
                        last_update,
                        fresh_result=fresh is not None,
                        now_s=now_s,
                    )
                    hold_event = (
                        fresh is None
                        and last_dry_run_step.hold_reason
                        in {"stale_result", "non_monotonic_time"}
                        and last_dry_run_step.hold_reason
                        != last_logged_hold_reason
                    )
                    if log_handle is not None and (
                        fresh is not None or hold_event
                    ):
                        _write_dry_run_log(
                            log_handle,
                            fresh,
                            last_dry_run_step,
                            config.workspace.initial_target,
                            processed_timestamp_ms=now_ms,
                        )
                        last_logged_hold_reason = last_dry_run_step.hold_reason
                elif fresh is not None and log_handle is not None:
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
            if stage == 1 and last_update is not None:
                draw_target_status(cv2_module, display, last_update)
            elif stage == 2 and last_dry_run_step is not None:
                draw_dry_run_status(cv2_module, display, last_dry_run_step)
            cv2_module.imshow(title, display)
            key = cv2_module.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        _close_runtime_resources(
            log_handle=log_handle,
            source=source,
            tracker=tracker,
            cv2_module=cv2_module,
        )
    return 0


def _close_runtime_resources(
    *,
    log_handle: Any | None = None,
    source: Any | None = None,
    tracker: Any | None = None,
    cv2_module: Any | None = None,
) -> None:
    """Best-effort cleanup that never skips a later independent resource."""
    callbacks = (
        ("JSONL log", getattr(log_handle, "close", None)),
        ("tracker", getattr(tracker, "close", None)),
        ("camera", getattr(source, "close", None)),
        ("OpenCV windows", getattr(cv2_module, "destroyAllWindows", None)),
    )
    for label, callback in callbacks:
        if callback is None:
            continue
        try:
            callback()
        except Exception as exc:
            print(f"WARNING: could not close {label}: {exc}", file=sys.stderr)


def _write_log(handle, result: HandTrackingResult, update: TargetUpdate) -> None:
    record = {
        "timestamp_ms": result.timestamp_ms,
        "state": update.state.value,
        "owner_hand_id": update.owner_hand_id,
        "owner_handedness": update.owner_handedness,
        "pinch_ratio": update.pinch_ratio,
        "motion_eligible": update.motion_eligible,
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
    handle.write(
        json.dumps(
            record,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()


def _finite_float(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _pose_record(pose: EndEffectorPose) -> dict[str, float | None]:
    return {
        "x_m": _finite_float(pose.x_m),
        "y_m": _finite_float(pose.y_m),
        "z_m": _finite_float(pose.z_m),
        "pitch_rad": _finite_float(pose.pitch_rad),
    }


def _target_delta_record(
    target: VirtualEETarget,
    initial: VirtualEETarget,
) -> dict[str, float | None]:
    return {
        "x_m": _finite_float(target.x_m - initial.x_m),
        "y_m": _finite_float(target.y_m - initial.y_m),
        "z_m": _finite_float(target.z_m - initial.z_m),
        "pitch_rad": _finite_float(target.pitch_rad - initial.pitch_rad),
    }


def _joint_record(
    joints: dict[str, float] | None,
) -> dict[str, float | None] | None:
    if joints is None:
        return None
    return {name: _finite_float(value) for name, value in joints.items()}


def _write_dry_run_log(
    handle,
    result: HandTrackingResult | None,
    step: DryRunStep,
    initial_target: VirtualEETarget,
    *,
    processed_timestamp_ms: int,
) -> None:
    """Write one versioned Stage 2 diagnostic record; never emit commands."""
    update = step.target_update
    ik = step.ik_result
    ik_record = None
    if ik is not None:
        ik_record = {
            "status": ik.status.value,
            "target_pose": _pose_record(ik.target_pose),
            "achieved_pose": _pose_record(ik.achieved_pose),
            "position_error_m": _finite_float(ik.position_error_m),
            "pitch_error_rad": _finite_float(ik.pitch_error_rad),
            "iterations": int(ik.iterations),
            "active_limits": list(ik.active_limits),
            "raw_joints_rad": _joint_record(ik.joints_rad),
        }
    record = {
        "schema_version": 1,
        "timestamp_ms": int(
            result.timestamp_ms if result is not None else processed_timestamp_ms
        ),
        "processed_timestamp_ms": int(processed_timestamp_ms),
        "result_timestamp_ms": (
            int(result.timestamp_ms) if result is not None else None
        ),
        "fresh_result": result is not None,
        "hand_count": len(result.observations) if result is not None else 0,
        "stage1_input": {
            "state": update.state.value,
            "owner_hand_id": update.owner_hand_id,
            "owner_handedness": update.owner_handedness,
            "pinch_ratio": update.pinch_ratio,
            "motion_eligible": update.motion_eligible,
            "raw_delta": _target_delta_record(update.raw_target, initial_target),
            "filtered_delta": _target_delta_record(update.target, initial_target),
            "clamped_axes": list(update.clamped_axes),
        },
        "desired_pose": _pose_record(step.desired_pose),
        "ik": ik_record,
        "output": {
            "joints_rad": _joint_record(step.output_joints_rad),
            "pose": _pose_record(step.output_pose),
            "position_error_m": _finite_float(step.output_position_error_m),
            "pitch_error_rad": _finite_float(step.output_pitch_error_rad),
            "moved": bool(step.moved),
            "slewing": bool(step.slewing),
            "hold_reason": step.hold_reason,
        },
    }
    handle.write(
        json.dumps(
            record,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()


def load_config_for_tests(path: Path) -> TeleoperationConfig:
    """Small explicit seam used by CLI tests without importing app dependencies."""
    return load_teleoperation_config(path)
