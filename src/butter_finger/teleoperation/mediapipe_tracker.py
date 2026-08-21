"""Lazy MediaPipe Tasks adapter with latest-only asynchronous results."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from butter_finger.config import REPO_ROOT
from butter_finger.teleoperation.buffering import LatestResultBuffer
from butter_finger.teleoperation.config import HandTrackingConfig
from butter_finger.teleoperation.types import HandObservation, HandTrackingResult

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "hand_landmarker.task"


def _import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Install the Stage 0/1 extras with:\n"
            "    python -m pip install -e '.[teleop]'"
        ) from exc
    return mp


class MediaPipeHandTracker:
    """Submit RGB frames without blocking; consume only the newest result."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        config: HandTrackingConfig | None = None,
    ) -> None:
        from butter_finger.teleoperation.config import load_teleoperation_config

        self._config = config or load_teleoperation_config().hand_tracking
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise RuntimeError(
                f"MediaPipe hand model not found at {self._model_path}.\n"
                "Download and verify it with:\n"
                "    python scripts/download_hand_landmarker.py\n"
                "or pass --model-path /path/to/hand_landmarker.task."
            )
        mp = _import_mediapipe()
        self._mp = mp
        self._buffer: LatestResultBuffer[HandTrackingResult] = LatestResultBuffer()
        self._lock = threading.Lock()
        self._submitted_count = 0
        self._completed_count = 0
        self._last_submit_ms = -1
        self._result_times: deque[float] = deque(maxlen=60)
        self._track_centers: dict[str, tuple[str, float, float, int]] = {}
        self._next_track_id = 1
        self._last_error: str | None = None

        options = mp.tasks.vision.HandLandmarkerOptions(
            # CPU is deliberate: the Tasks default may select Metal on macOS,
            # which can abort from a worker thread when no accelerated GL
            # service is available. CPU also keeps Linux/Mac behavior aligned.
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_hand_presence_confidence=self._config.min_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            result_callback=self._on_result,
        )
        try:
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize MediaPipe HandLandmarker with "
                f"{self._model_path}: {exc}"
            ) from exc

    @property
    def submitted_count(self) -> int:
        with self._lock:
            return self._submitted_count

    @property
    def completed_count(self) -> int:
        with self._lock:
            return self._completed_count

    @property
    def dropped_count(self) -> int:
        with self._lock:
            inferred = max(0, self._submitted_count - self._completed_count - 1)
        return inferred + self._buffer.overwritten_count

    @property
    def result_fps(self) -> float:
        with self._lock:
            times = tuple(self._result_times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def submit(self, frame_rgb: Any, timestamp_ms: int) -> int:
        """Submit one logical RGB frame and return its monotonic timestamp."""
        array = np.asarray(frame_rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("frame_rgb must be an HxWx3 uint8 array")
        if timestamp_ms <= self._last_submit_ms:
            timestamp_ms = self._last_submit_ms + 1
        self._last_submit_ms = timestamp_ms
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(array),
        )
        with self._lock:
            self._submitted_count += 1
        self._landmarker.detect_async(image, timestamp_ms)
        return timestamp_ms

    def take_latest(self, *, now_ms: int, max_age_ms: int) -> HandTrackingResult | None:
        return self._buffer.take_latest(now_ms=now_ms, max_age_ms=max_age_ms)

    def _on_result(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        try:
            observations = self._convert_result(result, timestamp_ms)
            frame = np.asarray(output_image.numpy_view()).copy()
            self._buffer.publish(
                HandTrackingResult(
                    timestamp_ms=timestamp_ms,
                    observations=observations,
                    frame_rgb=frame,
                )
            )
            with self._lock:
                self._completed_count += 1
                self._result_times.append(time.monotonic())
        except Exception as exc:  # callback exceptions cannot escape MediaPipe
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

    def _convert_result(
        self, result: Any, timestamp_ms: int
    ) -> tuple[HandObservation, ...]:
        image_hands = tuple(getattr(result, "hand_landmarks", ()) or ())
        world_hands = tuple(getattr(result, "hand_world_landmarks", ()) or ())
        handedness = tuple(getattr(result, "handedness", ()) or ())
        raw: list[tuple[str, float, tuple, tuple, float, float]] = []
        for index, image_points in enumerate(image_hands):
            if index >= len(world_hands):
                continue
            categories = handedness[index] if index < len(handedness) else ()
            category = categories[0] if categories else None
            label = str(getattr(category, "category_name", "Unknown") or "Unknown")
            confidence = float(getattr(category, "score", 0.0) or 0.0)
            image_tuple = tuple(
                (float(point.x), float(point.y), float(point.z))
                for point in image_points
            )
            world_tuple = tuple(
                (float(point.x), float(point.y), float(point.z))
                for point in world_hands[index]
            )
            if len(image_tuple) != 21 or len(world_tuple) != 21:
                continue
            center_u = sum(image_tuple[i][0] for i in (0, 5, 9, 13, 17)) / 5.0
            center_v = sum(image_tuple[i][1] for i in (0, 5, 9, 13, 17)) / 5.0
            raw.append((label, confidence, image_tuple, world_tuple, center_u, center_v))

        ids = self._assign_track_ids(raw, timestamp_ms)
        return tuple(
            HandObservation(
                timestamp_s=timestamp_ms / 1000.0,
                hand_id=hand_id,
                handedness=item[0],
                confidence=item[1],
                image_landmarks=item[2],
                world_landmarks=item[3],
            )
            for hand_id, item in zip(ids, raw)
        )

    def _assign_track_ids(self, hands: list[tuple], timestamp_ms: int) -> list[str]:
        with self._lock:
            self._track_centers = {
                track_id: track
                for track_id, track in self._track_centers.items()
                if timestamp_ms - track[3] <= 500
            }
            unused = set(self._track_centers)
            assigned: list[str] = []
            for hand in hands:
                label, _, _, _, center_u, center_v = hand
                matches = [
                    track_id
                    for track_id in unused
                    if self._track_centers[track_id][0].lower() == label.lower()
                ]
                if matches:
                    nearest = min(
                        matches,
                        key=lambda track_id: math.hypot(
                            center_u - self._track_centers[track_id][1],
                            center_v - self._track_centers[track_id][2],
                        ),
                    )
                    distance = math.hypot(
                        center_u - self._track_centers[nearest][1],
                        center_v - self._track_centers[nearest][2],
                    )
                    track_id = nearest if distance <= self._config.owner_match_distance else ""
                else:
                    track_id = ""
                if not track_id:
                    track_id = f"hand-{self._next_track_id}"
                    self._next_track_id += 1
                unused.discard(track_id)
                self._track_centers[track_id] = (
                    label,
                    center_u,
                    center_v,
                    timestamp_ms,
                )
                assigned.append(track_id)
            return assigned

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "MediaPipeHandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
