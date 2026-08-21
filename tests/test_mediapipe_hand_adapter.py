"""MediaPipe result conversion tests using plain fake result objects."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from butter_finger.teleoperation.config import load_teleoperation_config
from butter_finger.teleoperation.mediapipe_tracker import MediaPipeHandTracker


@dataclass
class Point:
    x: float
    y: float
    z: float


@dataclass
class Category:
    category_name: str
    score: float


@dataclass
class Result:
    hand_landmarks: list[list[Point]]
    hand_world_landmarks: list[list[Point]]
    handedness: list[list[Category]]


def tracker_without_runtime() -> MediaPipeHandTracker:
    tracker = object.__new__(MediaPipeHandTracker)
    tracker._config = load_teleoperation_config().hand_tracking
    tracker._lock = threading.Lock()
    tracker._track_centers = {}
    tracker._next_track_id = 1
    return tracker


def fake_result(center_x: float, label: str = "Right") -> Result:
    image = [Point(center_x, 0.5, -0.01 * index) for index in range(21)]
    world = [Point(0.001 * index, 0.002 * index, 0.003 * index) for index in range(21)]
    return Result([image], [world], [[Category(label, 0.91)]])


def test_converts_all_21_image_and_world_landmarks() -> None:
    tracker = tracker_without_runtime()
    observations = tracker._convert_result(fake_result(0.4), 1234)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.timestamp_s == 1.234
    assert observation.handedness == "Right"
    assert observation.confidence == 0.91
    assert len(observation.image_landmarks) == 21
    assert len(observation.world_landmarks) == 21


def test_nearby_same_handedness_keeps_stable_track_id() -> None:
    tracker = tracker_without_runtime()
    first = tracker._convert_result(fake_result(0.40), 1000)[0]
    second = tracker._convert_result(fake_result(0.45), 1033)[0]
    assert second.hand_id == first.hand_id


def test_far_or_expired_hand_gets_new_track_id() -> None:
    tracker = tracker_without_runtime()
    first = tracker._convert_result(fake_result(0.10), 1000)[0]
    far = tracker._convert_result(fake_result(0.90), 1033)[0]
    expired = tracker._convert_result(fake_result(0.90), 2000)[0]
    assert far.hand_id != first.hand_id
    assert expired.hand_id != far.hand_id
