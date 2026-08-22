"""Dependency-free tests for Stage 0/1 hand tracking and retargeting."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from butter_finger.teleoperation import (
    ExponentialTargetFilter,
    HandObservation,
    LatestResultBuffer,
    TrackingState,
    VirtualEETarget,
    VirtualTargetController,
    extract_hand_features,
    load_teleoperation_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def hand(
    timestamp_s: float,
    *,
    hand_id: str = "right-1",
    handedness: str = "Right",
    confidence: float = 0.9,
    u: float = 0.5,
    v: float = 0.5,
    scale: float = 0.1,
    pinch_ratio: float = 0.6,
    pinch_u: float | None = None,
    pinch_v: float | None = None,
    pitch_rad: float = 0.0,
) -> HandObservation:
    image = [[u, v, 0.0] for _ in range(21)]
    image[5] = [u - scale / 2.0, v, 0.0]
    image[17] = [u + scale / 2.0, v, 0.0]
    pinch_center_u = u if pinch_u is None else pinch_u
    pinch_center_v = v if pinch_v is None else pinch_v
    half_separation = pinch_ratio * scale / 2.0
    image[4] = [pinch_center_u - half_separation, pinch_center_v, 0.0]
    image[8] = [pinch_center_u + half_separation, pinch_center_v, 0.0]

    world = [[0.0, 0.0, 0.0] for _ in range(21)]
    world[5] = [1.0 if handedness.lower() == "right" else -1.0, 0.0, 0.0]
    world[17] = [0.0, -math.cos(pitch_rad), math.sin(pitch_rad)]
    return HandObservation(
        timestamp_s=timestamp_s,
        hand_id=hand_id,
        handedness=handedness,
        confidence=confidence,
        image_landmarks=tuple(tuple(point) for point in image),
        world_landmarks=tuple(tuple(point) for point in world),
    )


def engage(controller: VirtualTargetController, *, hand_id: str = "right-1") -> None:
    for frame in range(3):
        timestamp = 0.01 * (frame + 1)
        update = controller.update(
            [hand(timestamp, hand_id=hand_id, pinch_ratio=0.2)], timestamp
        )
    assert update.state is TrackingState.CLUTCHED


def test_extracts_center_scale_pinch_and_pitch_for_both_hands() -> None:
    right = extract_hand_features(
        hand(1.0, u=0.4, v=0.6, scale=0.2, pinch_ratio=0.25, pitch_rad=0.3)
    )
    left = extract_hand_features(
        hand(
            1.0,
            hand_id="left-1",
            handedness="Left",
            u=0.4,
            v=0.6,
            scale=0.2,
            pinch_ratio=0.25,
            pitch_rad=0.3,
        )
    )
    assert right.palm_u == pytest.approx(0.4)
    assert right.palm_v == pytest.approx(0.6)
    assert right.pinch_u == pytest.approx(0.4)
    assert right.pinch_v == pytest.approx(0.6)
    assert right.palm_scale == pytest.approx(0.2)
    assert right.pinch_ratio == pytest.approx(0.25)
    assert right.palm_pitch_rad == pytest.approx(0.3)
    assert left.palm_pitch_rad == pytest.approx(right.palm_pitch_rad)


@pytest.mark.parametrize("failure", ["zero_scale", "nan", "short"])
def test_rejects_invalid_landmarks(failure: str) -> None:
    observation = hand(1.0)
    image = list(observation.image_landmarks)
    if failure == "zero_scale":
        image[17] = image[5]
    elif failure == "nan":
        image[3] = (float("nan"), 0.0, 0.0)
    else:
        image.pop()
    invalid = HandObservation(
        observation.timestamp_s,
        observation.hand_id,
        observation.handedness,
        observation.confidence,
        tuple(image),
        observation.world_landmarks,
    )
    with pytest.raises(ValueError):
        extract_hand_features(invalid)


def test_filter_response_is_loop_rate_independent() -> None:
    initial = VirtualEETarget(0.0, 0.0, 0.0, 0.0, 0.0)

    def run(rate_hz: int) -> float:
        filt = ExponentialTargetFilter(initial, cutoff_hz=3.0)
        filt.update(initial)
        for step in range(1, rate_hz + 1):
            filt.update(VirtualEETarget(1.0, 0.0, 0.0, 0.0, step / rate_hz))
        return filt.value.x_m

    assert run(30) == pytest.approx(run(120), abs=1e-12)


def test_three_frame_debounce_and_engage_has_no_target_jump() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    initial = controller.target
    for frame in range(2):
        update = controller.update(
            [hand(0.01 * (frame + 1), pinch_ratio=0.2)], 0.01 * (frame + 1)
        )
        assert update.state is TrackingState.HOVER
    update = controller.update([hand(0.03, pinch_ratio=0.2)], 0.03)
    assert update.state is TrackingState.CLUTCHED
    assert update.target.x_m == pytest.approx(initial.x_m)
    assert update.target.y_m == pytest.approx(initial.y_m)
    assert update.target.z_m == pytest.approx(initial.z_m)
    assert update.target.pitch_rad == pytest.approx(initial.pitch_rad)


def test_relative_mapping_directions_and_release_hold() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller)
    moved = controller.update(
        [
            hand(
                0.10,
                u=0.6,
                v=0.4,
                scale=0.12,
                pinch_ratio=0.2,
                pitch_rad=0.2,
            )
        ],
        0.10,
    )
    initial = load_teleoperation_config().workspace.initial_target
    assert moved.raw_target.x_m > initial.x_m
    assert moved.raw_target.y_m < initial.y_m
    assert moved.raw_target.z_m > initial.z_m
    assert moved.raw_target.pitch_rad > initial.pitch_rad

    held_target = moved.target
    for frame in range(3):
        timestamp = 0.11 + frame * 0.01
        released = controller.update(
            [hand(timestamp, pinch_ratio=0.6)], timestamp
        )
        assert released.motion_eligible is False
    assert released.state is TrackingState.HOLD
    assert released.target == held_target


def test_screen_plane_mapping_uses_pinch_midpoint_not_palm_center() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller)
    initial = controller.target

    moved = controller.update(
        [
            hand(
                0.10,
                u=0.5,
                v=0.5,
                pinch_u=0.6,
                pinch_v=0.4,
                scale=0.1,
                pinch_ratio=0.2,
            )
        ],
        0.10,
    )

    assert moved.raw_target.x_m == pytest.approx(initial.x_m)
    assert moved.raw_target.y_m < initial.y_m
    assert moved.raw_target.z_m > initial.z_m


def test_release_debounce_and_brief_owner_gap_freeze_stage2_immediately() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller)

    releasing = controller.update([hand(0.04, pinch_ratio=0.6)], 0.04)
    missing = controller.update([], 0.05)

    assert releasing.state is TrackingState.CLUTCHED
    assert releasing.motion_eligible is False
    assert missing.state is TrackingState.CLUTCHED
    assert missing.motion_eligible is False


def test_same_frame_tie_uses_higher_confidence() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    for frame in range(3):
        timestamp = 0.01 * (frame + 1)
        update = controller.update(
            [
                hand(timestamp, hand_id="low", confidence=0.6, pinch_ratio=0.2),
                hand(
                    timestamp,
                    hand_id="high",
                    handedness="Left",
                    confidence=0.95,
                    pinch_ratio=0.2,
                ),
            ],
            timestamp,
        )
    assert update.owner_hand_id == "high"
    assert update.owner_handedness == "Left"


def test_owner_does_not_switch_to_other_visible_hand() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller, hand_id="owner")
    update = controller.update(
        [
            hand(0.1, hand_id="owner", confidence=0.6, pinch_ratio=0.2),
            hand(
                0.1,
                hand_id="other",
                handedness="Left",
                confidence=0.99,
                pinch_ratio=0.2,
            ),
        ],
        0.1,
    )
    assert update.owner_hand_id == "owner"


def test_lost_owner_requires_open_then_new_pinch() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller)
    assert controller.update([], 0.20).state is TrackingState.LOST

    for frame in range(4):
        timestamp = 0.21 + frame * 0.01
        update = controller.update([hand(timestamp, pinch_ratio=0.2)], timestamp)
    assert update.state is TrackingState.LOST
    assert update.owner_hand_id is None

    for frame in range(3):
        timestamp = 0.30 + frame * 0.01
        update = controller.update([hand(timestamp, pinch_ratio=0.6)], timestamp)
    assert update.state is TrackingState.HOVER

    for frame in range(3):
        timestamp = 0.40 + frame * 0.01
        update = controller.update([hand(timestamp, pinch_ratio=0.2)], timestamp)
    assert update.state is TrackingState.CLUTCHED


def test_workspace_clamps_all_axes() -> None:
    config = load_teleoperation_config()
    controller = VirtualTargetController(config)
    engage(controller)
    update = controller.update(
        [hand(1.0, u=2.0, v=-2.0, scale=10.0, pinch_ratio=0.2, pitch_rad=2.0)],
        1.0,
    )
    assert set(update.clamped_axes) == {"x_m", "y_m", "z_m", "pitch_rad"}
    assert config.workspace.x_m.contains(update.raw_target.x_m)
    assert config.workspace.y_m.contains(update.raw_target.y_m)
    assert config.workspace.z_m.contains(update.raw_target.z_m)
    assert config.workspace.pitch_rad.contains(update.raw_target.pitch_rad)


def test_stale_observation_does_not_update_target() -> None:
    controller = VirtualTargetController(load_teleoperation_config())
    engage(controller)
    before = controller.target
    update = controller.update(
        [hand(0.04, u=0.9, scale=0.2, pinch_ratio=0.2)],
        0.30,
    )
    assert update.target == before


@dataclass(frozen=True)
class Buffered:
    timestamp_ms: int
    value: str


def test_latest_buffer_is_once_only_and_rejects_stale_results() -> None:
    buffer: LatestResultBuffer[Buffered] = LatestResultBuffer()
    buffer.publish(Buffered(100, "old"))
    buffer.publish(Buffered(110, "new"))
    assert buffer.overwritten_count == 1
    assert buffer.take_latest(now_ms=120, max_age_ms=20).value == "new"
    assert buffer.take_latest(now_ms=120, max_age_ms=20) is None
    buffer.publish(Buffered(130, "stale"))
    assert buffer.take_latest(now_ms=200, max_age_ms=20) is None


def test_config_rejects_bad_clutch_hysteresis(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config" / "teleoperation.yaml").read_text(encoding="utf-8")
    )
    invalid = copy.deepcopy(raw)
    invalid["teleoperation"]["clutch"]["engage_ratio"] = 0.5
    invalid["teleoperation"]["clutch"]["release_ratio"] = 0.4
    path = tmp_path / "teleoperation.yaml"
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="less than release"):
        load_teleoperation_config(path)


def test_default_config_is_camera_only_and_provisional() -> None:
    config = load_teleoperation_config()
    assert config.webcam.width == 640
    assert config.webcam.height == 480
    assert config.webcam.fps == 30
    assert config.webcam.mirror is True
    assert config.hand_tracking.num_hands == 2
    assert config.clutch.engage_ratio == pytest.approx(0.40)
    assert config.clutch.release_ratio == pytest.approx(0.50)
    assert config.filter_cutoff_hz == pytest.approx(3.0)
    assert config.ik.anchor_pose == "idle_ready"
    assert config.ik.end_effector == "wrist_tip"
    assert config.ik.position_tolerance_m == pytest.approx(0.002)
    assert config.ik.pitch_tolerance_rad == pytest.approx(0.035)
    assert set(config.ik.joint_rate_limits_rad_s) == {
        "base",
        "shoulder",
        "elbow",
        "wrist",
    }


def test_stage01_can_load_legacy_config_without_ik(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config" / "teleoperation.yaml").read_text(encoding="utf-8")
    )
    del raw["teleoperation"]["ik"]
    path = tmp_path / "legacy-teleoperation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    stage01 = load_teleoperation_config(path)

    assert stage01.ik is None
    assert stage01.workspace.initial_target.x_m == pytest.approx(0.18)
    with pytest.raises(ValueError, match="teleoperation.ik"):
        load_teleoperation_config(path, include_ik=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_forward_component", 1.0, "between 0 and 1"),
        ("max_iterations", 0, "positive integer"),
        ("max_slew_dt_s", float("nan"), "finite"),
    ],
)
def test_config_rejects_invalid_ik_values(
    tmp_path: Path, field: str, value, message: str
) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config" / "teleoperation.yaml").read_text(encoding="utf-8")
    )
    raw["teleoperation"]["ik"][field] = value
    path = tmp_path / "teleoperation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_teleoperation_config(path)


def test_config_rejects_incomplete_ik_joint_rate_map(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config" / "teleoperation.yaml").read_text(encoding="utf-8")
    )
    del raw["teleoperation"]["ik"]["joint_rate_limits_rad_s"]["wrist"]
    path = tmp_path / "teleoperation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly"):
        load_teleoperation_config(path)
