"""CLI boundary tests; no MediaPipe, OpenCV, camera, or arm is required."""
from __future__ import annotations

import builtins
import io
import json
from pathlib import Path

import numpy as np
import pytest

from examples import hand_landmarks, hand_target, hand_teleoperation
from butter_finger.teleoperation.types import (
    DryRunStep,
    EndEffectorPose,
    HandTrackingResult,
    IKResult,
    IKStatus,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)


@pytest.mark.parametrize(
    "main", [hand_landmarks.main, hand_target.main, hand_teleoperation.main]
)
def test_help_exits_before_opening_runtime_dependencies(main) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


class FakeSource:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_missing_model_returns_before_opening_camera(tmp_path: Path) -> None:
    source_calls: list[tuple] = []

    result = hand_landmarks.main(
        ["--model-path", str(tmp_path / "missing.task")],
        source_factory=lambda *args, **kwargs: source_calls.append((args, kwargs)),
        cv2_module=object(),
    )

    assert result == 1
    assert source_calls == []


@pytest.mark.parametrize("main", [hand_landmarks.main, hand_target.main])
def test_camera_failure_returns_cleanly_without_constructing_tracker(main) -> None:
    tracker_calls: list[tuple] = []

    def fail_camera(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    result = main(
        [],
        source_factory=fail_camera,
        tracker_factory=lambda *args: tracker_calls.append(args),
        cv2_module=object(),
    )

    assert result == 1
    assert tracker_calls == []


def test_stage2_requires_dry_run_before_loading_any_runtime(monkeypatch) -> None:
    from butter_finger.teleoperation import viewer

    calls: list[str] = []

    def fail_config(*args, **kwargs):
        calls.append("config")
        raise AssertionError("configuration must not be read")

    monkeypatch.setattr(viewer, "load_teleoperation_config", fail_config)
    result = hand_teleoperation.main(
        [],
        source_factory=lambda *a, **k: calls.append("source"),
        tracker_factory=lambda *a, **k: calls.append("tracker"),
        controller_factory=lambda config: calls.append("controller"),
        cv2_module=object(),
    )

    assert result == 2
    assert calls == []


def test_stage2_default_runtime_constructs_no_arm_or_external_robot_runtime(
    monkeypatch,
) -> None:
    from butter_finger import PWMRobotArm, PyBulletArm, RaspberryPiArm
    from butter_finger.teleoperation import viewer

    def fail_arm(*args, **kwargs):
        raise AssertionError("Stage 2 must not construct an arm")

    for arm_type in (PWMRobotArm, PyBulletArm, RaspberryPiArm):
        monkeypatch.setattr(arm_type, "__init__", fail_arm)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"pybullet", "ros_robot_controller_sdk"}:
            raise AssertionError(f"Stage 2 must not load {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source = FakeSource()
    source.read = lambda: np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeTracker:
        result_fps = 0.0
        dropped_count = 0
        last_error = None

        def submit(self, frame, timestamp_ms):
            return timestamp_ms

        def take_latest(self, **kwargs):
            return None

        def close(self):
            pass

    class FakeCV2:
        def imshow(self, title, display):
            pass

        def waitKey(self, delay):
            return ord("q")

        def destroyAllWindows(self):
            pass

    monkeypatch.setattr(viewer, "draw_hands", lambda cv, image, obs: image)
    monkeypatch.setattr(viewer, "draw_runtime_status", lambda *a, **k: None)
    monkeypatch.setattr(viewer, "draw_dry_run_status", lambda *a, **k: None)

    result = hand_teleoperation.main(
        ["--dry-run"],
        source_factory=lambda *args, **kwargs: source,
        tracker_factory=lambda *args, **kwargs: FakeTracker(),
        cv2_module=FakeCV2(),
        clock=lambda: 1.0,
    )

    assert result == 0


def test_clean_quit_closes_camera_tracker_and_windows(monkeypatch) -> None:
    from butter_finger.teleoperation import viewer

    source = FakeSource()
    source.read = lambda: np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeTracker:
        result_fps = 0.0
        dropped_count = 0
        last_error = None

        def __init__(self) -> None:
            self.closed = False

        def submit(self, frame, timestamp_ms):
            return timestamp_ms

        def take_latest(self, **kwargs):
            return None

        def close(self):
            self.closed = True

    tracker = FakeTracker()

    class FakeCV2:
        def __init__(self) -> None:
            self.destroyed = False

        def imshow(self, title, display):
            pass

        def waitKey(self, delay):
            return ord("q")

        def destroyAllWindows(self):
            self.destroyed = True

    cv2 = FakeCV2()
    monkeypatch.setattr(viewer, "draw_hands", lambda cv, frame, obs: frame)
    monkeypatch.setattr(viewer, "draw_runtime_status", lambda *a, **k: None)
    monkeypatch.setattr(viewer, "draw_target_status", lambda *a, **k: None)

    result = hand_target.main(
        [],
        source_factory=lambda *args, **kwargs: source,
        tracker_factory=lambda *args, **kwargs: tracker,
        cv2_module=cv2,
        clock=lambda: 1.0,
    )

    assert result == 0
    assert source.closed is True
    assert tracker.closed is True
    assert cv2.destroyed is True


def test_cleanup_attempts_every_resource_after_an_individual_failure(capsys) -> None:
    from butter_finger.teleoperation import viewer

    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    class CV2:
        def destroyAllWindows(self) -> None:
            calls.append("windows")

    viewer._close_runtime_resources(
        log_handle=Resource("log", fail=True),
        tracker=Resource("tracker"),
        source=Resource("camera", fail=True),
        cv2_module=CV2(),
    )

    assert calls == ["log", "tracker", "camera", "windows"]
    warning = capsys.readouterr().err
    assert "could not close JSONL log" in warning
    assert "could not close camera" in warning


def test_stage2_dry_run_writes_versioned_diagnostics_and_cleans_up(
    monkeypatch, tmp_path: Path
) -> None:
    from butter_finger.teleoperation import viewer
    source = FakeSource()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    source.read = lambda: frame

    class FakeTracker:
        result_fps = 30.0
        dropped_count = 0
        last_error = None

        def __init__(self) -> None:
            self.closed = False
            self._taken = False

        def submit(self, frame_rgb, timestamp_ms):
            return timestamp_ms

        def take_latest(self, **kwargs):
            if self._taken:
                return None
            self._taken = True
            return HandTrackingResult(1000, (), frame)

        def close(self):
            self.closed = True

    tracker = FakeTracker()

    class FakeDryRunController:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, float]] = []

        def step(self, update, *, fresh_result, now_s):
            self.calls.append((fresh_result, now_s))
            pose = EndEffectorPose(0.10, 0.02, 0.20, 0.05)
            joints = {
                "base": 0.0,
                "shoulder": -0.3,
                "elbow": -0.5,
                "wrist": -1.0,
            }
            ik = IKResult(
                status=IKStatus.SOLVED,
                target_pose=pose,
                achieved_pose=pose,
                joints_rad=joints,
                position_error_m=0.001,
                pitch_error_rad=0.01,
                iterations=4,
                active_limits=(),
            )
            return DryRunStep(
                target_update=update,
                desired_pose=pose,
                ik_result=ik if fresh_result else None,
                output_joints_rad=joints,
                output_pose=pose,
                output_position_error_m=0.001,
                output_pitch_error_rad=0.01,
                moved=False,
                slewing=fresh_result,
                hold_reason=None if fresh_result else "stale_result",
            )

    controller = FakeDryRunController()

    class FakeCV2:
        def __init__(self) -> None:
            self.destroyed = False
            self.frames = 0

        def imshow(self, title, display):
            pass

        def waitKey(self, delay):
            self.frames += 1
            return ord("q") if self.frames == 2 else -1

        def destroyAllWindows(self):
            self.destroyed = True

    cv2 = FakeCV2()
    monkeypatch.setattr(viewer, "draw_hands", lambda cv, image, obs: image)
    monkeypatch.setattr(viewer, "draw_runtime_status", lambda *a, **k: None)
    monkeypatch.setattr(viewer, "draw_dry_run_status", lambda *a, **k: None)
    log_path = tmp_path / "dry-run.jsonl"
    times = iter((1.0, 1.2))

    result = hand_teleoperation.main(
        ["--dry-run", "--log-jsonl", str(log_path)],
        source_factory=lambda *args, **kwargs: source,
        tracker_factory=lambda *args, **kwargs: tracker,
        controller_factory=lambda config: controller,
        cv2_module=cv2,
        clock=lambda: next(times),
    )

    assert result == 0
    assert controller.calls == [(True, 1.0), (False, 1.2)]
    assert source.closed is True
    assert tracker.closed is True
    assert cv2.destroyed is True
    raw_records = log_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_records) == 2
    record = json.loads(raw_records[0])
    assert record["schema_version"] == 1
    assert record["stage1_input"]["filtered_delta"] == {
        "pitch_rad": 0.0,
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
    }
    assert record["ik"]["status"] == "SOLVED"
    assert record["ik"]["raw_joints_rad"]["shoulder"] == -0.3
    assert record["output"]["slewing"] is True
    stale_record = json.loads(raw_records[1])
    assert stale_record["fresh_result"] is False
    assert stale_record["result_timestamp_ms"] is None
    assert stale_record["output"]["hold_reason"] == "stale_result"
    assert "pwm" not in "\n".join(raw_records).lower()


def test_dry_run_jsonl_replaces_nonfinite_diagnostics_with_null() -> None:
    from butter_finger.teleoperation import viewer

    initial = VirtualEETarget(0.18, 0.0, 0.14, 0.0)
    update = TargetUpdate(
        state=TrackingState.CLUTCHED,
        target=initial,
        raw_target=initial,
        motion_eligible=True,
    )
    desired = EndEffectorPose(float("nan"), 0.0, 0.1, 0.0)
    achieved = EndEffectorPose(0.03, 0.03, 0.19, 0.06)
    ik = IKResult(
        status=IKStatus.NUMERICAL_FAILURE,
        target_pose=desired,
        achieved_pose=achieved,
        joints_rad=None,
        position_error_m=float("inf"),
        pitch_error_rad=0.06,
        iterations=0,
    )
    step = DryRunStep(
        target_update=update,
        desired_pose=desired,
        ik_result=ik,
        output_joints_rad={"base": 0.0},
        output_pose=achieved,
        output_position_error_m=float("inf"),
        output_pitch_error_rad=0.06,
        moved=False,
        slewing=False,
        hold_reason="ik_numerical_failure",
    )
    handle = io.StringIO()

    viewer._write_dry_run_log(
        handle,
        None,
        step,
        initial,
        processed_timestamp_ms=1234,
    )

    raw_record = handle.getvalue()
    record = json.loads(raw_record)
    assert "NaN" not in raw_record
    assert "Infinity" not in raw_record
    assert record["desired_pose"]["x_m"] is None
    assert record["ik"]["position_error_m"] is None
    assert record["output"]["position_error_m"] is None
