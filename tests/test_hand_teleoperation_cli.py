"""CLI boundary tests; no MediaPipe, OpenCV, camera, or arm is required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples import hand_landmarks, hand_target


@pytest.mark.parametrize("main", [hand_landmarks.main, hand_target.main])
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


def test_stage_sources_do_not_import_arm_backends() -> None:
    paths = [
        Path(hand_landmarks.__file__),
        Path(hand_target.__file__),
        Path(hand_landmarks.__file__).parents[1]
        / "src"
        / "butter_finger"
        / "teleoperation",
    ]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        if path.is_file()
        else "\n".join(
            child.read_text(encoding="utf-8") for child in path.glob("*.py")
        )
        for path in paths
    )
    assert "butter_finger.backends" not in source
    assert "butter_finger.arm" not in source


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
