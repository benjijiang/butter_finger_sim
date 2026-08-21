"""OpenCV drawing helpers for the camera-only Stage 0/1 tools."""
from __future__ import annotations

from butter_finger.teleoperation.features import extract_hand_features
from butter_finger.teleoperation.types import HandObservation, TargetUpdate

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def draw_hands(cv2, frame_rgb, observations: tuple[HandObservation, ...]):
    """Return a BGR display image with all 21 points and hand metadata."""
    display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = display.shape[:2]
    colors = ((70, 230, 70), (240, 180, 50))
    for hand_index, observation in enumerate(observations):
        color = colors[hand_index % len(colors)]
        points = [
            (int(point[0] * width), int(point[1] * height))
            for point in observation.image_landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(display, points[start], points[end], color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(display, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display, point, 4, color, 1, cv2.LINE_AA)
        try:
            features = extract_hand_features(observation)
            pinch = f" pinch={features.pinch_ratio:.2f}"
        except ValueError:
            pinch = " pinch=invalid"
        label = (
            f"{observation.hand_id} {observation.handedness} "
            f"{observation.confidence:.2f}{pinch}"
        )
        anchor = points[0]
        cv2.putText(
            display,
            label,
            (max(5, anchor[0] - 20), max(20, anchor[1] + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return display


def draw_runtime_status(
    cv2,
    display,
    *,
    capture_fps: float,
    result_fps: float,
    latency_ms: int | None,
    dropped: int,
    error: str | None = None,
) -> None:
    latency = "--" if latency_ms is None else str(latency_ms)
    lines = [
        f"capture {capture_fps:4.1f} FPS  result {result_fps:4.1f} FPS",
        f"latency {latency} ms  dropped/overwritten {dropped}",
    ]
    if error:
        lines.append(f"tracker error: {error}")
    _draw_lines(cv2, display, lines, 22, (230, 230, 230))


def draw_target_status(cv2, display, update: TargetUpdate) -> None:
    raw = update.raw_target
    target = update.target
    clamp = ",".join(update.clamped_axes) if update.clamped_axes else "none"
    owner = update.owner_hand_id or "none"
    pinch = "--" if update.pinch_ratio is None else f"{update.pinch_ratio:.2f}"
    lines = [
        f"state {update.state.value}  owner {owner}  pinch {pinch}",
        f"raw      x={raw.x_m:+.3f} y={raw.y_m:+.3f} z={raw.z_m:+.3f} p={raw.pitch_rad:+.2f}",
        f"filtered x={target.x_m:+.3f} y={target.y_m:+.3f} z={target.z_m:+.3f} p={target.pitch_rad:+.2f}",
        f"workspace clamp: {clamp}",
        "PINCH to move virtual target; RELEASE to hold. Robot disabled.",
    ]
    _draw_lines(cv2, display, lines, display.shape[0] - 92, (80, 255, 255))


def _draw_lines(cv2, display, lines: list[str], y: int, color) -> None:
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (10, y + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
