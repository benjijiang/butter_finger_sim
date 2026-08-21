"""OpenCV drawing helpers for the camera-only hand-teleoperation tools."""
from __future__ import annotations

from butter_finger.teleoperation.features import extract_hand_features
from butter_finger.teleoperation.types import (
    DryRunStep,
    HandObservation,
    TargetUpdate,
)

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
        f"state {update.state.value}  owner {owner}  pinch {pinch}  move={update.motion_eligible}",
        f"raw      x={raw.x_m:+.3f} y={raw.y_m:+.3f} z={raw.z_m:+.3f} p={raw.pitch_rad:+.2f}",
        f"filtered x={target.x_m:+.3f} y={target.y_m:+.3f} z={target.z_m:+.3f} p={target.pitch_rad:+.2f}",
        f"workspace clamp: {clamp}",
        "PINCH to move virtual target; RELEASE to hold. Robot disabled.",
    ]
    _draw_lines(cv2, display, lines, display.shape[0] - 92, (80, 255, 255))


def draw_dry_run_status(cv2, display, step: DryRunStep) -> None:
    """Overlay Stage 2 target, raw IK, and actual slew-limited output."""
    cv2.putText(
        display,
        "*** DRY RUN ONLY - NO ROBOT COMMANDS ***",
        (10, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    desired = step.desired_pose
    ik = step.ik_result
    if ik is None:
        raw_status = "NOT RUN"
        raw_residual = "raw IK residual: --"
        raw_joints = "raw IK joints: --"
        limits = "active limits: --"
    else:
        raw_status = ik.status.value
        raw_residual = (
            "raw IK residual: "
            f"pos={ik.position_error_m:.4f}m pitch={ik.pitch_error_rad:+.3f}rad "
            f"iter={ik.iterations}"
        )
        raw_joints = f"raw IK joints: {_format_joints(ik.joints_rad)}"
        limits = (
            "active limits: "
            + (",".join(ik.active_limits) if ik.active_limits else "none")
        )

    if step.slewing:
        motion = "SLEWING"
    elif step.hold_reason:
        motion = f"HOLD ({step.hold_reason})"
    else:
        motion = "TRACKING"
    lines = [
        f"IK {raw_status}  output {motion}  moved={step.moved}",
        f"hold reason: {step.hold_reason or 'none'}",
        (
            "task target "
            f"x={desired.x_m:+.3f} y={desired.y_m:+.3f} "
            f"z={desired.z_m:+.3f} p={desired.pitch_rad:+.2f}"
        ),
        raw_residual,
        raw_joints,
        (
            "output FK residual: "
            f"pos={step.output_position_error_m:.4f}m "
            f"pitch={step.output_pitch_error_rad:+.3f}rad"
        ),
        f"output joints: {_format_joints(step.output_joints_rad)}",
        limits,
        "Configuration-domain candidates only; no collision/hardware validation.",
    ]
    start_y = max(100, display.shape[0] - (len(lines) * 18 + 8))
    _draw_lines(cv2, display, lines, start_y, (60, 220, 255))


def _format_joints(joints: dict[str, float] | None) -> str:
    if joints is None:
        return "--"
    preferred_order = ("base", "shoulder", "elbow", "wrist")
    names = [name for name in preferred_order if name in joints]
    names.extend(sorted(set(joints) - set(names)))
    return " ".join(f"{name}={joints[name]:+.2f}" for name in names)


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
