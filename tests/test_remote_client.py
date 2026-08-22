"""Mac-side Stage 3 shaping and release/arm coordinator tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from butter_finger.remote_control.client import (
    RemoteClientStatus,
    RemoteControlError,
    Stage3Coordinator,
)
from butter_finger.teleoperation.types import (
    DryRunStep,
    EndEffectorPose,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)


class FakeClient:
    def __init__(self) -> None:
        self.server_state = "DISARMED"
        self.failure = None
        self.holds: list[str] = []
        self.arm_requests = 0
        self.candidates: list[tuple[dict[str, float], float]] = []

    @property
    def status(self) -> RemoteClientStatus:
        return RemoteClientStatus(
            connected=self.failure is None,
            mode="hardware",
            server_state=self.server_state,
            last_sent_seq=None,
            last_applied_seq=None,
            applied_joints_rad=None,
            reason=None,
            failure=self.failure,
        )

    def publish_hold(self, reason: str) -> None:
        self.holds.append(reason)

    def mark_control_alive(self) -> None:
        pass

    def request_arm(self) -> None:
        self.arm_requests += 1

    def publish_candidate(self, joints_rad, *, now_s: float):
        self.candidates.append((dict(joints_rad), now_s))


def step(
    state: TrackingState,
    *,
    pinch: float | None,
    motion_eligible: bool,
    hold_reason: str | None,
) -> DryRunStep:
    target = VirtualEETarget(0.18, 0.0, 0.14, 0.0)
    update = TargetUpdate(
        state,
        target,
        target,
        pinch_ratio=pinch,
        motion_eligible=motion_eligible,
    )
    pose = EndEffectorPose(0.1, 0.0, 0.2, 0.0)
    return DryRunStep(
        target_update=update,
        desired_pose=pose,
        ik_result=None,
        output_joints_rad={
            "base": 0.0,
            "shoulder": -0.3,
            "elbow": -0.5,
            "wrist": -1.0,
        },
        output_pose=pose,
        output_position_error_m=0.0,
        output_pitch_error_rad=0.0,
        moved=False,
        slewing=False,
        hold_reason=hold_reason,
    )


def coordinator() -> tuple[Stage3Coordinator, FakeClient]:
    client = FakeClient()
    clutch = SimpleNamespace(release_ratio=0.45, debounce_frames=3)
    return Stage3Coordinator(client, clutch), client


def test_requires_three_explicit_open_frames_before_arm() -> None:
    control, client = coordinator()
    open_step = step(
        TrackingState.HOVER,
        pinch=0.6,
        motion_eligible=False,
        hold_reason="tracking_hover",
    )
    for index in range(3):
        control.process_step(open_step, fresh_result=True, now_s=index * 0.01)

    assert client.holds[-1] == "operator_released"
    assert control.status.release_required is False

    reset = step(
        TrackingState.CLUTCHED,
        pinch=0.2,
        motion_eligible=True,
        hold_reason="clutch_reset",
    )
    control.process_step(reset, fresh_result=True, now_s=0.04)

    assert client.arm_requests == 1
    assert control.status.arm_requested is True


def test_pinching_from_start_never_arms_without_open_release() -> None:
    control, client = coordinator()
    reset = step(
        TrackingState.CLUTCHED,
        pinch=0.2,
        motion_eligible=True,
        hold_reason="clutch_reset",
    )

    control.process_step(reset, fresh_result=True, now_s=1.0)

    assert client.arm_requests == 0
    assert client.holds[-1] == "release_required"


def test_target_requires_arm_ack_and_short_no_result_keeps_latest() -> None:
    control, client = coordinator()
    client.server_state = "ARMED"
    commanding = step(
        TrackingState.CLUTCHED,
        pinch=0.2,
        motion_eligible=True,
        hold_reason=None,
    )
    control.process_step(commanding, fresh_result=True, now_s=1.0)
    assert len(client.candidates) == 1

    short_gap = step(
        TrackingState.CLUTCHED,
        pinch=0.2,
        motion_eligible=False,
        hold_reason="no_fresh_result",
    )
    hold_count = len(client.holds)
    control.process_step(short_gap, fresh_result=False, now_s=1.03)
    assert len(client.holds) == hold_count

    stale = step(
        TrackingState.CLUTCHED,
        pinch=0.2,
        motion_eligible=False,
        hold_reason="stale_result",
    )
    control.process_step(stale, fresh_result=False, now_s=1.2)
    assert client.holds[-1] == "stale_result"


def test_transport_failure_is_terminal() -> None:
    control, client = coordinator()
    client.failure = "link down"
    with pytest.raises(RemoteControlError, match="link down"):
        control.process_step(
            step(
                TrackingState.HOVER,
                pinch=0.6,
                motion_eligible=False,
                hold_reason="tracking_hover",
            ),
            fresh_result=True,
            now_s=1.0,
        )


def test_force_hold_clears_release_latch_and_requires_new_open_hand() -> None:
    control, client = coordinator()
    open_step = step(
        TrackingState.HOVER,
        pinch=0.6,
        motion_eligible=False,
        hold_reason="tracking_hover",
    )
    for index in range(3):
        control.process_step(open_step, fresh_result=True, now_s=index * 0.01)
    assert control.status.release_required is False

    control.force_hold("camera_frame_missing")

    assert client.holds[-1] == "camera_frame_missing"
    assert control.status.release_required is True
