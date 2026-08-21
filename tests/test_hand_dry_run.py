"""Deterministic unit tests for the Stage 2 dry-run state controller."""
from __future__ import annotations

import builtins
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from butter_finger.teleoperation.dry_run import DryRunTeleoperationController
from butter_finger.teleoperation.types import (
    EndEffectorPose,
    IKResult,
    IKStatus,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)


def test_default_controller_never_imports_external_robot_runtimes(
    monkeypatch,
) -> None:
    from butter_finger.teleoperation.config import load_teleoperation_config

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"pybullet", "ros_robot_controller_sdk"}:
            raise AssertionError(f"Stage 2 must not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    controller = DryRunTeleoperationController(load_teleoperation_config())

    assert set(controller.output_joints_rad) == {
        "base",
        "shoulder",
        "elbow",
        "wrist",
    }


@dataclass
class FakeModel:
    anchor_joints: dict[str, float]

    def __init__(self) -> None:
        self.anchor_joints = {"base": 0.0, "wrist": 0.0}
        self.anchor_pose = EndEffectorPose(0.0, 0.0, 0.0, 0.0)

    def forward(self, joints: dict[str, float]) -> EndEffectorPose:
        return EndEffectorPose(joints["base"], 0.0, 0.0, joints["wrist"])

    def target_from_virtual_delta(
        self,
        target: VirtualEETarget,
        initial: VirtualEETarget,
    ) -> EndEffectorPose:
        return EndEffectorPose(
            target.x_m - initial.x_m,
            target.y_m - initial.y_m,
            target.z_m - initial.z_m,
            target.pitch_rad - initial.pitch_rad,
        )


class FakeSolver:
    def __init__(self) -> None:
        self.results: list[IKResult] = []
        self.calls: list[tuple[EndEffectorPose, tuple[dict[str, float], ...], dict[str, float]]] = []

    def push(
        self,
        target: EndEffectorPose,
        *,
        joints: dict[str, float] | None,
        status: IKStatus = IKStatus.SOLVED,
    ) -> None:
        achieved = target if joints is not None else EndEffectorPose(0.0, 0.0, 0.0, 0.0)
        self.results.append(
            IKResult(
                status=status,
                target_pose=target,
                achieved_pose=achieved,
                joints_rad=joints,
                position_error_m=0.0 if joints is not None else 1.0,
                pitch_error_rad=0.0,
                iterations=3,
            )
        )

    def solve(
        self,
        target: EndEffectorPose,
        seeds: tuple[dict[str, float], ...],
        reference_joints: dict[str, float],
    ) -> IKResult:
        self.calls.append((target, seeds, reference_joints))
        return self.results.pop(0)


def config() -> SimpleNamespace:
    initial = VirtualEETarget(0.18, 0.0, 0.14, 0.0)
    return SimpleNamespace(
        workspace=SimpleNamespace(initial_target=initial),
        hand_tracking=SimpleNamespace(max_result_age_s=0.15),
        ik=SimpleNamespace(
            joint_rate_limits_rad_s={"base": 0.35, "wrist": 0.35},
            max_solution_jump_rad=0.75,
            max_slew_dt_s=0.10,
        ),
    )


def update(
    state: TrackingState = TrackingState.CLUTCHED,
    *,
    dx: float = 0.0,
    pitch: float = 0.0,
    motion_eligible: bool | None = None,
) -> TargetUpdate:
    target = VirtualEETarget(0.18 + dx, 0.0, 0.14, pitch)
    return TargetUpdate(
        state=state,
        target=target,
        raw_target=target,
        motion_eligible=(
            state is TrackingState.CLUTCHED
            if motion_eligible is None
            else motion_eligible
        ),
    )


def make_controller() -> tuple[DryRunTeleoperationController, FakeSolver]:
    solver = FakeSolver()
    controller = DryRunTeleoperationController(config(), FakeModel(), solver)
    return controller, solver


def test_rejects_stage01_config_without_ik() -> None:
    incomplete = config()
    incomplete.ik = None

    with pytest.raises(ValueError, match="requires a teleoperation.ik"):
        DryRunTeleoperationController(incomplete, FakeModel(), FakeSolver())


def test_starts_at_anchor_and_first_clutched_frame_only_resets() -> None:
    controller, solver = make_controller()

    step = controller.step(update(dx=0.2), fresh_result=True, now_s=1.0)

    assert step.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert step.output_pose == EndEffectorPose(0.0, 0.0, 0.0, 0.0)
    assert step.hold_reason == "clutch_reset"
    assert not step.moved
    assert solver.calls == []


def test_slew_limited_output_and_raw_ik_have_separate_residuals() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)
    target = EndEffectorPose(0.2, 0.0, 0.0, 0.1)
    solver.push(target, joints={"base": 0.2, "wrist": 0.1})

    step = controller.step(
        update(dx=0.2, pitch=0.1), fresh_result=True, now_s=1.1
    )

    assert step.ik_result is not None
    assert step.ik_result.position_error_m == 0.0
    assert step.output_joints_rad == pytest.approx({"base": 0.035, "wrist": 0.035})
    assert step.output_position_error_m == pytest.approx(0.165)
    assert step.output_pitch_error_rad == pytest.approx(0.065)
    assert step.moved
    assert step.slewing
    assert step.hold_reason is None
    _, seeds, reference = solver.calls[0]
    assert seeds == (
        {"base": 0.0, "wrist": 0.0},
        {"base": 0.0, "wrist": 0.0},
        {"base": 0.0, "wrist": 0.0},
    )
    assert reference == {"base": 0.0, "wrist": 0.0}


def test_no_fresh_result_holds_but_short_gap_stays_armed() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)
    held = controller.step(update(dx=0.2), fresh_result=False, now_s=1.05)
    target = EndEffectorPose(0.2, 0.0, 0.0, 0.0)
    solver.push(target, joints={"base": 0.2, "wrist": 0.0})

    moved = controller.step(update(dx=0.2), fresh_result=True, now_s=1.1)

    assert held.hold_reason == "no_fresh_result"
    assert held.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert moved.output_joints_rad["base"] == pytest.approx(0.035)
    assert len(solver.calls) == 1


def test_clutched_but_motion_ineligible_freezes_and_disarms() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)

    held = controller.step(
        update(dx=0.2, motion_eligible=False),
        fresh_result=True,
        now_s=1.05,
    )
    reset = controller.step(update(dx=0.2), fresh_result=True, now_s=1.10)

    assert held.hold_reason == "motion_ineligible"
    assert held.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert reset.hold_reason == "clutch_reset"
    assert solver.calls == []


def test_nonfinite_target_freezes_as_numerical_failure() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)

    step = controller.step(
        update(dx=float("nan")),
        fresh_result=True,
        now_s=1.1,
    )

    assert step.hold_reason == "ik_numerical_failure"
    assert step.ik_result is not None
    assert step.ik_result.status is IKStatus.NUMERICAL_FAILURE
    assert step.ik_result.joints_rad is None
    assert math.isinf(step.output_position_error_m)
    assert math.isinf(step.output_pitch_error_rad)
    assert step.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert solver.calls == []


def test_nonfinite_target_in_hold_never_escapes_freeze_boundary() -> None:
    controller, solver = make_controller()

    step = controller.step(
        update(TrackingState.HOLD, dx=float("nan")),
        fresh_result=True,
        now_s=1.0,
    )

    assert step.hold_reason == "tracking_hold"
    assert step.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert solver.calls == []


def test_stale_gap_disarms_and_recovery_frame_does_not_move() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)
    stale = controller.step(update(dx=0.2), fresh_result=False, now_s=1.2)
    recovered = controller.step(update(dx=0.2), fresh_result=True, now_s=1.21)

    assert stale.hold_reason == "stale_result"
    assert recovered.hold_reason == "stale_reset"
    assert recovered.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert solver.calls == []


def test_long_gap_detected_on_next_fresh_frame_is_reset_only() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)

    recovered = controller.step(update(dx=0.2), fresh_result=True, now_s=1.3)

    assert recovered.hold_reason == "stale_reset"
    assert recovered.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert solver.calls == []


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (TrackingState.HOLD, "tracking_hold"),
        (TrackingState.LOST, "tracking_lost"),
        (TrackingState.HOVER, "tracking_hover"),
    ],
)
def test_non_clutched_state_freezes_and_requires_reclutch_reset(
    state: TrackingState,
    reason: str,
) -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)

    held = controller.step(update(state), fresh_result=True, now_s=1.05)
    reset = controller.step(update(dx=0.2), fresh_result=True, now_s=1.1)

    assert held.hold_reason == reason
    assert reset.hold_reason == "clutch_reset"
    assert solver.calls == []


def test_non_monotonic_time_resets_without_invoking_ik() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)

    reset = controller.step(update(dx=0.2), fresh_result=True, now_s=0.9)

    assert reset.hold_reason == "non_monotonic_time"
    assert reset.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert solver.calls == []


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (IKStatus.UNREACHABLE, "ik_unreachable"),
        (IKStatus.NUMERICAL_FAILURE, "ik_numerical_failure"),
    ],
)
def test_failed_ik_freezes_output(status: IKStatus, reason: str) -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)
    target = EndEffectorPose(0.2, 0.0, 0.0, 0.0)
    solver.push(target, joints=None, status=status)

    step = controller.step(update(dx=0.2), fresh_result=True, now_s=1.1)

    assert step.hold_reason == reason
    assert step.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert not step.moved


def test_solution_branch_jump_is_rejected() -> None:
    controller, solver = make_controller()
    controller.step(update(), fresh_result=True, now_s=1.0)
    target = EndEffectorPose(1.0, 0.0, 0.0, 0.0)
    solver.push(target, joints={"base": 1.0, "wrist": 0.0})

    step = controller.step(update(dx=1.0), fresh_result=True, now_s=1.1)

    assert step.ik_result is not None
    assert step.ik_result.status is IKStatus.SOLVED
    assert step.hold_reason == "solution_jump"
    assert step.output_joints_rad == {"base": 0.0, "wrist": 0.0}
    assert not step.moved


def test_real_model_solver_and_controller_integrate_without_initial_jump() -> None:
    from butter_finger.teleoperation.config import load_teleoperation_config

    real_config = load_teleoperation_config()
    controller = DryRunTeleoperationController(real_config)
    anchor_joints = controller.output_joints_rad
    initial = real_config.workspace.initial_target
    initial_update = TargetUpdate(
        TrackingState.CLUTCHED,
        target=initial,
        raw_target=initial,
        motion_eligible=True,
    )

    reset = controller.step(initial_update, fresh_result=True, now_s=1.0)
    settled = controller.step(initial_update, fresh_result=True, now_s=1.0 + 1 / 30)
    moved_target = VirtualEETarget(
        initial.x_m + 0.02,
        initial.y_m,
        initial.z_m,
        initial.pitch_rad,
    )
    moving = controller.step(
        TargetUpdate(
            TrackingState.CLUTCHED,
            target=moved_target,
            raw_target=moved_target,
            motion_eligible=True,
        ),
        fresh_result=True,
        now_s=1.0 + 2 / 30,
    )

    assert reset.hold_reason == "clutch_reset"
    assert reset.output_joints_rad == pytest.approx(anchor_joints)
    assert settled.ik_result is not None
    assert settled.ik_result.status is IKStatus.SOLVED
    assert not settled.moved
    assert moving.ik_result is not None
    assert moving.ik_result.status is IKStatus.SOLVED
    assert moving.moved and moving.slewing
    for joint, angle in moving.output_joints_rad.items():
        max_delta = real_config.ik.joint_rate_limits_rad_s[joint] / 30
        assert abs(angle - reset.output_joints_rad[joint]) <= max_delta + 1e-12
