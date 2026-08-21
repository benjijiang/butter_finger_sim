"""Stage 2 inverse-kinematics dry-run state controller.

The controller in this module deliberately stops at joint-angle candidates.  It
does not import an arm backend, produce PWM values, or send commands anywhere.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from butter_finger.teleoperation.config import TeleoperationConfig
from butter_finger.teleoperation.types import (
    DryRunStep,
    EndEffectorPose,
    IKResult,
    IKStatus,
    TargetUpdate,
    TrackingState,
)

if TYPE_CHECKING:
    from butter_finger.teleoperation.kinematics import KinematicModel


def _wrapped_delta(angle: float, reference: float) -> float:
    return (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


def _pose_residual(
    desired: EndEffectorPose,
    achieved: EndEffectorPose,
) -> tuple[float, float]:
    position_error = math.sqrt(
        (desired.x_m - achieved.x_m) ** 2
        + (desired.y_m - achieved.y_m) ** 2
        + (desired.z_m - achieved.z_m) ** 2
    )
    pitch_error = abs(_wrapped_delta(desired.pitch_rad, achieved.pitch_rad))
    return position_error, pitch_error


class DryRunTeleoperationController:
    """Turn fresh, clutched Stage 1 updates into slew-limited IK diagnostics."""

    def __init__(
        self,
        config: TeleoperationConfig,
        model: KinematicModel | None = None,
        solver: Any | None = None,
    ) -> None:
        ik_config = config.ik
        if ik_config is None:
            raise ValueError(
                "Stage 2 dry-run requires a teleoperation.ik configuration"
            )
        # Imports stay local so importing this dependency-free controller never
        # constructs the model or solver unless the caller asks for defaults.
        if model is None:
            from butter_finger.teleoperation.kinematics import KinematicModel

            model = KinematicModel.from_config(ik_config)
        if solver is None:
            from butter_finger.teleoperation.kinematics import IKSolver

            solver = IKSolver(model, ik_config)

        self._config = config
        self._ik_config = ik_config
        self._model = model
        self._solver = solver
        self._anchor_joints = dict(model.anchor_joints)
        if not self._anchor_joints:
            raise ValueError("kinematic model anchor_joints must not be empty")

        rates = dict(ik_config.joint_rate_limits_rad_s)
        if set(rates) != set(self._anchor_joints):
            raise ValueError(
                "IK joint-rate keys must exactly match model anchor joints"
            )
        if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates.values()):
            raise ValueError("IK joint rates must be finite and positive")
        self._joint_rates = rates

        for value, label in (
            (config.hand_tracking.max_result_age_s, "tracking freshness timeout"),
            (ik_config.max_solution_jump_rad, "IK solution jump limit"),
            (ik_config.max_slew_dt_s, "IK maximum slew dt"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")

        self._output_joints = dict(self._anchor_joints)
        self._output_pose = model.forward(self._output_joints)
        self._last_desired_pose = self._output_pose
        self._previous_raw_joints = dict(self._anchor_joints)
        self._last_call_s: float | None = None
        self._last_fresh_s: float | None = None
        self._last_motion_clock_s: float | None = None
        self._armed = False

    @property
    def output_joints_rad(self) -> dict[str, float]:
        """Copy of the current configuration-domain candidate."""
        return dict(self._output_joints)

    @property
    def output_pose(self) -> EndEffectorPose:
        return self._output_pose

    def step(
        self,
        update: TargetUpdate,
        *,
        fresh_result: bool,
        now_s: float,
    ) -> DryRunStep:
        """Advance one diagnostic step without commanding any arm.

        Only a fresh ``CLUTCHED`` update may invoke IK.  Re-clutching, a stale
        interval, and a non-monotonic clock all consume one fresh frame to
        reset timing before motion can resume.
        """
        if not isinstance(fresh_result, bool):
            raise TypeError("fresh_result must be a boolean")
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")

        desired_valid = True
        try:
            desired = self._model.target_from_virtual_delta(
                update.target,
                self._config.workspace.initial_target,
            )
            desired_valid = all(
                math.isfinite(value)
                for value in (
                    desired.x_m,
                    desired.y_m,
                    desired.z_m,
                    desired.pitch_rad,
                )
            )
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            desired_valid = False
        if desired_valid:
            self._last_desired_pose = desired
        else:
            # A malformed input must never escape the dry-run freeze boundary.
            # Retain the last meaningful task-frame target for the overlay.
            desired = self._last_desired_pose

        non_monotonic = (
            self._last_call_s is not None and now_s <= self._last_call_s
        )
        stale_gap = (
            self._last_fresh_s is not None
            and now_s - self._last_fresh_s
            > self._config.hand_tracking.max_result_age_s
        )
        self._last_call_s = now_s

        if non_monotonic:
            self._armed = False
            self._last_motion_clock_s = None
            # Establish the new time epoch immediately when the regressing call
            # itself carries a result.  The frame remains a reset-only frame.
            self._last_fresh_s = now_s if fresh_result else None
            if (
                fresh_result
                and update.state is TrackingState.CLUTCHED
                and update.motion_eligible
            ):
                self._armed = True
                self._last_motion_clock_s = now_s
            return self._hold_step(update, desired, "non_monotonic_time")

        if not fresh_result:
            if stale_gap:
                self._armed = False
                self._last_motion_clock_s = None
                return self._hold_step(update, desired, "stale_result")
            return self._hold_step(update, desired, "no_fresh_result")

        self._last_fresh_s = now_s
        if update.state is not TrackingState.CLUTCHED:
            self._armed = False
            self._last_motion_clock_s = None
            return self._hold_step(
                update,
                desired,
                f"tracking_{update.state.value.lower()}",
            )

        if not update.motion_eligible:
            self._armed = False
            self._last_motion_clock_s = None
            return self._hold_step(update, desired, "motion_ineligible")

        if stale_gap:
            self._armed = True
            self._last_motion_clock_s = now_s
            return self._hold_step(update, desired, "stale_reset")

        if not self._armed or self._last_motion_clock_s is None:
            self._armed = True
            self._last_motion_clock_s = now_s
            return self._hold_step(update, desired, "clutch_reset")

        elapsed_s = now_s - self._last_motion_clock_s
        self._last_motion_clock_s = now_s
        if elapsed_s <= 0.0:
            # This is defensive; the non-monotonic branch above normally owns
            # this case.
            return self._hold_step(update, desired, "non_monotonic_time")
        dt_s = min(elapsed_s, self._ik_config.max_slew_dt_s)

        if not desired_valid:
            failure = IKResult(
                status=IKStatus.NUMERICAL_FAILURE,
                target_pose=desired,
                achieved_pose=self._output_pose,
                joints_rad=None,
                position_error_m=math.inf,
                pitch_error_rad=math.inf,
                iterations=0,
                active_limits=(),
            )
            return self._hold_step(
                update,
                desired,
                "ik_numerical_failure",
                ik_result=failure,
                residual_override=(math.inf, math.inf),
            )

        ik_result = self._solver.solve(
            desired,
            seeds=(
                dict(self._previous_raw_joints),
                dict(self._output_joints),
                dict(self._anchor_joints),
            ),
            reference_joints=dict(self._previous_raw_joints),
        )
        if ik_result.status is not IKStatus.SOLVED:
            reason = (
                "ik_unreachable"
                if ik_result.status is IKStatus.UNREACHABLE
                else "ik_numerical_failure"
            )
            return self._hold_step(update, desired, reason, ik_result=ik_result)

        raw_joints = ik_result.joints_rad
        if not self._valid_solution(raw_joints):
            return self._hold_step(
                update,
                desired,
                "ik_numerical_failure",
                ik_result=ik_result,
            )
        assert raw_joints is not None
        if any(
            abs(raw_joints[name] - self._previous_raw_joints[name])
            > self._ik_config.max_solution_jump_rad
            for name in self._anchor_joints
        ):
            return self._hold_step(
                update,
                desired,
                "solution_jump",
                ik_result=ik_result,
            )

        self._previous_raw_joints = dict(raw_joints)
        previous_output = self._output_joints
        next_output: dict[str, float] = {}
        slewing = False
        for name in self._anchor_joints:
            previous = previous_output[name]
            requested_delta = raw_joints[name] - previous
            max_delta = self._joint_rates[name] * dt_s
            applied_delta = min(max_delta, max(-max_delta, requested_delta))
            next_output[name] = previous + applied_delta
            if not math.isclose(
                applied_delta,
                requested_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                slewing = True

        moved = any(
            not math.isclose(
                next_output[name],
                previous_output[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in self._anchor_joints
        )
        self._output_joints = next_output
        self._output_pose = self._model.forward(next_output)
        position_error, pitch_error = _pose_residual(desired, self._output_pose)
        return DryRunStep(
            target_update=update,
            desired_pose=desired,
            ik_result=ik_result,
            output_joints_rad=dict(next_output),
            output_pose=self._output_pose,
            output_position_error_m=position_error,
            output_pitch_error_rad=pitch_error,
            moved=moved,
            slewing=slewing,
            hold_reason=None,
        )

    def _valid_solution(self, joints: Mapping[str, float] | None) -> bool:
        return (
            joints is not None
            and set(joints) == set(self._anchor_joints)
            and all(math.isfinite(joints[name]) for name in self._anchor_joints)
        )

    def _hold_step(
        self,
        update: TargetUpdate,
        desired: EndEffectorPose,
        reason: str,
        *,
        ik_result: IKResult | None = None,
        residual_override: tuple[float, float] | None = None,
    ) -> DryRunStep:
        position_error, pitch_error = (
            residual_override
            if residual_override is not None
            else _pose_residual(desired, self._output_pose)
        )
        return DryRunStep(
            target_update=update,
            desired_pose=desired,
            ik_result=ik_result,
            output_joints_rad=dict(self._output_joints),
            output_pose=self._output_pose,
            output_position_error_m=position_error,
            output_pitch_error_rad=pitch_error,
            moved=False,
            slewing=False,
            hold_reason=reason,
        )
