"""Pinch-clutched relative retargeting for a virtual end-effector target."""
from __future__ import annotations

import math

from butter_finger.teleoperation.config import TeleoperationConfig
from butter_finger.teleoperation.features import extract_hand_features
from butter_finger.teleoperation.types import (
    HandFeatures,
    HandObservation,
    TargetUpdate,
    TrackingState,
    VirtualEETarget,
)


def _wrapped_delta(angle: float, reference: float) -> float:
    """Smallest signed angular difference in [-pi, pi)."""
    return (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


class ExponentialTargetFilter:
    """First-order low-pass whose response is independent of loop rate."""

    def __init__(self, initial: VirtualEETarget, cutoff_hz: float) -> None:
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be finite and positive")
        self._value = initial
        self._cutoff_hz = cutoff_hz
        self._last_timestamp_s: float | None = None

    @property
    def value(self) -> VirtualEETarget:
        return self._value

    def update(self, target: VirtualEETarget) -> VirtualEETarget:
        timestamp = target.timestamp_s
        if not all(
            math.isfinite(value)
            for value in (
                target.x_m,
                target.y_m,
                target.z_m,
                target.pitch_rad,
                timestamp,
            )
        ):
            raise ValueError("target values and timestamp must be finite")
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp
            self._value = target
            return self._value
        dt = timestamp - self._last_timestamp_s
        if dt <= 0:
            return self._value
        alpha = 1.0 - math.exp(-2.0 * math.pi * self._cutoff_hz * dt)
        previous = self._value
        self._value = VirtualEETarget(
            x_m=previous.x_m + alpha * (target.x_m - previous.x_m),
            y_m=previous.y_m + alpha * (target.y_m - previous.y_m),
            z_m=previous.z_m + alpha * (target.z_m - previous.z_m),
            pitch_rad=previous.pitch_rad
            + alpha * (target.pitch_rad - previous.pitch_rad),
            timestamp_s=timestamp,
        )
        self._last_timestamp_s = timestamp
        return self._value


class VirtualTargetController:
    """Owns hand selection, clutch state, relative mapping, clamp, and filter."""

    def __init__(self, config: TeleoperationConfig) -> None:
        self._config = config
        initial = config.workspace.initial_target
        self._filter = ExponentialTargetFilter(initial, config.filter_cutoff_hz)
        self._raw_target = initial
        self._state = TrackingState.NO_HAND
        self._owner_id: str | None = None
        self._owner_handedness: str | None = None
        self._owner_center: tuple[float, float] | None = None
        self._last_owner_seen_s: float | None = None
        self._engage_counts: dict[str, int] = {}
        self._release_count = 0
        self._rearm_required = False
        self._rearm_open_count = 0
        self._ever_clutched = False
        self._baseline_features: HandFeatures | None = None
        self._baseline_target = initial
        self._last_pitch_rad: float | None = None
        self._unwrapped_pitch_rad = 0.0
        self._last_update_s: float | None = None
        self._clamped_axes: tuple[str, ...] = ()
        self._last_pinch_ratio: float | None = None

    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def target(self) -> VirtualEETarget:
        return self._filter.value

    def update(
        self,
        observations: tuple[HandObservation, ...] | list[HandObservation],
        now_s: float,
    ) -> TargetUpdate:
        """Advance from a fresh result, or an empty sequence when none is fresh."""
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_update_s is not None and now_s < self._last_update_s:
            return self._snapshot()
        self._last_update_s = now_s

        features: list[HandFeatures] = []
        for observation in observations:
            try:
                feature = extract_hand_features(observation)
            except ValueError:
                continue
            age_s = now_s - feature.timestamp_s
            if -1e-6 <= age_s <= self._config.hand_tracking.max_result_age_s:
                features.append(feature)

        if self._owner_id is not None:
            self._update_owner(features, now_s)
        else:
            self._update_unowned(features, now_s)
        return self._snapshot()

    def _update_owner(self, features: list[HandFeatures], now_s: float) -> None:
        owner = next((item for item in features if item.hand_id == self._owner_id), None)
        if owner is None and self._owner_center is not None:
            same_hand = [
                item
                for item in features
                if item.handedness.lower() == (self._owner_handedness or "").lower()
            ]
            if same_hand:
                nearest = min(
                    same_hand,
                    key=lambda item: math.hypot(
                        item.palm_u - self._owner_center[0],
                        item.palm_v - self._owner_center[1],
                    ),
                )
                distance = math.hypot(
                    nearest.palm_u - self._owner_center[0],
                    nearest.palm_v - self._owner_center[1],
                )
                if distance <= self._config.hand_tracking.owner_match_distance:
                    owner = nearest
                    self._owner_id = owner.hand_id

        if owner is None:
            if (
                self._last_owner_seen_s is not None
                and now_s - self._last_owner_seen_s >= self._config.clutch.lost_timeout_s
            ):
                self._owner_id = None
                self._owner_handedness = None
                self._owner_center = None
                self._state = TrackingState.LOST
                self._rearm_required = True
                self._rearm_open_count = 0
                self._engage_counts.clear()
            return

        self._owner_id = owner.hand_id
        self._owner_handedness = owner.handedness
        self._owner_center = (owner.palm_u, owner.palm_v)
        self._last_owner_seen_s = owner.timestamp_s
        self._last_pinch_ratio = owner.pinch_ratio

        if owner.pinch_ratio > self._config.clutch.release_ratio:
            self._release_count += 1
        else:
            self._release_count = 0
        if self._release_count >= self._config.clutch.debounce_frames:
            self._owner_id = None
            self._owner_handedness = None
            self._owner_center = None
            self._release_count = 0
            self._engage_counts.clear()
            self._state = TrackingState.HOLD
            return
        if self._release_count:
            # Freeze immediately while the release gesture is debounced. This
            # prevents the opening fingers from producing two extra motion
            # samples before the stable HOLD transition is confirmed.
            self._state = TrackingState.CLUTCHED
            return

        baseline = self._baseline_features
        if baseline is None or self._last_pitch_rad is None:
            return
        pitch_delta = _wrapped_delta(owner.palm_pitch_rad, self._last_pitch_rad)
        self._unwrapped_pitch_rad += pitch_delta
        self._last_pitch_rad = owner.palm_pitch_rad
        mapping = self._config.mapping
        requested = VirtualEETarget(
            x_m=self._baseline_target.x_m
            + mapping.gain_x_m * math.log(owner.palm_scale / baseline.palm_scale),
            y_m=self._baseline_target.y_m
            - mapping.gain_y_m * (owner.palm_u - baseline.palm_u),
            z_m=self._baseline_target.z_m
            - mapping.gain_z_m * (owner.palm_v - baseline.palm_v),
            pitch_rad=self._baseline_target.pitch_rad
            + mapping.gain_pitch * self._unwrapped_pitch_rad,
            timestamp_s=owner.timestamp_s,
        )
        self._raw_target, self._clamped_axes = self._clamp(requested)
        self._filter.update(self._raw_target)
        self._state = TrackingState.CLUTCHED

    def _update_unowned(self, features: list[HandFeatures], now_s: float) -> None:
        if self._rearm_required:
            if any(
                item.pinch_ratio > self._config.clutch.release_ratio
                for item in features
            ):
                self._rearm_open_count += 1
            else:
                self._rearm_open_count = 0
            if self._rearm_open_count >= self._config.clutch.debounce_frames:
                self._rearm_required = False
                self._rearm_open_count = 0
                self._state = TrackingState.HOVER
            else:
                self._state = TrackingState.LOST
            return

        visible_ids = {item.hand_id for item in features}
        self._engage_counts = {
            hand_id: count
            for hand_id, count in self._engage_counts.items()
            if hand_id in visible_ids
        }
        candidates: list[HandFeatures] = []
        for item in features:
            if item.pinch_ratio < self._config.clutch.engage_ratio:
                self._engage_counts[item.hand_id] = self._engage_counts.get(item.hand_id, 0) + 1
            else:
                self._engage_counts[item.hand_id] = 0
            if self._engage_counts[item.hand_id] >= self._config.clutch.debounce_frames:
                candidates.append(item)

        if candidates:
            owner = max(candidates, key=lambda item: item.confidence)
            self._engage(owner, now_s)
        elif features:
            self._state = TrackingState.HOVER
        else:
            self._state = (
                TrackingState.HOLD if self._ever_clutched else TrackingState.NO_HAND
            )

    def _engage(self, owner: HandFeatures, now_s: float) -> None:
        self._owner_id = owner.hand_id
        self._owner_handedness = owner.handedness
        self._owner_center = (owner.palm_u, owner.palm_v)
        self._last_owner_seen_s = owner.timestamp_s
        self._release_count = 0
        self._last_pinch_ratio = owner.pinch_ratio
        self._baseline_features = owner
        self._baseline_target = self._filter.value
        self._last_pitch_rad = owner.palm_pitch_rad
        self._unwrapped_pitch_rad = 0.0
        self._raw_target = VirtualEETarget(
            self._baseline_target.x_m,
            self._baseline_target.y_m,
            self._baseline_target.z_m,
            self._baseline_target.pitch_rad,
            owner.timestamp_s,
        )
        self._filter.update(self._raw_target)
        self._clamped_axes = ()
        self._ever_clutched = True
        self._state = TrackingState.CLUTCHED
        self._engage_counts.clear()

    def _clamp(
        self, requested: VirtualEETarget
    ) -> tuple[VirtualEETarget, tuple[str, ...]]:
        workspace = self._config.workspace
        values = {
            "x_m": (requested.x_m, workspace.x_m),
            "y_m": (requested.y_m, workspace.y_m),
            "z_m": (requested.z_m, workspace.z_m),
            "pitch_rad": (requested.pitch_rad, workspace.pitch_rad),
        }
        clamped = {name: limit.clamp(value) for name, (value, limit) in values.items()}
        axes = tuple(name for name, (value, _) in values.items() if clamped[name] != value)
        return (
            VirtualEETarget(**clamped, timestamp_s=requested.timestamp_s),
            axes,
        )

    def _snapshot(self) -> TargetUpdate:
        return TargetUpdate(
            state=self._state,
            target=self._filter.value,
            raw_target=self._raw_target,
            owner_hand_id=self._owner_id,
            owner_handedness=self._owner_handedness,
            pinch_ratio=self._last_pinch_ratio,
            clamped_axes=self._clamped_axes,
        )
