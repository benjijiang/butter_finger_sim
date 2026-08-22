"""Pure-NumPy wrist/end-effector forward and inverse kinematics for teleop.

The CAD transforms loaded here have not been independently validated against
the physical arm.  Results from this module are configuration-domain
candidates only; this module deliberately has no arm-backend integration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from butter_finger.config import CONFIG_DIR, JointLimits, load_arm_config
from butter_finger.teleoperation.config import IKConfig
from butter_finger.teleoperation.types import (
    EndEffectorPose,
    IKResult,
    IKStatus,
    VirtualEETarget,
)


_END_EFFECTOR_FORWARD = np.array((0.0, 1.0, 0.0), dtype=float)
_BASE_UP = np.array((0.0, 0.0, 1.0), dtype=float)
_LIMIT_EPSILON_RAD = 1e-10


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector3(value: object, label: str, *, nonzero: bool = False) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    result = np.array(
        [_finite_float(component, label) for component in value], dtype=float
    )
    if nonzero and float(np.linalg.norm(result)) <= 0.0:
        raise ValueError(f"{label} must be non-zero")
    return result


def _translation_rotation(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    """Return URDF fixed-axis RPY as Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _pose_values(pose: EndEffectorPose) -> np.ndarray:
    return np.array((pose.x_m, pose.y_m, pose.z_m, pose.pitch_rad), dtype=float)


@dataclass(frozen=True)
class _JointTransform:
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class _Attempt:
    converged: bool
    joints: np.ndarray
    pose: EndEffectorPose
    position_error_m: float
    pitch_error_rad: float
    objective: float
    iterations: int
    active_limits: tuple[str, ...]
    numerical_failure: bool = False


class KinematicModel:
    """Serial four-joint model ending at the configured wrist/tool frame."""

    def __init__(
        self,
        *,
        ik_config: IKConfig,
        joint_order: Sequence[str],
        limits: Mapping[str, JointLimits],
        anchor_joints: Mapping[str, float],
        joint_transforms: Sequence[_JointTransform],
        end_effector_transform: np.ndarray,
    ) -> None:
        self.ik_config = ik_config
        self.joint_order = tuple(joint_order)
        self.limits = dict(limits)
        self.anchor_joints = {
            name: float(anchor_joints[name]) for name in self.joint_order
        }
        self._joint_transforms = tuple(joint_transforms)
        self._end_effector_transform = np.array(
            end_effector_transform, dtype=float, copy=True
        )
        # Triangle inequality over every fixed translation gives a conservative
        # position-radius bound, independent of joint angles and limits.
        self.max_position_radius_m = float(
            sum(
                np.linalg.norm(joint.origin[:3, 3])
                for joint in self._joint_transforms
            )
            + np.linalg.norm(self._end_effector_transform[:3, 3])
        )

        anchor_transform = self._transform(self._joint_vector(self.anchor_joints))
        anchor_forward = anchor_transform[:3, :3] @ _END_EFFECTOR_FORWARD
        horizontal = anchor_forward - float(anchor_forward @ _BASE_UP) * _BASE_UP
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 1e-9:
            raise ValueError(
                f"anchor pose {ik_config.anchor_pose!r} has a vertical end-effector axis"
            )
        self._task_x = horizontal / horizontal_norm
        self._task_z = _BASE_UP.copy()
        self._task_y = np.cross(self._task_z, self._task_x)
        self.task_x = tuple(float(value) for value in self._task_x)
        self.task_y = tuple(float(value) for value in self._task_y)
        self.task_z = tuple(float(value) for value in self._task_z)
        # Explicit aliases make the frame semantics clear to callers.
        self.task_x_axis = self.task_x
        self.task_y_axis = self.task_y
        self.task_z_axis = self.task_z
        self.anchor_pose = self.forward(self.anchor_joints)

    @classmethod
    def from_config(
        cls,
        ik_config: IKConfig,
        config_dir: Path = CONFIG_DIR,
    ) -> "KinematicModel":
        """Load and cross-check CAD geometry, limits, and the anchor pose."""
        arm = load_arm_config(config_dir)
        if ik_config.anchor_pose not in arm.poses:
            raise ValueError(f"unknown IK anchor pose {ik_config.anchor_pose!r}")

        geometry_path = config_dir / "geometry.yaml"
        raw = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("geometry.yaml must contain a mapping")
        geometry_joints = raw.get("joints")
        if not isinstance(geometry_joints, dict):
            raise ValueError("geometry.joints must be a mapping")

        transforms: list[_JointTransform] = []
        expected_parent = "base_link"
        for logical_name in arm.joint_order:
            urdf_name = f"{logical_name}_joint"
            joint = geometry_joints.get(urdf_name)
            if not isinstance(joint, dict):
                raise ValueError(f"geometry.joints.{urdf_name} must be a mapping")
            if joint.get("parent") != expected_parent:
                raise ValueError(
                    f"{urdf_name}.parent must be {expected_parent!r} for the "
                    "serial CAD chain"
                )
            child = joint.get("child")
            if not isinstance(child, str) or not child:
                raise ValueError(f"{urdf_name}.child must be a non-empty string")

            xyz = _vector3(joint.get("origin_xyz"), f"{urdf_name}.origin_xyz")
            rpy = _vector3(joint.get("origin_rpy"), f"{urdf_name}.origin_rpy")
            axis = _vector3(
                joint.get("axis_xyz"), f"{urdf_name}.axis_xyz", nonzero=True
            )
            axis /= np.linalg.norm(axis)
            transforms.append(
                _JointTransform(
                    origin=_translation_rotation(xyz, _rpy_rotation(rpy)),
                    axis=axis,
                )
            )
            expected_parent = child

        expected_joint_names = {f"{name}_joint" for name in arm.joint_order}
        if set(geometry_joints) != expected_joint_names:
            raise ValueError(
                "geometry.joints must contain exactly the configured arm joints"
            )

        camera = raw.get("camera_mount")
        if not isinstance(camera, dict):
            raise ValueError("geometry.camera_mount must be a mapping")
        if camera.get("parent") != expected_parent:
            raise ValueError(
                f"camera_mount.parent must be {expected_parent!r}, got "
                f"{camera.get('parent')!r}"
            )
        endpoint = raw.get("teleoperation_endpoint")
        if not isinstance(endpoint, dict):
            raise ValueError("geometry.teleoperation_endpoint must be a mapping")
        if endpoint.get("parent") != expected_parent:
            raise ValueError(
                f"teleoperation_endpoint.parent must be {expected_parent!r}, got "
                f"{endpoint.get('parent')!r}"
            )

        if ik_config.end_effector == expected_parent:
            # The wrist-link origin is the endpoint of the actuated serial
            # chain, immediately after the wrist joint transform.
            end_effector_transform = np.eye(4, dtype=float)
        elif ik_config.end_effector == endpoint.get("child"):
            endpoint_xyz = _vector3(
                endpoint.get("origin_xyz"), "teleoperation_endpoint.origin_xyz"
            )
            endpoint_rpy = _vector3(
                endpoint.get("origin_rpy"), "teleoperation_endpoint.origin_rpy"
            )
            end_effector_transform = _translation_rotation(
                endpoint_xyz, _rpy_rotation(endpoint_rpy)
            )
        elif ik_config.end_effector == camera.get("child"):
            camera_xyz = _vector3(
                camera.get("origin_xyz"), "camera_mount.origin_xyz"
            )
            camera_rpy = _vector3(
                camera.get("origin_rpy"), "camera_mount.origin_rpy"
            )
            end_effector_transform = _translation_rotation(
                camera_xyz, _rpy_rotation(camera_rpy)
            )
        else:
            raise ValueError(
                "configured IK end_effector must be wrist_link, wrist_tip, "
                "or camera_link"
            )

        anchor = arm.poses[ik_config.anchor_pose]
        if set(anchor) != set(arm.joint_order):
            raise ValueError(
                f"anchor pose {ik_config.anchor_pose!r} must contain every joint"
            )
        for name in arm.joint_order:
            value = _finite_float(anchor[name], f"anchor_pose.{name}")
            if not arm.sim_limits[name].contains(value):
                raise ValueError(
                    f"anchor pose joint {name}={value} is outside its limits"
                )

        return cls(
            ik_config=ik_config,
            joint_order=arm.joint_order,
            limits=arm.sim_limits,
            anchor_joints=anchor,
            joint_transforms=transforms,
            end_effector_transform=end_effector_transform,
        )

    def forward(self, joints: Mapping[str, float]) -> EndEffectorPose:
        """Return the configured endpoint position and task-frame pitch."""
        vector = self._joint_vector(joints)
        return self._pose_from_transform(self._transform(vector))

    def forward_component(self, joints: Mapping[str, float]) -> float:
        """Return camera-forward dot task-X; low values are invalid IK branches."""
        vector = self._joint_vector(joints)
        transform = self._transform(vector)
        forward = transform[:3, :3] @ _END_EFFECTOR_FORWARD
        return float(forward @ self._task_x)

    def target_from_virtual_delta(
        self,
        target: VirtualEETarget,
        initial: VirtualEETarget,
    ) -> EndEffectorPose:
        """Anchor a Stage 1 relative target in the fixed idle task frame."""
        target_values = np.array(
            (target.x_m, target.y_m, target.z_m, target.pitch_rad), dtype=float
        )
        initial_values = np.array(
            (initial.x_m, initial.y_m, initial.z_m, initial.pitch_rad), dtype=float
        )
        if not np.all(np.isfinite(target_values)):
            raise ValueError("virtual target values must be finite")
        if not np.all(np.isfinite(initial_values)):
            raise ValueError("initial virtual target values must be finite")
        delta = target_values - initial_values
        position = (
            np.array(
                (
                    self.anchor_pose.x_m,
                    self.anchor_pose.y_m,
                    self.anchor_pose.z_m,
                )
            )
            + self._task_x * delta[0]
            + self._task_y * delta[1]
            + self._task_z * delta[2]
        )
        return EndEffectorPose(
            x_m=float(position[0]),
            y_m=float(position[1]),
            z_m=float(position[2]),
            # Keep the anchor plus relative increment literally so a zero
            # Stage 1 delta is bit-for-bit the idle anchor.  IK residuals wrap
            # angles when comparing equivalent orientations.
            pitch_rad=self.anchor_pose.pitch_rad + float(delta[3]),
        )

    def midpoint_joints(self) -> dict[str, float]:
        return {
            name: (self.limits[name].lower_rad + self.limits[name].upper_rad) / 2.0
            for name in self.joint_order
        }

    def _joint_vector(self, joints: Mapping[str, float]) -> np.ndarray:
        if not isinstance(joints, Mapping):
            raise ValueError("joints must be a mapping")
        if set(joints) != set(self.joint_order):
            raise ValueError(
                f"joints must contain exactly {self.joint_order}, got "
                f"{tuple(joints)}"
            )
        vector = np.array(
            [_finite_float(joints[name], f"joints.{name}") for name in self.joint_order]
        )
        for index, name in enumerate(self.joint_order):
            if not self.limits[name].contains(float(vector[index])):
                limit = self.limits[name]
                raise ValueError(
                    f"joint {name}={vector[index]} is outside "
                    f"[{limit.lower_rad}, {limit.upper_rad}]"
                )
        return vector

    def _transform(self, joint_vector: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        for joint, angle in zip(self._joint_transforms, joint_vector):
            rotation = _translation_rotation(
                np.zeros(3), _axis_angle_rotation(joint.axis, float(angle))
            )
            # URDF semantics: Trans(xyz) Rz(yaw) Ry(pitch) Rx(roll) Rot(axis,q).
            transform = transform @ joint.origin @ rotation
        return transform @ self._end_effector_transform

    def _pose_from_transform(self, transform: np.ndarray) -> EndEffectorPose:
        position = transform[:3, 3]
        forward = transform[:3, :3] @ _END_EFFECTOR_FORWARD
        pitch = math.atan2(
            float(forward @ self._task_z), float(forward @ self._task_x)
        )
        return EndEffectorPose(
            x_m=float(position[0]),
            y_m=float(position[1]),
            z_m=float(position[2]),
            pitch_rad=pitch,
        )

    def _unchecked_pose(self, joint_vector: np.ndarray) -> EndEffectorPose:
        return self._pose_from_transform(self._transform(joint_vector))

    def _unchecked_forward_component(self, joint_vector: np.ndarray) -> float:
        transform = self._transform(joint_vector)
        forward = transform[:3, :3] @ _END_EFFECTOR_FORWARD
        return float(forward @ self._task_x)


class IKSolver:
    """Deterministic, bounded damped-least-squares IK for the endpoint pose."""

    def __init__(
        self,
        model: KinematicModel,
        config: IKConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config if config is not None else model.ik_config
        self._lower = np.array(
            [model.limits[name].lower_rad for name in model.joint_order], dtype=float
        )
        self._upper = np.array(
            [model.limits[name].upper_rad for name in model.joint_order], dtype=float
        )
        self._span = self._upper - self._lower

    def solve(
        self,
        target: EndEffectorPose,
        seeds: Iterable[Mapping[str, float]],
        reference_joints: Mapping[str, float],
    ) -> IKResult:
        """Solve deterministic seeds and rank candidates near the reference."""
        try:
            target_values = _pose_values(target)
            if not np.all(np.isfinite(target_values)):
                raise ValueError("IK target values must be finite")
            reference = self.model._joint_vector(reference_joints)
        except (TypeError, ValueError):
            return self._failure_result(
                IKStatus.NUMERICAL_FAILURE,
                target,
                self.model.anchor_joints,
                iterations=0,
            )

        closest_allowed_pitch_to_forward = max(
            0.0,
            abs(_wrap_angle(target.pitch_rad))
            - self.config.pitch_tolerance_rad,
        )
        if (
            float(np.linalg.norm(target_values[:3]))
            > self.model.max_position_radius_m
            + self.config.position_tolerance_m
            or math.cos(closest_allowed_pitch_to_forward)
            < self.config.min_forward_component
        ):
            return self._failure_result(
                IKStatus.UNREACHABLE,
                target,
                reference_joints,
                iterations=0,
            )

        all_seeds: list[Mapping[str, float]] = list(seeds)
        all_seeds.extend((self.model.anchor_joints, self.model.midpoint_joints()))
        seed_vectors: list[np.ndarray] = []
        for seed in all_seeds:
            vector = self._seed_vector(seed)
            if vector is None:
                continue
            if not any(np.array_equal(vector, existing) for existing in seed_vectors):
                seed_vectors.append(vector)

        if not seed_vectors:
            return self._failure_result(
                IKStatus.NUMERICAL_FAILURE,
                target,
                reference_joints,
                iterations=0,
            )

        attempts = [self._solve_seed(target, seed) for seed in seed_vectors]
        # A mathematically converged solution remains a raw IK success even
        # when it is far from the previous command candidate.  The dry-run
        # state controller owns the temporal branch-jump policy so diagnostics
        # can distinguish SOLVED + solution_jump from a truly unreachable
        # Cartesian target.
        converged = [attempt for attempt in attempts if attempt.converged]
        if converged:
            chosen = min(
                enumerate(converged),
                key=lambda indexed: (
                    float(
                        np.linalg.norm(
                            (indexed[1].joints - reference) / self._span
                        )
                    ),
                    indexed[0],
                    tuple(float(value) for value in indexed[1].joints),
                ),
            )[1]
            joints = self._joint_mapping(chosen.joints)
            return IKResult(
                status=IKStatus.SOLVED,
                target_pose=target,
                achieved_pose=chosen.pose,
                joints_rad=joints,
                position_error_m=chosen.position_error_m,
                pitch_error_rad=chosen.pitch_error_rad,
                iterations=chosen.iterations,
                active_limits=chosen.active_limits,
            )

        best = min(
            attempts,
            key=lambda attempt: (
                attempt.objective,
                attempt.iterations,
                tuple(float(value) for value in attempt.joints),
            ),
        )
        # Iteration exhaustion is not proof that a target is unreachable. Keep
        # that diagnostic conservative and report it as a solver failure. The
        # early position/pitch bounds above own provable kinematic rejection.
        return IKResult(
            status=IKStatus.NUMERICAL_FAILURE,
            target_pose=target,
            achieved_pose=best.pose,
            joints_rad=None,
            position_error_m=best.position_error_m,
            pitch_error_rad=best.pitch_error_rad,
            iterations=best.iterations,
            active_limits=best.active_limits,
        )

    def _solve_seed(self, target: EndEffectorPose, seed: np.ndarray) -> _Attempt:
        q = np.clip(seed, self._lower, self._upper)
        damping = self.config.initial_damping
        best = self._attempt_state(target, q, iterations=0)
        active_limits: tuple[str, ...] = best.active_limits

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                current = self._attempt_state(target, q, iterations=iteration - 1)
                if self._is_converged(current, q):
                    return _Attempt(
                        True,
                        q.copy(),
                        current.pose,
                        current.position_error_m,
                        current.pitch_error_rad,
                        current.objective,
                        iteration - 1,
                        self._limit_names(q),
                    )
                if current.objective < best.objective:
                    best = current

                jacobian = self._central_jacobian(q)
                error = self._weighted_error(target, current.pose)
                dq = self._dls_step(jacobian, error, damping)
                active = self._outward_active_set(q, dq)
                if np.any(active):
                    free = ~active
                    dq = np.zeros_like(q)
                    if np.any(free):
                        reduced = jacobian[:, free]
                        dq[free] = self._dls_step(reduced, error, damping)
                    active_limits = tuple(
                        name
                        for name, is_active in zip(self.model.joint_order, active)
                        if is_active
                    )
                dq = np.clip(
                    dq,
                    -self.config.max_iteration_step_rad,
                    self.config.max_iteration_step_rad,
                )
                if not np.all(np.isfinite(dq)):
                    raise FloatingPointError("non-finite IK update")
                if float(np.linalg.norm(dq)) <= 1e-14:
                    break

                accepted = False
                for level in range(self.config.max_backtracking_steps):
                    scale = 0.5**level
                    candidate_q = np.clip(q + scale * dq, self._lower, self._upper)
                    candidate = self._attempt_state(
                        target, candidate_q, iterations=iteration
                    )
                    if candidate.objective + 1e-14 < current.objective:
                        q = candidate_q
                        if candidate.objective < best.objective:
                            best = candidate
                        damping = max(self.config.min_damping, damping * 0.5)
                        accepted = True
                        break
                if not accepted:
                    damping = min(self.config.max_damping, damping * 10.0)
                    if damping >= self.config.max_damping:
                        break

            final = self._attempt_state(
                target, q, iterations=min(self.config.max_iterations, iteration)
            )
            if self._is_converged(final, q):
                return _Attempt(
                    True,
                    q.copy(),
                    final.pose,
                    final.position_error_m,
                    final.pitch_error_rad,
                    final.objective,
                    final.iterations,
                    self._limit_names(q),
                )
            if final.objective < best.objective:
                best = final
            return _Attempt(
                False,
                best.joints,
                best.pose,
                best.position_error_m,
                best.pitch_error_rad,
                best.objective,
                best.iterations,
                active_limits or best.active_limits,
            )
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return _Attempt(
                False,
                best.joints,
                best.pose,
                best.position_error_m,
                best.pitch_error_rad,
                best.objective,
                best.iterations,
                best.active_limits,
                numerical_failure=True,
            )

    def _attempt_state(
        self,
        target: EndEffectorPose,
        q: np.ndarray,
        *,
        iterations: int,
    ) -> _Attempt:
        pose = self.model._unchecked_pose(q)
        position_error = float(
            np.linalg.norm(_pose_values(target)[:3] - _pose_values(pose)[:3])
        )
        pitch_error = abs(_wrap_angle(target.pitch_rad - pose.pitch_rad))
        objective = float(np.linalg.norm(self._weighted_error(target, pose)))
        if not all(math.isfinite(value) for value in (position_error, pitch_error, objective)):
            raise FloatingPointError("non-finite IK residual")
        return _Attempt(
            False,
            q.copy(),
            pose,
            position_error,
            pitch_error,
            objective,
            iterations,
            self._limit_names(q),
        )

    def _central_jacobian(self, q: np.ndarray) -> np.ndarray:
        step = self.config.finite_difference_step_rad
        jacobian = np.empty((4, len(q)), dtype=float)
        for column in range(len(q)):
            plus = q.copy()
            minus = q.copy()
            plus[column] += step
            minus[column] -= step
            plus_pose = self.model._unchecked_pose(plus)
            minus_pose = self.model._unchecked_pose(minus)
            plus_values = _pose_values(plus_pose)
            minus_values = _pose_values(minus_pose)
            jacobian[:3, column] = (plus_values[:3] - minus_values[:3]) / (
                2.0 * step
            )
            jacobian[3, column] = (
                self.config.orientation_weight_m_per_rad
                * _wrap_angle(plus_values[3] - minus_values[3])
                / (2.0 * step)
            )
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("non-finite IK Jacobian")
        return jacobian

    def _weighted_error(
        self, target: EndEffectorPose, achieved: EndEffectorPose
    ) -> np.ndarray:
        target_values = _pose_values(target)
        achieved_values = _pose_values(achieved)
        return np.array(
            (
                *(target_values[:3] - achieved_values[:3]),
                self.config.orientation_weight_m_per_rad
                * _wrap_angle(target.pitch_rad - achieved.pitch_rad),
            ),
            dtype=float,
        )

    @staticmethod
    def _dls_step(
        jacobian: np.ndarray, error: np.ndarray, damping: float
    ) -> np.ndarray:
        regularized = jacobian @ jacobian.T + (damping**2) * np.eye(4)
        return jacobian.T @ np.linalg.solve(regularized, error)

    def _is_converged(self, attempt: _Attempt, q: np.ndarray) -> bool:
        return (
            attempt.position_error_m <= self.config.position_tolerance_m
            and attempt.pitch_error_rad <= self.config.pitch_tolerance_rad
            and self.model._unchecked_forward_component(q)
            >= self.config.min_forward_component
        )

    def _outward_active_set(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return ((q <= self._lower + _LIMIT_EPSILON_RAD) & (dq < 0.0)) | (
            (q >= self._upper - _LIMIT_EPSILON_RAD) & (dq > 0.0)
        )

    def _limit_names(self, q: np.ndarray) -> tuple[str, ...]:
        at_limit = (q <= self._lower + _LIMIT_EPSILON_RAD) | (
            q >= self._upper - _LIMIT_EPSILON_RAD
        )
        return tuple(
            name
            for name, active in zip(self.model.joint_order, at_limit)
            if bool(active)
        )

    def _seed_vector(self, seed: Mapping[str, float]) -> np.ndarray | None:
        if not isinstance(seed, Mapping) or set(seed) != set(self.model.joint_order):
            return None
        try:
            vector = np.array(
                [float(seed[name]) for name in self.model.joint_order], dtype=float
            )
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(vector)):
            return None
        return np.clip(vector, self._lower, self._upper)

    def _joint_mapping(self, q: np.ndarray) -> dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.model.joint_order, q)
        }

    def _failure_result(
        self,
        status: IKStatus,
        target: EndEffectorPose,
        fallback_joints: Mapping[str, float],
        *,
        iterations: int,
    ) -> IKResult:
        try:
            pose = self.model.forward(fallback_joints)
        except (TypeError, ValueError):
            pose = self.model.anchor_pose
        try:
            position_error = float(
                np.linalg.norm(_pose_values(target)[:3] - _pose_values(pose)[:3])
            )
            pitch_error = abs(_wrap_angle(target.pitch_rad - pose.pitch_rad))
            if not math.isfinite(position_error):
                position_error = math.inf
            if not math.isfinite(pitch_error):
                pitch_error = math.inf
        except (AttributeError, TypeError, ValueError):
            position_error = math.inf
            pitch_error = math.inf
        return IKResult(
            status=status,
            target_pose=target,
            achieved_pose=pose,
            joints_rad=None,
            position_error_m=position_error,
            pitch_error_rad=pitch_error,
            iterations=iterations,
            active_limits=(),
        )


__all__ = ["IKSolver", "KinematicModel"]
