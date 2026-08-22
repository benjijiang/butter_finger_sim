"""Pure-math tests for Stage 2 wrist-tip FK and IK."""
from __future__ import annotations

import math

import pytest

from butter_finger.teleoperation.config import load_teleoperation_config
from butter_finger.teleoperation.kinematics import IKSolver, KinematicModel
from butter_finger.teleoperation.types import (
    EndEffectorPose,
    IKResult,
    IKStatus,
    VirtualEETarget,
)


@pytest.fixture(scope="module")
def setup() -> tuple[object, KinematicModel, IKSolver]:
    config = load_teleoperation_config()
    model = KinematicModel.from_config(config.ik)
    return config, model, IKSolver(model)


def test_idle_ready_golden_wrist_tip_pose_and_task_frame(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, _ = setup
    pose = model.anchor_pose

    assert (pose.x_m, pose.y_m, pose.z_m) == pytest.approx(
        (0.0421674772, 0.0262428745, 0.1818476718), abs=1e-9
    )
    assert pose.pitch_rad == pytest.approx(0.0606031186, abs=1e-9)
    assert model.forward(model.anchor_joints) == pose

    task_x = model.task_x
    task_y = model.task_y
    task_z = model.task_z
    assert task_z == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert sum(a * b for a, b in zip(task_x, task_z)) == pytest.approx(0.0)
    assert sum(a * b for a, b in zip(task_y, task_z)) == pytest.approx(0.0)
    assert sum(value * value for value in task_x) == pytest.approx(1.0)
    assert sum(value * value for value in task_y) == pytest.approx(1.0)
    assert model.forward_component(model.anchor_joints) == pytest.approx(
        math.cos(pose.pitch_rad), abs=1e-12
    )


def test_virtual_delta_is_expressed_in_fixed_idle_task_frame(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    config, model, _ = setup
    initial = config.workspace.initial_target
    target = VirtualEETarget(
        x_m=initial.x_m + 0.02,
        y_m=initial.y_m - 0.03,
        z_m=initial.z_m + 0.04,
        pitch_rad=initial.pitch_rad + 0.25,
    )

    desired = model.target_from_virtual_delta(target, initial)
    expected_position = tuple(
        anchor
        + 0.02 * task_x
        - 0.03 * task_y
        + 0.04 * task_z
        for anchor, task_x, task_y, task_z in zip(
            (model.anchor_pose.x_m, model.anchor_pose.y_m, model.anchor_pose.z_m),
            model.task_x,
            model.task_y,
            model.task_z,
        )
    )
    assert (desired.x_m, desired.y_m, desired.z_m) == pytest.approx(
        expected_position
    )
    assert desired.pitch_rad == pytest.approx(model.anchor_pose.pitch_rad + 0.25)

    zero_delta = model.target_from_virtual_delta(initial, initial)
    assert zero_delta == model.anchor_pose


def test_forward_rejects_missing_nonfinite_and_out_of_range_joints(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, _ = setup
    missing = dict(model.anchor_joints)
    missing.pop("wrist")
    with pytest.raises(ValueError, match="exactly"):
        model.forward(missing)

    nonfinite = dict(model.anchor_joints, elbow=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model.forward(nonfinite)

    out_of_range = dict(
        model.anchor_joints,
        shoulder=model.limits["shoulder"].upper_rad + 0.01,
    )
    with pytest.raises(ValueError, match="outside"):
        model.forward(out_of_range)


@pytest.mark.parametrize(
    "expected_joints",
    [
        {"base": 0.12, "shoulder": -0.38, "elbow": -0.62, "wrist": -0.90},
        {"base": -0.15, "shoulder": -0.50, "elbow": -0.80, "wrist": -0.75},
        {"base": 0.30, "shoulder": -0.70, "elbow": -1.00, "wrist": -1.10},
    ],
)
def test_fk_ik_round_trip_for_fixed_nonsingular_poses(
    setup: tuple[object, KinematicModel, IKSolver],
    expected_joints: dict[str, float],
) -> None:
    config, model, solver = setup
    target = model.forward(expected_joints)

    result = solver.solve(
        target,
        seeds=(model.anchor_joints,),
        reference_joints=model.anchor_joints,
    )

    assert result.status is IKStatus.SOLVED
    assert result.joints_rad is not None
    assert result.position_error_m <= config.ik.position_tolerance_m
    assert result.pitch_error_rad <= config.ik.pitch_tolerance_rad
    assert model.forward_component(result.joints_rad) >= config.ik.min_forward_component
    assert set(result.joints_rad) == set(model.joint_order)
    for name, angle in result.joints_rad.items():
        assert model.limits[name].contains(angle)


def test_solver_is_repeatable_for_the_same_seed_set(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    joints = {"base": 0.18, "shoulder": -0.42, "elbow": -0.7, "wrist": -0.82}
    target = model.forward(joints)
    seeds = (model.anchor_joints, model.midpoint_joints())

    first = solver.solve(target, seeds, model.anchor_joints)
    second = solver.solve(target, seeds, model.anchor_joints)

    assert first.status is IKStatus.SOLVED
    assert second.status is IKStatus.SOLVED
    assert second.joints_rad == pytest.approx(first.joints_rad, abs=1e-14)
    assert second.iterations == first.iterations


def test_raw_solver_reports_large_but_converged_branch_for_controller_policy(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    distant = dict(model.anchor_joints, base=0.9)
    target = model.forward(distant)

    result = solver.solve(target, (distant,), model.anchor_joints)

    assert result.status is IKStatus.SOLVED
    assert result.joints_rad is not None
    assert abs(result.joints_rad["base"] - model.anchor_joints["base"]) > 0.75


def test_unreachable_target_has_no_joint_candidate(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    target = EndEffectorPose(4.0, 4.0, 4.0, 0.0)

    result = solver.solve(target, (model.anchor_joints,), model.anchor_joints)

    assert result.status is IKStatus.UNREACHABLE
    assert result.joints_rad is None
    assert result.position_error_m > 1.0


def test_solver_nonconvergence_is_not_mislabeled_unreachable(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    known_reachable = {
        "base": 0.1807383849,
        "shoulder": -1.3945738167,
        "elbow": -2.6884911858,
        "wrist": -2.4691619960,
    }
    target = model.forward(known_reachable)

    result = solver.solve(target, (model.anchor_joints,), model.anchor_joints)

    assert result.status in {IKStatus.SOLVED, IKStatus.NUMERICAL_FAILURE}


def test_alternate_backward_branch_does_not_make_target_unreachable(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    known_forward_solution = {
        "base": -0.7085396811142849,
        "shoulder": -0.15013578982740408,
        "elbow": -1.3366684890253024,
        "wrist": -2.3754029037123336,
    }
    assert (
        model.forward_component(known_forward_solution)
        >= model.ik_config.min_forward_component
    )
    target = model.forward(known_forward_solution)

    result = solver.solve(target, (model.anchor_joints,), model.anchor_joints)

    assert result.status in {IKStatus.SOLVED, IKStatus.NUMERICAL_FAILURE}


def test_unreachable_pitch_precheck_respects_convergence_tolerance(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    joints = {
        "base": 0.1588227785,
        "shoulder": -0.3293759496,
        "elbow": -0.2797757814,
        "wrist": -2.1640976253,
    }
    achieved = model.forward(joints)
    assert model.forward_component(joints) >= model.ik_config.min_forward_component
    target = EndEffectorPose(
        achieved.x_m,
        achieved.y_m,
        achieved.z_m,
        -1.3745388766,
    )
    assert abs(target.pitch_rad - achieved.pitch_rad) < model.ik_config.pitch_tolerance_rad

    result = solver.solve(target, (joints,), joints)

    assert result.status is IKStatus.SOLVED


def test_backward_or_near_singular_exact_pose_is_rejected(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    config, model, solver = setup
    backward = {
        name: model.limits[name].lower_rad for name in model.joint_order
    }
    assert model.forward_component(backward) < config.ik.min_forward_component
    target = model.forward(backward)

    result = solver.solve(target, (backward,), backward)

    assert result.status is IKStatus.UNREACHABLE
    assert result.joints_rad is None
    assert result.position_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.pitch_error_rad == pytest.approx(0.0, abs=1e-12)


def test_solution_at_joint_boundary_reports_active_limit(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    boundary = dict(model.anchor_joints)
    boundary["shoulder"] = model.limits["shoulder"].lower_rad
    target = model.forward(boundary)

    result = solver.solve(target, (boundary,), boundary)

    assert result.status is IKStatus.SOLVED
    assert result.joints_rad is not None
    assert "shoulder" in result.active_limits


def test_nonfinite_target_is_a_numerical_failure(
    setup: tuple[object, KinematicModel, IKSolver],
) -> None:
    _, model, solver = setup
    target = EndEffectorPose(float("nan"), 0.0, 0.1, 0.0)

    result = solver.solve(target, (model.anchor_joints,), model.anchor_joints)

    assert result.status is IKStatus.NUMERICAL_FAILURE
    assert result.joints_rad is None
    assert math.isinf(result.position_error_m)
    assert math.isfinite(result.pitch_error_rad)


def test_ik_result_enforces_joint_candidate_status_invariant() -> None:
    pose = EndEffectorPose(0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="exactly when IK is SOLVED"):
        IKResult(IKStatus.SOLVED, pose, pose, None, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="exactly when IK is SOLVED"):
        IKResult(
            IKStatus.UNREACHABLE,
            pose,
            pose,
            {"base": 0.0},
            0.0,
            0.0,
            0,
        )
