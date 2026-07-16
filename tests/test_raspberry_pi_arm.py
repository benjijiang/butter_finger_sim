"""Calibrated Raspberry Pi backend tests with no SDK or serial hardware."""
from __future__ import annotations

from dataclasses import replace

import pytest

from butter_finger import (
    JOINT_NAMES,
    JointLimitError,
    JointStateUnavailableError,
    RaspberryPiArm,
    UnknownJointError,
)
from butter_finger.backends.pwm_robot_arm import PWMRobotArm


class FakePort:
    def __init__(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False


class FakeBoard:
    def __init__(self) -> None:
        self.calls: list[tuple[float, list[list[int]]]] = []
        self.reception: list[bool] = []
        self.port = FakePort()

    def pwm_servo_set_position(
        self, duration: float, positions: list[list[int]]
    ) -> None:
        self.calls.append((duration, positions))

    def enable_reception(self, enabled: bool) -> None:
        self.reception.append(enabled)


@pytest.fixture()
def hardware() -> tuple[RaspberryPiArm, FakeBoard, list[float]]:
    board = FakeBoard()
    pwm_arm = PWMRobotArm(board=board)
    waits: list[float] = []
    pwm_arm.wait = waits.append  # type: ignore[method-assign]
    return RaspberryPiArm(pwm_arm=pwm_arm), board, waits


@pytest.mark.parametrize(
    ("joint", "position_rad", "expected_pwm"),
    [
        ("base", -1.5708, 510.0),
        ("base", 0.0, 1500.0),
        ("base", 1.5708, 2490.0),
        ("shoulder", -0.765, 1700.0),
        ("elbow", -1.505, 1500.0),
        ("wrist", -1.4124, 1500.0),
    ],
)
def test_rad_to_pwm_endpoints_and_midpoints(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
    joint: str,
    position_rad: float,
    expected_pwm: float,
) -> None:
    arm, board, _ = hardware

    assert arm.rad_to_pwm(joint, position_rad) == pytest.approx(expected_pwm)
    assert arm.pwm_to_rad(joint, expected_pwm) == pytest.approx(position_rad)
    assert board.calls == []


def test_timed_move_sends_one_atomic_command_and_waits(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, board, waits = hardware

    arm.move_joints({"base": 0.0, "shoulder": -0.765}, duration_s=0.5)

    assert board.calls == [(0.5, [[1, 1500], [3, 1700]])]
    assert waits == [0.5]


def test_untimed_move_uses_pwm_default_without_waiting(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, board, waits = hardware

    arm.move_joint("wrist", -1.4124)

    assert board.calls == [(1.0, [[5, 1500]])]
    assert waits == []


@pytest.mark.parametrize(
    ("targets", "error"),
    [
        ({"gripper": 0.0}, UnknownJointError),
        ({"base": float("nan")}, ValueError),
        ({"base": True}, ValueError),
        ({"base": 1.5709}, JointLimitError),
        ({"base": 0.0, "wrist": 0.2}, JointLimitError),
    ],
)
def test_invalid_targets_are_rejected_before_hardware_call(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
    targets: dict[str, float],
    error: type[Exception],
) -> None:
    arm, board, waits = hardware

    with pytest.raises(error):
        arm.move_joints(targets, duration_s=1.0)
    assert board.calls == []
    assert waits == []


@pytest.mark.parametrize(
    "duration",
    [True, "1", 0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_duration_is_rejected_before_hardware_call(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
    duration,
) -> None:
    arm, board, _ = hardware

    with pytest.raises(ValueError, match="duration_s"):
        arm.move_joint("base", 0.0, duration_s=duration)
    assert board.calls == []


def test_get_positions_requires_a_complete_last_command(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, _, _ = hardware

    with pytest.raises(JointStateUnavailableError, match="no joint-angle feedback"):
        arm.get_joint_positions()
    arm.move_joint("base", 0.25)
    with pytest.raises(JointStateUnavailableError, match="missing"):
        arm.get_joint_positions()


def test_complete_target_establishes_last_command_estimate(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, _, _ = hardware
    targets = {
        "base": 0.25,
        "shoulder": -0.5,
        "elbow": -1.0,
        "wrist": -1.25,
    }

    arm.move_joints(targets)

    assert arm.get_joint_positions() == targets


def test_home_uses_exact_physical_pwm_waits_and_establishes_state(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, board, waits = hardware

    arm.go_home()

    duration, positions = board.calls[0]
    assert duration == 3.0
    assert {port: pulse for port, pulse in positions} == {
        1: 1500,
        3: 2200,
        4: 2490,
        5: 1400,
    }
    assert waits == [3.0]
    state = arm.get_joint_positions()
    assert tuple(state) == JOINT_NAMES
    for joint, pulse in arm.config.home_pwm_us.items():
        assert state[joint] == pytest.approx(arm.pwm_to_rad(joint, pulse))


def test_disconnect_closes_injected_board_port(
    hardware: tuple[RaspberryPiArm, FakeBoard, list[float]],
) -> None:
    arm, board, _ = hardware

    arm.disconnect()
    arm.disconnect()

    assert board.reception == [False, False]
    assert board.port.is_open is False


def test_rejects_mismatched_injected_pwm_configuration() -> None:
    pwm_arm = PWMRobotArm(board=FakeBoard())
    mismatched = replace(
        pwm_arm.config,
        home_pwm_us={**pwm_arm.config.home_pwm_us, "base": 1499},
    )

    with pytest.raises(ValueError, match="same PhysicalConfig"):
        RaspberryPiArm(pwm_arm=pwm_arm, config=mismatched)
