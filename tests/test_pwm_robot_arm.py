"""PWM hardware-layer tests using a fake Board.

No serial port, no SDK, and no PyBullet required: a fake board object is
injected, so these tests run on any machine.
"""
from __future__ import annotations

import pytest

from butter_finger.arm import JointLimitError, UnknownJointError
from butter_finger.backends.pwm_robot_arm import PWMRobotArm
from butter_finger.config import JOINT_NAMES, load_physical_config


class FakeBoard:
    def __init__(self) -> None:
        self.calls: list[tuple[float, list[list[int]]]] = []

    def pwm_servo_set_position(self, duration, positions) -> None:
        self.calls.append((duration, positions))


@pytest.fixture()
def board() -> FakeBoard:
    return FakeBoard()


@pytest.fixture()
def arm(board: FakeBoard) -> PWMRobotArm:
    return PWMRobotArm(board=board)


def test_physical_config_loads() -> None:
    config = load_physical_config()
    assert set(config.joint_order) == set(JOINT_NAMES)
    assert config.pwm_ports == {"base": 1, "shoulder": 3, "elbow": 4, "wrist": 5}
    for joint in config.joint_order:
        assert config.pulse_limits_us[joint].contains(config.home_pwm_us[joint])


def test_move_joint_sends_port_and_pulse(arm: PWMRobotArm, board: FakeBoard) -> None:
    arm.move_joint("base", 1500, duration=1.0)
    assert board.calls == [(1.0, [[1, 1500]])]


def test_move_joint_rounds_pulse(arm: PWMRobotArm, board: FakeBoard) -> None:
    arm.move_joint("wrist", 1400.4, duration=1.0)
    assert board.calls == [(1.0, [[5, 1400]])]


def test_move_joint_rejects_out_of_range(arm: PWMRobotArm, board: FakeBoard) -> None:
    with pytest.raises(JointLimitError):
        arm.move_joint("shoulder", 2300)
    assert board.calls == []


def test_move_joint_rejects_unknown_joint(arm: PWMRobotArm, board: FakeBoard) -> None:
    with pytest.raises(UnknownJointError):
        arm.move_joint("gripper", 1500)
    assert board.calls == []


def test_move_joint_rejects_nonpositive_duration(arm: PWMRobotArm, board: FakeBoard) -> None:
    with pytest.raises(ValueError):
        arm.move_joint("base", 1500, duration=0)
    assert board.calls == []


def test_move_joints_sends_one_command(arm: PWMRobotArm, board: FakeBoard) -> None:
    arm.move_joints({"base": 1500, "elbow": 2490}, duration=2.0)
    assert board.calls == [(2.0, [[1, 1500], [4, 2490]])]


def test_move_joints_validates_all_before_sending(arm: PWMRobotArm, board: FakeBoard) -> None:
    with pytest.raises(JointLimitError):
        arm.move_joints({"base": 1500, "shoulder": 5000})
    assert board.calls == []


def test_home_uses_recorded_pose(arm: PWMRobotArm, board: FakeBoard) -> None:
    arm.home()
    duration, positions = board.calls[0]
    assert duration == 3.0
    expected = {
        arm.config.pwm_ports[joint]: pulse
        for joint, pulse in arm.config.home_pwm_us.items()
    }
    assert {port: pulse for port, pulse in positions} == expected


def test_limits_match_tested_values() -> None:
    config = load_physical_config()
    assert (config.pulse_limits_us["base"].min_us, config.pulse_limits_us["base"].max_us) == (505, 2495)
    assert (config.pulse_limits_us["shoulder"].min_us, config.pulse_limits_us["shoulder"].max_us) == (1200, 2220)
    assert (config.pulse_limits_us["elbow"].min_us, config.pulse_limits_us["elbow"].max_us) == (505, 2495)
    assert (config.pulse_limits_us["wrist"].min_us, config.pulse_limits_us["wrist"].max_us) == (505, 2495)
