#!/usr/bin/env python3
"""REAL HARDWARE: step every joint to the recorded home pose, one at a time.

Run ONLY on the Raspberry Pi 5 connected to the RasAdapter5A. This is the
ported board_demo/test_pose.py flow using the tested motion timing
(duration 2.0 s per joint, 2.5 s settle; coordinated move 3.0 s + 3.5 s).

    python examples/pi_test_pose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, PWMRobotArm


def main() -> int:
    try:
        arm = PWMRobotArm()
    except BackendUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    home = arm.config.home_pwm_us

    # Move separately first to avoid unexpected combined motion.
    for joint, pulse in home.items():
        input(f"Press Enter to move {joint} to pulse {pulse}...")
        arm.move_joint(joint, pulse, duration=2.0)
        arm.wait(2.5)

    print("Individual movements complete.")

    input("Press Enter to test coordinated movement...")
    arm.move_joints(home, duration=3.0)
    arm.wait(3.5)

    print("Home-pose test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
