#!/usr/bin/env python3
"""REAL HARDWARE: sweep the base joint around its home pulse.

Run ONLY on the Raspberry Pi 5 connected to the RasAdapter5A. This is the
ported board_demo/test_robot_arm.py flow (center, left, right, center),
using the tested motion timing (duration 1.0 s, 1.5 s settle).

    python examples/pi_sweep_base.py
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

    # PWM pulse widths in microseconds; approximate angles from the original
    # test: 1500 ~ 90 deg (center), 1280 ~ 70 deg, 1720 ~ 110 deg.
    for pulse in (1500, 1280, 1720, 1500):
        arm.move_joint("base", pulse, duration=1.0)
        arm.wait(1.5)

    print("Test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
