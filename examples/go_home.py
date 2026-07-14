#!/usr/bin/env python3
"""Move the simulated arm to the configured sim_home pose and hold it.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/go_home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, PyBulletArm


def main() -> int:
    try:
        arm = PyBulletArm(gui=True)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with arm:
        # Start from an offset pose so the homing motion is visible.
        arm.reset_joints({"base": 0.9, "shoulder": 0.7, "elbow": -0.8, "wrist": 0.5})
        print("Moving to sim_home (all joints 0.0 rad)...")
        arm.go_home()
        positions = arm.get_joint_positions()
        print("Joint positions after homing (rad):")
        for joint, position in positions.items():
            print(f"  {joint:<8} {position:+.4f}")
        print("Holding for 3 seconds...")
        arm.run_for(3.0)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
