#!/usr/bin/env python3
"""Scripted smooth-motion demo using smoothstep interpolation.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/scripted_motion.py

Sequence: home -> rotate base -> reach with shoulder+elbow -> tilt wrist
-> return home. All targets are radians within the temporary simulation
limits.
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
        home = arm.config.home_pose
        print("Scripted motion demo (all values in radians)")

        print("1/5  Moving to sim_home...")
        arm.go_home()

        print("2/5  Rotating base...")
        arm.move_joint("base", 1.0, duration_s=2.0)

        print("3/5  Reaching with shoulder and elbow...")
        arm.move_joints(
            {"shoulder": 0.7, "elbow": -1.1},
            duration_s=2.5,
        )

        print("4/5  Tilting wrist...")
        arm.move_joint("wrist", 0.6, duration_s=1.5)

        print("5/5  Returning home...")
        arm.move_joints(home, duration_s=3.0)
        arm.run_for(1.0)

        print("Final joint positions (rad):")
        for joint, position in arm.get_joint_positions().items():
            print(f"  {joint:<8} {position:+.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
