#!/usr/bin/env python3
"""Open the PyBullet GUI and display the Butter Finger arm at sim_home.

Run on the simulation machine (Linux/Mac), inside the project's .venv,
from a local terminal on the machine's own display (not an SSH session):

    python examples/view_robot.py

Close the PyBullet window to exit.
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
        arm.reset_joints(arm.config.home_pose)
        print("Butter Finger viewer")
        print(f"  URDF: {REPO_ROOT / 'models' / 'butter_finger_simple.urdf'}")
        print(f"  Joints: {', '.join(arm.joint_names)}")
        print("  Close the PyBullet window to exit.")
        try:
            while arm.is_connected():
                arm.step(realtime=True)
        except arm.pb.error:
            pass  # GUI window was closed mid-step
    print("Viewer closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
