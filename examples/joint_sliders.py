#!/usr/bin/env python3
"""Interactive four-joint control with PyBullet GUI sliders.

Run on the simulation machine (Linux/Mac), inside the project's .venv,
from a local terminal on the machine's own display (not an SSH session):

    python examples/joint_sliders.py

One slider per joint (base, shoulder, elbow, wrist), using the configured
simulation limits from config/joints.yaml. Close the PyBullet window to
exit.
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
        print("Butter Finger joint sliders")
        print(f"  Time step: {arm.time_step_s:.6f} s ({arm.config.control_rate_hz:g} Hz)")
        print("  Sliders (radians, temporary simulation limits):")
        sliders: dict[str, int] = {}
        for joint in arm.joint_names:
            limits = arm.config.sim_limits[joint]
            start = arm.config.home_pose[joint]
            sliders[joint] = arm.pb.addUserDebugParameter(
                f"{joint} (rad)",
                limits.lower_rad,
                limits.upper_rad,
                start,
                physicsClientId=arm.client_id,
            )
            print(
                f"    {joint:<8} [{limits.lower_rad:+.4f}, {limits.upper_rad:+.4f}]"
                f"  start={start:+.4f}"
            )
        print("  Close the PyBullet window to exit.")

        try:
            while arm.is_connected():
                targets: dict[str, float] = {}
                for joint, slider in sliders.items():
                    value = arm.pb.readUserDebugParameter(
                        slider, physicsClientId=arm.client_id
                    )
                    # Debug sliders return float32-like values, so an endpoint
                    # can differ from its configured limit by a few nanoradians.
                    limits = arm.config.sim_limits[joint]
                    targets[joint] = limits.clamp(value)
                arm.move_joints(targets)
                arm.step(realtime=True)
        except arm.pb.error:
            pass  # GUI window was closed mid-step
    print("Sliders closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
