#!/usr/bin/env python3
"""Run the simulation-only no-person idle scan in the PyBullet GUI.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/idle_motion.py

The targets are calibrated radians, but this continuously streamed controller
is intentionally simulation-only. Close the GUI or press Ctrl+C to stop.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, IdleController, PyBulletArm


def main() -> int:
    try:
        arm = PyBulletArm(gui=True)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with arm:
        idle = IdleController(arm)
        idle.resume()
        print("Butter Finger idle scan (simulation-only controller)")
        print(
            f"  {idle.config.scan_joint}: "
            f"{idle.config.lower_rad:+.2f} to {idle.config.upper_rad:+.2f} rad, "
            f"{idle.config.half_cycle_s:.1f} s per half-cycle"
        )
        print("  Close the PyBullet window or press Ctrl+C to exit.")
        try:
            while arm.is_connected():
                idle.update(arm.time_step_s)
                arm.step(realtime=True)
        except (KeyboardInterrupt, arm.pb.error):
            pass

    print("Idle scan closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
