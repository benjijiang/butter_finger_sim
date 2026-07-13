#!/usr/bin/env python3
"""Scripted smooth-motion demo using smoothstep interpolation.

Run on the Mac, inside the project's .venv:

    python examples/scripted_motion.py

Sequence: home -> rotate base -> reach with shoulder+elbow -> tilt wrist
-> return home. All targets are radians within the temporary simulation
limits.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, PyBulletArm


def interpolate_pose(
    start: Mapping[str, float],
    target: Mapping[str, float],
    duration: float,
    control_rate: float,
) -> Iterator[dict[str, float]]:
    """Yield smoothstep-interpolated poses from start to target."""
    steps = max(1, round(duration * control_rate))
    for step in range(1, steps + 1):
        alpha = step / steps
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3
        yield {
            joint: start[joint]
            + smooth_alpha * (target[joint] - start[joint])
            for joint in target
        }


def move_smoothly(arm: PyBulletArm, target: Mapping[str, float], duration: float) -> None:
    start = arm.get_joint_positions()
    for pose in interpolate_pose(start, target, duration, arm.config.control_rate_hz):
        arm.move_joints(pose)
        arm.step(realtime=True)


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
        move_smoothly(arm, {"base": 1.0}, duration=2.0)

        print("3/5  Reaching with shoulder and elbow...")
        move_smoothly(arm, {"shoulder": 0.7, "elbow": -1.1}, duration=2.5)

        print("4/5  Tilting wrist...")
        move_smoothly(arm, {"wrist": 0.6}, duration=1.5)

        print("5/5  Returning home...")
        move_smoothly(arm, home, duration=3.0)
        arm.run_for(1.0)

        print("Final joint positions (rad):")
        for joint, position in arm.get_joint_positions().items():
            print(f"  {joint:<8} {position:+.4f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
