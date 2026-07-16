#!/usr/bin/env python3
"""Play selected emotional actions in one PyBullet GUI session.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/emotion_showcase.py --all
    python examples/emotion_showcase.py happy curious sad surprised

All motion uses temporary simulation radians and is not approved for PWM.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import (
    ActionConfig,
    ActionRunner,
    BackendUnavailableError,
    IdleController,
    PyBulletArm,
    load_action_config,
)

EMOTION_ACTION_NAMES = (
    "greet",
    "nod_yes",
    "shake_no",
    "attentive",
    "happy",
    "excited",
    "proud",
    "playful",
    "affectionate",
    "shy",
    "curious",
    "thinking",
    "confused",
    "sad",
    "disappointed",
    "bored",
    "sleepy",
    "surprised",
    "scared",
    "angry",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "actions",
        nargs="*",
        help="emotional action names to play in the supplied order",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="play_all",
        help="play all configured emotional actions",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=1.5,
        help="seconds of slow idle scanning between actions (default: 1.5)",
    )
    return parser


def select_action_names(
    requested: list[str],
    *,
    play_all: bool,
    config: ActionConfig,
) -> tuple[str, ...]:
    """Validate the requested showcase playlist without opening PyBullet."""
    if play_all and requested:
        raise ValueError("choose either --all or explicit action names, not both")
    if not play_all and not requested:
        raise ValueError("provide action names or use --all")

    names = EMOTION_ACTION_NAMES if play_all else tuple(requested)
    unknown = [name for name in names if name not in config.actions]
    if unknown:
        raise ValueError(
            f"unknown action(s) {unknown}; choose from {list(EMOTION_ACTION_NAMES)}"
        )
    return tuple(names)


def _run_idle_gap(
    arm: PyBulletArm,
    idle: IdleController,
    duration_s: float,
) -> None:
    """Resume idle and advance it for a finite showcase gap."""
    idle.resume()
    steps = round(duration_s / arm.time_step_s)
    for _ in range(steps):
        idle.update(arm.time_step_s)
        arm.step(realtime=True)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if (
        not math.isfinite(args.idle_seconds)
        or args.idle_seconds < 0
    ):
        parser.error("--idle-seconds must be a finite number at least zero")

    config = load_action_config()
    try:
        names = select_action_names(
            args.actions,
            play_all=args.play_all,
            config=config,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        arm = PyBulletArm(gui=True)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    completed = True
    with arm:
        runner = ActionRunner(arm, config)
        idle = IdleController(arm)
        try:
            for index, name in enumerate(names, start=1):
                action = config.actions[name]
                print(f"[{index}/{len(names)}] {name}: {action.description}")
                runner.run(name)
                if args.idle_seconds > 0:
                    _run_idle_gap(arm, idle, args.idle_seconds)
        except (KeyboardInterrupt, arm.pb.error):
            completed = False

    print("Showcase complete." if completed else "Showcase stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
