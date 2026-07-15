#!/usr/bin/env python3
"""List or run a configured named action in the PyBullet GUI.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/run_action.py --list
    python examples/run_action.py base_scan

The bundled actions use temporary simulation radians and are not approved for
the physical PWM-controlled arm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import (
    ActionRunner,
    BackendUnavailableError,
    PyBulletArm,
    load_action_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", help="name of the action to run")
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_actions",
        help="list configured actions without opening PyBullet",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_action_config()

    if args.list_actions:
        print("Configured simulation actions (radians; not approved for PWM):")
        for name, action in config.actions.items():
            print(f"  {name:<18} {action.description}")
        return 0

    if args.action is None:
        parser.error("provide an action name or use --list")
    if args.action not in config.actions:
        parser.error(
            f"unknown action {args.action!r}; choose from "
            f"{', '.join(config.action_names)}"
        )

    try:
        arm = PyBulletArm(gui=True)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with arm:
        action = config.actions[args.action]
        print(f"Running {action.name}: {action.description}")
        ActionRunner(arm, config).run(action.name)
        arm.run_for(1.0)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
