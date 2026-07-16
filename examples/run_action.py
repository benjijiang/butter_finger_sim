#!/usr/bin/env python3
"""List or run a configured named action in simulation or on the real arm.

Examples:

    python examples/run_action.py --list
    python examples/run_action.py base_scan
    python examples/run_action.py happy --backend real --confirm-hardware
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import (
    ActionRunner,
    ArmBackend,
    BackendUnavailableError,
    PyBulletArm,
    RaspberryPiArm,
    load_action_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", help="name of the action to run")
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_actions",
        help="list configured actions without opening a backend",
    )
    parser.add_argument(
        "--backend",
        choices=("sim", "real"),
        default="sim",
        help="arm backend to use (default: sim)",
    )
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required acknowledgement before opening the real arm",
    )
    return parser


def _create_arm(backend: str) -> ArmBackend:
    if backend == "sim":
        return PyBulletArm(gui=True)
    return RaspberryPiArm()


def main(
    argv: Sequence[str] | None = None,
    *,
    arm_factory: Callable[[str], ArmBackend] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_action_config()

    if args.list_actions:
        print("Configured arm actions (calibrated radians):")
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
    if args.backend == "real" and not args.confirm_hardware:
        parser.error("real backend requires --confirm-hardware")

    factory = arm_factory if arm_factory is not None else _create_arm
    try:
        arm = factory(args.backend)
    except (BackendUnavailableError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with arm:
        if args.backend == "real":
            print("Moving real arm to its exact physical home PWM...")
            arm.go_home()
        action = config.actions[args.action]
        print(f"Running {action.name} on {args.backend}: {action.description}")
        ActionRunner(arm, config).run(action.name)
        if args.backend == "sim":
            arm.run_for(1.0)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
