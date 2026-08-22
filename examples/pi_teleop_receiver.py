#!/usr/bin/env python3
"""Stage 3 Raspberry Pi receiver for Mac-computed hand teleoperation.

Run on the Raspberry Pi only. It has no camera, MediaPipe, or model dependency.
The hardware path requires an explicit acknowledgement:

    python examples/pi_teleop_receiver.py --confirm-hardware

A network-only commissioning path is also available:

    python examples/pi_teleop_receiver.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.arm import ArmBackend, JointStateUnavailableError
from butter_finger.config import JOINT_NAMES, load_arm_config
from butter_finger.remote_control.config import (
    REMOTE_TELEOPERATION_CONFIG_PATH,
    RemoteTeleoperationConfig,
    compute_profile_sha256,
    load_remote_teleoperation_config,
)
from butter_finger.remote_control.receiver import PiTeleopReceiver


class LoggingArm(ArmBackend):
    """Pi receiver dry-run arm; validates radians and never opens hardware."""

    def __init__(self, limits, home_pose: dict[str, float]) -> None:
        self._limits = limits
        self._home = dict(home_pose)
        self._positions: dict[str, float] = {}

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        for joint, raw in targets_rad.items():
            if joint not in self._limits:
                raise ValueError(f"unknown joint {joint!r}")
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or not self._limits[joint].contains(float(raw))
            ):
                raise ValueError(f"invalid target for {joint}")

    def move_joint(self, joint, position_rad, *, duration_s=None) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(self, targets_rad, *, duration_s=None) -> None:
        self.validate_targets(targets_rad)
        self._positions.update({joint: float(v) for joint, v in targets_rad.items()})
        print(f"[dry-run arm] {self._positions}")

    def get_joint_positions(self) -> dict[str, float]:
        if set(self._positions) != set(JOINT_NAMES):
            raise JointStateUnavailableError("dry-run arm has no complete state")
        return dict(self._positions)

    def go_home(self) -> None:
        self._positions = dict(self._home)
        print("[dry-run arm] home")

    def disconnect(self) -> None:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="receive, validate, rate-limit, and log without opening UART",
    )
    mode.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required acknowledgement before opening the real arm",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REMOTE_TELEOPERATION_CONFIG_PATH,
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="append receiver events and applied command estimates",
    )
    return parser


def _real_arm(config: RemoteTeleoperationConfig) -> ArmBackend:
    # Hardware imports remain behind the explicit --confirm-hardware gate.
    from butter_finger.backends.pwm_robot_arm import PWMRobotArm
    from butter_finger.backends.raspberry_pi_arm import RaspberryPiArm

    return RaspberryPiArm(
        pwm_arm=PWMRobotArm(verbose=False),
        stream_duration_s=config.stream_duration_s,
    )


def main(
    argv: list[str] | None = None,
    *,
    arm_factory: Callable[[RemoteTeleoperationConfig], ArmBackend] | None = None,
    receiver_factory: Callable[..., Any] = PiTeleopReceiver,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.confirm_hardware:
        parser.error("choose --dry-run or explicitly pass --confirm-hardware")

    try:
        remote_config = load_remote_teleoperation_config(args.config)
        arm_config = load_arm_config()
        startup_joints = dict(arm_config.poses[remote_config.startup_pose])
        fingerprint = compute_profile_sha256(remote_config, arm_config)
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR: invalid Stage 3 configuration: {exc}", file=sys.stderr)
        return 1

    log_handle = None
    if args.log_jsonl is not None:
        try:
            log_handle = args.log_jsonl.open("a", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not open receiver log: {exc}", file=sys.stderr)
            return 1

    def log_record(record: dict[str, Any]) -> None:
        if log_handle is None:
            return
        log_handle.write(
            json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False)
            + "\n"
        )
        log_handle.flush()

    factory = arm_factory
    if factory is None:
        if args.dry_run:
            factory = lambda _cfg: LoggingArm(arm_config.sim_limits, arm_config.home_pose)
        else:
            factory = _real_arm

    arm: ArmBackend | None = None
    try:
        arm = factory(remote_config)
        mode = "dry_run" if args.dry_run else "hardware"
        print(f"Butter Finger Stage 3 Pi receiver ({mode})")
        print("  Initializing exact home, then idle_ready; no client is accepted yet.")
        arm.go_home()
        arm.move_joints(
            startup_joints,
            duration_s=remote_config.startup_duration_s,
        )
        receiver = receiver_factory(
            arm,
            remote_config,
            fingerprint,
            startup_joints,
            mode=mode,
            logger=log_record,
        )

        def ready(host: str, port: int) -> None:
            print(f"  READY: listening on {host}:{port} through the SSH tunnel")
            print("  One session only; Ctrl-C holds the last target and exits.")

        result = receiver.serve_once(ready_callback=ready)
        print(f"Receiver stopped: {result.reason}; arm remains at its last target.")
        return 1 if result.fatal else 0
    except KeyboardInterrupt:
        print("\nStopping receiver; holding the last target (no automatic home).")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: Stage 3 receiver failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"WARNING: could not disconnect arm: {exc}", file=sys.stderr)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
