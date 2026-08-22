#!/usr/bin/env python3
"""Stage 3 Mac client: hand tracking and IK command the arm through the Pi.

Run on the Mac after the Pi receiver and manual SSH tunnel are ready:

    python examples/hand_teleoperation_stage3.py --confirm-remote-hardware

This process owns the webcam, MediaPipe model, retargeting, IK, and the
Mac-side slew limiter. It never imports the Raspberry Pi or PWM backends.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.teleoperation.viewer import run_viewer


def main(argv: list[str] | None = None, **kwargs) -> int:
    return run_viewer(argv, stage=3, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
