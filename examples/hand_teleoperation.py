#!/usr/bin/env python3
"""Stage 2: preview hand-retargeted IK without commanding any robot.

DRY RUN ONLY. This program computes configuration-domain joint candidates for
diagnostics. It never constructs an arm backend or emits hardware commands.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.teleoperation.viewer import run_viewer


def main(argv: list[str] | None = None, **kwargs) -> int:
    return run_viewer(argv, stage=2, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
