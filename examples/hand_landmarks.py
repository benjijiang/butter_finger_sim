#!/usr/bin/env python3
"""Stage 0: display MediaPipe's 21 landmarks for up to two webcam hands.

CAMERA ONLY. This program never constructs an arm backend and cannot move the
simulated or physical Butter Finger arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.teleoperation.viewer import run_viewer


def main(argv: list[str] | None = None, **kwargs) -> int:
    return run_viewer(argv, stage=0, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
