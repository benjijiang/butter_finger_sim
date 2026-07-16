#!/usr/bin/env python3
"""Capture one simulated wrist-camera RGB frame as a PNG image.

Run on the simulation machine (Linux/Mac), inside the project's .venv:

    python examples/camera_snapshot.py
    python examples/camera_snapshot.py --output camera.png --gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger import BackendUnavailableError, PyBulletArm


def save_png(path: Path, rgb: np.ndarray) -> None:
    """Write one uint8 HWC RGB array as a PNG image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 HWC array with three channels")
    Image.fromarray(rgb).save(path, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera_snapshot.png"),
        help="output PNG path (default: camera_snapshot.png)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="also open the PyBullet GUI while capturing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        arm = PyBulletArm(gui=args.gui)
        with arm:
            arm.reset_joints(arm.config.home_pose)
            arm.step()
            rgb = arm.capture_rgb()
        save_png(args.output, rgb)
    except (BackendUnavailableError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    height, width, channels = rgb.shape
    print(
        f"Wrote {args.output} ({width}x{height}, {channels}-channel "
        f"{rgb.dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
