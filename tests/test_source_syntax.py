"""Compile every Python source file without executing imports, and enforce
that no real-hardware library is imported anywhere in the project.
"""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "examples", "tests")

# Hardware libraries that must never be imported directly in this repository.
# Real-hardware access goes only through the external Hiwonder SDK
# (ros_robot_controller_sdk, which lives outside the repo in board_demo/),
# loaded lazily by PWMRobotArm at instantiation.
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(RPi|serial|gpiozero|smbus2?|spidev|lgpio|pigpio)\b",
    re.MULTILINE,
)


def python_files() -> list[Path]:
    files = []
    for directory in SOURCE_DIRS:
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    assert files, "no Python sources found"
    return files


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_hardware_imports(path: Path) -> None:
    match = FORBIDDEN_IMPORT.search(path.read_text(encoding="utf-8"))
    assert match is None, (
        f"{path} imports hardware library {match.group(1)!r}; "
        "real-hardware code is forbidden in this repository"
    )
