#!/usr/bin/env python3
"""Download and verify the official MediaPipe Hand Landmarker model asset."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certificate_context() -> ssl.SSLContext:
    """Use certifi when available, avoiding broken macOS Python CA installs."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, destination: Path) -> None:
    """Stream one HTTPS asset using an explicit, verified CA bundle."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "butter-finger-sim-model-downloader/1"},
    )
    with urllib.request.urlopen(
        request,
        context=certificate_context(),
        timeout=60,
    ) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()

    if output.is_file() and sha256(output) == MODEL_SHA256:
        print(f"Verified existing model: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix="hand_landmarker-", suffix=".task", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"Downloading official MediaPipe model to {output} ...")
        download(MODEL_URL, temporary)
        actual = sha256(temporary)
        if actual != MODEL_SHA256:
            print(
                f"ERROR: model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}",
                file=sys.stderr,
            )
            return 1
        os.replace(temporary, output)
    except (OSError, urllib.error.URLError) as exc:
        print(f"ERROR: could not download model: {exc}", file=sys.stderr)
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print(
                "The teleop dependency set includes certifi; reinstall it with:\n"
                "    python -m pip install --upgrade certifi\n"
                "then retry this command.",
                file=sys.stderr,
            )
        return 1
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Downloaded and verified: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
