"""Tests for checksum and certificate-safe hand model downloading."""
from __future__ import annotations

import hashlib
import io
import ssl
from pathlib import Path

from scripts import download_hand_landmarker


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_download_passes_explicit_certificate_context(
    monkeypatch, tmp_path: Path
) -> None:
    contexts: list[ssl.SSLContext] = []

    def fake_urlopen(request, *, context, timeout):
        contexts.append(context)
        assert request.full_url == download_hand_landmarker.MODEL_URL
        assert timeout == 60
        return FakeResponse(b"verified model bytes")

    monkeypatch.setattr(download_hand_landmarker.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "model.task"
    download_hand_landmarker.download(
        download_hand_landmarker.MODEL_URL,
        destination,
    )

    assert destination.read_bytes() == b"verified model bytes"
    assert len(contexts) == 1
    assert isinstance(contexts[0], ssl.SSLContext)


def test_main_verifies_download_before_atomic_install(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"known hand model"
    expected = hashlib.sha256(payload).hexdigest()

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    monkeypatch.setattr(download_hand_landmarker, "download", fake_download)
    monkeypatch.setattr(download_hand_landmarker, "MODEL_SHA256", expected)
    output = tmp_path / "models" / "hand_landmarker.task"

    assert download_hand_landmarker.main(["--output", str(output)]) == 0
    assert output.read_bytes() == payload
