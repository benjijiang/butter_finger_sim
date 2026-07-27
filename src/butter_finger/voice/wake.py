"""Local PocketSphinx keyword spotting for the exact activation phrase."""
from __future__ import annotations

from typing import Any, Protocol


class WakeWordError(RuntimeError):
    """The local wake detector cannot be initialized."""


class WakeWordDetector(Protocol):
    def process(self, pcm16: bytes) -> bool: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class PocketSphinxWakeWordDetector:
    """Streaming keyphrase detector that never sends wake audio off-device."""

    def __init__(
        self,
        *,
        phrase: str,
        threshold: float,
        sample_rate_hz: int = 16000,
        decoder: Any | None = None,
    ) -> None:
        if phrase != "butter finger":
            raise ValueError("wake phrase must be exactly 'butter finger'")
        if threshold <= 0:
            raise ValueError("wake threshold must be greater than zero")
        if decoder is None:
            try:
                from pocketsphinx import Decoder
            except ImportError as exc:
                raise WakeWordError(
                    "PocketSphinx is not installed; run "
                    "pip install -e '.[voice]'"
                ) from exc
            decoder = Decoder(
                keyphrase=phrase,
                kws_threshold=threshold,
                samprate=sample_rate_hz,
            )
        self._phrase = phrase
        self._decoder = decoder
        self._active = False
        self.reset()

    def reset(self) -> None:
        if self._active:
            self._decoder.end_utt()
        self._decoder.start_utt()
        self._active = True

    def process(self, pcm16: bytes) -> bool:
        if len(pcm16) % 2:
            raise ValueError("PocketSphinx input must be PCM16")
        self._decoder.process_raw(pcm16, False, False)
        hypothesis = self._decoder.hyp()
        if hypothesis is None:
            return False
        text = " ".join(str(hypothesis.hypstr).casefold().split())
        if text != self._phrase:
            return False
        self.reset()
        return True

    def close(self) -> None:
        if self._active:
            self._decoder.end_utt()
            self._active = False
