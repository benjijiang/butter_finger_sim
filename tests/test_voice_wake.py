"""Dependency-free PocketSphinx wake detector behavior."""
from __future__ import annotations

from dataclasses import dataclass

from butter_finger.voice.wake import PocketSphinxWakeWordDetector


@dataclass
class Hypothesis:
    hypstr: str


class FakeDecoder:
    def __init__(self) -> None:
        self.current: Hypothesis | None = None
        self.starts = 0
        self.ends = 0
        self.processed: list[bytes] = []

    def start_utt(self) -> None:
        self.starts += 1

    def end_utt(self) -> None:
        self.ends += 1

    def process_raw(self, pcm: bytes, _no_search: bool, _full_utt: bool) -> None:
        self.processed.append(pcm)

    def hyp(self) -> Hypothesis | None:
        result = self.current
        self.current = None
        return result


def test_exact_butter_finger_detection_resets_decoder() -> None:
    decoder = FakeDecoder()
    detector = PocketSphinxWakeWordDetector(
        phrase="butter finger",
        threshold=1e-20,
        decoder=decoder,
    )
    decoder.current = Hypothesis("  BUTTER   FINGER ")

    assert detector.process(b"\x00\x00") is True
    assert decoder.starts == 2
    assert decoder.ends == 1


def test_non_exact_hypothesis_does_not_activate() -> None:
    decoder = FakeDecoder()
    detector = PocketSphinxWakeWordDetector(
        phrase="butter finger",
        threshold=1e-20,
        decoder=decoder,
    )
    decoder.current = Hypothesis("butter fingers")

    assert detector.process(b"\x00\x00") is False
    assert decoder.starts == 1
    assert decoder.ends == 0
