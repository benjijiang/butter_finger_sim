"""Thread-safe latest-only result buffering for asynchronous perception."""
from __future__ import annotations

import threading
from typing import Generic, Protocol, TypeVar


class Timestamped(Protocol):
    timestamp_ms: int


T = TypeVar("T", bound=Timestamped)


class LatestResultBuffer(Generic[T]):
    """One-slot buffer; publishing a new unread result drops the old one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: T | None = None
        self._last_taken_timestamp_ms = -1
        self._overwritten_count = 0

    @property
    def overwritten_count(self) -> int:
        with self._lock:
            return self._overwritten_count

    def publish(self, result: T) -> None:
        with self._lock:
            if (
                self._latest is not None
                and self._latest.timestamp_ms > self._last_taken_timestamp_ms
            ):
                self._overwritten_count += 1
            if self._latest is None or result.timestamp_ms >= self._latest.timestamp_ms:
                self._latest = result

    def take_latest(self, *, now_ms: int, max_age_ms: int) -> T | None:
        """Return each fresh result at most once; stale/future results are ignored."""
        with self._lock:
            result = self._latest
            if result is None or result.timestamp_ms <= self._last_taken_timestamp_ms:
                return None
            age_ms = now_ms - result.timestamp_ms
            if age_ms < 0 or age_ms > max_age_ms:
                return None
            self._last_taken_timestamp_ms = result.timestamp_ms
            return result
