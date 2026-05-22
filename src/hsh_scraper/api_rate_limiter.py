"""Rate limiting helpers for outbound API clients."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

import httpx


class SlidingWindowRateLimiter:
    """Synchroner Sliding-Window-Limiter fuer ausgehende Requests."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        on_wait=None,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._on_wait = on_wait
        self._lock = Lock()
        self._timestamps: deque[float] = deque()

    def wait(self, request: httpx.Request | None = None) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                wait_seconds = self.window_seconds - (now - self._timestamps[0])

            wait_seconds = max(0.05, wait_seconds)
            if self._on_wait is not None:
                self._on_wait(wait_seconds)
            time.sleep(wait_seconds)


def build_rate_limited_http_client(
    *,
    timeout: float | httpx.Timeout = 30.0,
    rate_limiter: SlidingWindowRateLimiter | None = None,
) -> httpx.Client:
    event_hooks = {}
    if rate_limiter is not None:
        event_hooks["request"] = [rate_limiter.wait]
    return httpx.Client(timeout=timeout, event_hooks=event_hooks)


__all__ = [
    "SlidingWindowRateLimiter",
    "build_rate_limited_http_client",
]
