from hsh_scraper import api_rate_limiter
from hsh_scraper.api_rate_limiter import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_waits_after_limit(monkeypatch):
    clock = {"now": 0.0}
    sleeps = []

    def monotonic():
        return clock["now"]

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(api_rate_limiter.time, "monotonic", monotonic)
    monkeypatch.setattr(api_rate_limiter.time, "sleep", sleep)

    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)

    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert sleeps == [60.0]
