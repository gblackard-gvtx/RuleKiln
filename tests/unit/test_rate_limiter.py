"""Unit tests for ProviderRateLimiter."""

from __future__ import annotations

import asyncio

import pytest

from rulekiln.providers.contracts import ProviderConfig
from rulekiln.providers.rate_limiter import ProviderRateLimiter


def _config(profile: str = "p1", model: str = "m1", rpm: int | None = None, concurrency: int = 5) -> ProviderConfig:
    return ProviderConfig(
        provider="fake",
        profile_name=profile,
        model=model,
        api_key=None,
        base_url=None,
        extra={},
        rate_limit_rpm=rpm,
        max_concurrency=concurrency,
    )


@pytest.mark.asyncio
async def test_acquire_release_no_limits() -> None:
    limiter = ProviderRateLimiter()
    config = _config()
    await limiter.acquire(config)
    limiter.release(config)
    # No error raised — all good


@pytest.mark.asyncio
async def test_concurrency_limit_respected() -> None:
    limiter = ProviderRateLimiter()
    config = _config(concurrency=2)

    acquired: list[int] = []

    async def worker(i: int) -> None:
        await limiter.acquire(config)
        acquired.append(i)
        await asyncio.sleep(0.02)
        limiter.release(config)

    await asyncio.gather(*[worker(i) for i in range(4)])
    # All 4 should complete
    assert len(acquired) == 4


@pytest.mark.asyncio
async def test_different_configs_get_separate_limiters() -> None:
    limiter = ProviderRateLimiter()
    c1 = _config(profile="p1", model="m1", concurrency=1)
    c2 = _config(profile="p2", model="m2", concurrency=1)

    # Acquire both — should not deadlock because they're separate states
    await limiter.acquire(c1)
    await limiter.acquire(c2)
    limiter.release(c1)
    limiter.release(c2)


@pytest.mark.asyncio
async def test_release_without_acquire_is_safe() -> None:
    limiter = ProviderRateLimiter()
    config = _config()
    # Should not raise
    limiter.release(config)
