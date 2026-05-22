"""In-process provider rate limiter: per-config semaphore + sliding-window RPM."""

from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog

from rulekiln.providers.contracts import ProviderConfig

logger = structlog.get_logger(__name__)


class _RpmWindow:
    """Sliding-window requests-per-minute tracker."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._timestamps: deque[float] = deque()

    def seconds_until_allowed(self) -> float:
        """Return how many seconds to wait before another request is allowed.

        Returns 0.0 if a slot is immediately available.
        """
        now = time.monotonic()
        window_start = now - 60.0
        # Drop timestamps outside the rolling window
        while self._timestamps and self._timestamps[0] < window_start:
            self._timestamps.popleft()

        if len(self._timestamps) < self._rpm:
            return 0.0

        # Oldest request timestamp + 60 s = when the slot frees
        return self._timestamps[0] - window_start

    def record(self) -> None:
        self._timestamps.append(time.monotonic())


class _LimiterState:
    def __init__(self, config: ProviderConfig) -> None:
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._rpm_window: _RpmWindow | None = (
            _RpmWindow(config.rate_limit_rpm) if config.rate_limit_rpm else None
        )
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int | None = None) -> float:
        """Acquire a slot; return total seconds waited."""
        wait_start = time.monotonic()

        # ── RPM check ────────────────────────────────────────────────────
        if self._rpm_window is not None:
            while True:
                async with self._lock:
                    wait_s = self._rpm_window.seconds_until_allowed()
                    if wait_s <= 0:
                        self._rpm_window.record()
                        break
                await asyncio.sleep(wait_s)

        # ── Concurrency semaphore ─────────────────────────────────────────
        await self._semaphore.acquire()
        return time.monotonic() - wait_start

    def release(self) -> None:
        self._semaphore.release()


class ProviderRateLimiter:
    """Manages per-ProviderConfig rate limiter state.

    A single shared instance should be held for the lifetime of a worker/request
    so that semaphores are shared across all concurrent calls to the same profile.
    """

    def __init__(self) -> None:
        self._states: dict[str, _LimiterState] = {}
        self._init_lock = asyncio.Lock()

    async def _get_state(self, config: ProviderConfig) -> _LimiterState:
        key = f"{config.profile_name}:{config.model}"
        if key not in self._states:
            async with self._init_lock:
                if key not in self._states:
                    self._states[key] = _LimiterState(config)
        return self._states[key]

    async def acquire(
        self,
        config: ProviderConfig,
        estimated_tokens: int | None = None,
    ) -> None:
        """Acquire a rate-limited slot; logs wait time if non-trivial."""
        state = await self._get_state(config)
        wait_s = await state.acquire(estimated_tokens)
        if wait_s > 0.05:
            logger.info(
                "rate_limit_wait",
                profile=config.profile_name,
                model=config.model,
                wait_seconds=round(wait_s, 3),
            )

    def release(self, config: ProviderConfig) -> None:
        """Release the concurrency semaphore slot for this config."""
        key = f"{config.profile_name}:{config.model}"
        if key in self._states:
            self._states[key].release()


# Module-level singleton used by provider adapters.
_global_limiter: ProviderRateLimiter | None = None


def get_rate_limiter() -> ProviderRateLimiter:
    """Return the process-global ProviderRateLimiter, creating it on first call."""
    global _global_limiter  # noqa: PLW0603
    if _global_limiter is None:
        _global_limiter = ProviderRateLimiter()
    return _global_limiter
