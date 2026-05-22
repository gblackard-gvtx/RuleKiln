"""Structured event helpers for model calls, stage timings, and token usage."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from rulekiln.observability.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def stage_timing(job_id: str, stage: str) -> AsyncIterator[None]:
    """Context manager that logs start/end of a pipeline stage with elapsed time."""
    start = time.monotonic()
    logger.info("stage_start", job_id=job_id, stage=stage)
    try:
        yield
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error(
            "stage_error", job_id=job_id, stage=stage, elapsed_s=round(elapsed, 3), error=str(exc)
        )
        raise
    else:
        elapsed = time.monotonic() - start
        logger.info("stage_end", job_id=job_id, stage=stage, elapsed_s=round(elapsed, 3))


def log_model_call(
    job_id: str,
    role: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    """Log a model invocation with token usage."""
    logger.info(
        "model_call",
        job_id=job_id,
        role=role,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def log_retry(job_id: str, stage: str, attempt: int, reason: str) -> None:
    """Log a retry attempt within a stage."""
    logger.warning("stage_retry", job_id=job_id, stage=stage, attempt=attempt, reason=reason)


def log_token_budget(job_id: str, strategy: str, prompt_tokens: int, budget: int) -> None:
    """Log token budget status for a compiled prompt."""
    over = prompt_tokens > budget
    logger.info(
        "token_budget",
        job_id=job_id,
        strategy=strategy,
        prompt_tokens=prompt_tokens,
        budget=budget,
        over_budget=over,
    )
