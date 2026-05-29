"""Provider call tracking: context variable, collector, and per-call logging."""

from __future__ import annotations

import hashlib
import time
from contextvars import ContextVar
from typing import Awaitable, Callable

from rulekiln.observability.logging import get_logger
from rulekiln.schemas.usage import (
    ChatCompletionResult,
    EmbeddingResult,
    ModelCallContext,
    ModelCallRecord,
    ModelUsage,
)
from rulekiln.usage.pricing import PricingService

logger = get_logger(__name__)

# ── Context Variables ─────────────────────────────────────────────────────────

_tracking_context_var: ContextVar[ModelCallContext | None] = ContextVar(
    "_tracking_context_var", default=None
)
_collector_var: ContextVar[ModelCallCollector | None] = ContextVar(
    "_collector_var", default=None
)

_pricing_service: PricingService | None = None


def _get_pricing_service() -> PricingService:
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service


# ── Collector ─────────────────────────────────────────────────────────────────


class ModelCallCollector:
    """Accumulates ModelCallRecord instances during a pipeline run."""

    def __init__(self) -> None:
        self._records: list[ModelCallRecord] = []

    def add(self, record: ModelCallRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[ModelCallRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


# ── Context Management ────────────────────────────────────────────────────────


def set_tracking_context(ctx: ModelCallContext, collector: ModelCallCollector) -> None:
    """Set the active tracking context and collector for the current async task."""
    _tracking_context_var.set(ctx)
    _collector_var.set(collector)


def clear_tracking_context() -> None:
    """Clear the active tracking context."""
    _tracking_context_var.set(None)
    _collector_var.set(None)


def get_tracking_context() -> ModelCallContext | None:
    """Get the active tracking context for the current async task."""
    return _tracking_context_var.get()


def get_collector() -> ModelCallCollector | None:
    """Get the active collector for the current async task."""
    return _collector_var.get()


def update_tracking_context(**kwargs: object) -> None:
    """Return an updated context with the given fields changed.

    Useful for updating case_id or strategy within a stage loop.
    """
    ctx = _tracking_context_var.get()
    if ctx is not None:
        _tracking_context_var.set(ctx.model_copy(update=kwargs))  # pyright: ignore[reportArgumentType]


# ── Tracking Helpers ──────────────────────────────────────────────────────────


def _record_call(
    *,
    ctx: ModelCallContext,
    collector: ModelCallCollector,
    usage: ModelUsage,
    latency_ms: int,
    status: str,
    request_fingerprint: str,
    error_type: str | None = None,
) -> None:
    """Build a ModelCallRecord, log it, and append it to the collector."""
    pricing = _get_pricing_service()
    cost = pricing.calculate(provider=ctx.provider, model=ctx.model, usage=usage)

    record = ModelCallRecord(
        job_id=ctx.job_id,
        stage=ctx.stage,
        role=ctx.role,
        provider_profile=ctx.provider_profile,
        provider=ctx.provider,
        model=ctx.model,
        student_id=ctx.student_id,
        strategy=ctx.strategy,
        case_id=ctx.case_id,
        usage=usage,
        cost=cost,
        latency_ms=latency_ms,
        status=status,  # pyright: ignore[reportArgumentType]
        error_type=error_type,
        idempotency_key=_build_idempotency_key(ctx, request_fingerprint),
    )

    collector.add(record)

    logger.info(
        "model_call_tracked",
        job_id=ctx.job_id,
        stage=ctx.stage,
        role=ctx.role,
        provider=ctx.provider,
        model=ctx.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=str(cost.total_cost_usd),
        latency_ms=latency_ms,
        status=status,
        case_id=ctx.case_id,
    )


def _build_idempotency_key(ctx: ModelCallContext, request_fingerprint: str) -> str:
    """Build a stable key for de-duplicating persisted model call events."""
    strategy = ctx.strategy or "-"
    student_id = ctx.student_id or "-"
    case_id = ctx.case_id or "-"
    request_hash = _hash_request_fingerprint(request_fingerprint)
    return (
        f"{ctx.job_id}:{ctx.stage}:{ctx.role}:"
        f"{strategy}:{student_id}:{case_id}:{request_hash}"
    )


def _hash_request_fingerprint(request_fingerprint: str) -> str:
    if not request_fingerprint:
        return "empty"
    digest = hashlib.sha256(request_fingerprint.encode("utf-8")).hexdigest()
    return digest[:16]


# ── Tracked Call Wrappers ─────────────────────────────────────────────────────


async def tracked_chat_call(
    *,
    call: Callable[[], Awaitable[ChatCompletionResult]],
    fallback_input_text: str = "",
    fallback_output_text: str = "",
) -> ChatCompletionResult:
    """Execute a chat provider call and record usage/cost to the active collector.

    If no tracking context is active, the call is executed without tracking.
    """
    from rulekiln.providers.estimation import estimate_usage_from_text

    ctx = _tracking_context_var.get()
    collector = _collector_var.get()

    started = time.monotonic()
    try:
        result = await call()
        latency_ms = int((time.monotonic() - started) * 1000)

        if ctx is not None and collector is not None:
            usage = result.usage or estimate_usage_from_text(
                input_text=fallback_input_text,
                output_text=result.content,
            )
            _record_call(
                ctx=ctx,
                collector=collector,
                usage=usage,
                latency_ms=latency_ms,
                status="success",
                request_fingerprint=fallback_input_text,
            )

        return result

    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        if ctx is not None and collector is not None:
            from rulekiln.providers.estimation import estimate_usage_from_text
            usage = estimate_usage_from_text(input_text=fallback_input_text)
            _record_call(
                ctx=ctx,
                collector=collector,
                usage=usage,
                latency_ms=latency_ms,
                status="failed",
                request_fingerprint=fallback_input_text,
                error_type=type(exc).__name__,
            )
        raise


async def tracked_embedding_call(
    *,
    call: Callable[[], Awaitable[EmbeddingResult]],
    fallback_input_text: str = "",
) -> EmbeddingResult:
    """Execute an embedding provider call and record usage/cost to the active collector.

    If no tracking context is active, the call is executed without tracking.
    """
    from rulekiln.providers.estimation import estimate_usage_from_text

    ctx = _tracking_context_var.get()
    collector = _collector_var.get()

    started = time.monotonic()
    try:
        result = await call()
        latency_ms = int((time.monotonic() - started) * 1000)

        if ctx is not None and collector is not None:
            usage = result.usage or estimate_usage_from_text(
                input_text=fallback_input_text,
            )
            _record_call(
                ctx=ctx,
                collector=collector,
                usage=usage,
                latency_ms=latency_ms,
                status="success",
                request_fingerprint=fallback_input_text,
            )

        return result

    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        if ctx is not None and collector is not None:
            usage = estimate_usage_from_text(input_text=fallback_input_text)
            _record_call(
                ctx=ctx,
                collector=collector,
                usage=usage,
                latency_ms=latency_ms,
                status="failed",
                request_fingerprint=fallback_input_text,
                error_type=type(exc).__name__,
            )
        raise
