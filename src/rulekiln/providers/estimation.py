"""Token count estimation fallback for providers that don't return usage."""

from __future__ import annotations

import math

from rulekiln.schemas.usage import ModelUsage


def estimate_usage_from_text(
    *,
    input_text: str,
    output_text: str = "",
) -> ModelUsage:
    """Estimate token usage from text character counts.

    Uses the approximation: 1 token ≈ 4 characters.
    This is a conservative MVP fallback for providers that do not return usage.
    """
    input_tokens = math.ceil(len(input_text) / 4)
    output_tokens = math.ceil(len(output_text) / 4)
    total_tokens = input_tokens + output_tokens

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        estimated=True,
    )


def build_usage_from_provider(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None = None,
) -> ModelUsage:
    """Build a ModelUsage from provider-supplied token counts."""
    computed_total = total_tokens
    if computed_total is None and input_tokens is not None and output_tokens is not None:
        computed_total = input_tokens + output_tokens

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=computed_total,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        estimated=False,
    )
