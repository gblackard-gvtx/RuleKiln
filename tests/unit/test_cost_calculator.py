"""Unit tests for PricingService (cost calculator)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rulekiln.schemas.usage import ModelUsage
from rulekiln.usage.pricing import PricingService


@pytest.fixture()
def pricing() -> PricingService:
    return PricingService()


def _usage(input_tokens: int, output_tokens: int) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated=False,
    )


def test_cost_calculator_uses_pricing_config(pricing: PricingService) -> None:
    """OpenAI gpt-4o-mini has a known pricing entry; cost must be non-zero."""
    usage = _usage(1_000_000, 1_000_000)
    cost = pricing.calculate(provider="openai", model="gpt-4o-mini", usage=usage)
    assert cost.input_cost_usd > Decimal("0")
    assert cost.output_cost_usd > Decimal("0")
    assert cost.total_cost_usd == cost.input_cost_usd + cost.output_cost_usd


def test_cost_calculator_uses_decimal_math(pricing: PricingService) -> None:
    """Cost must be a Decimal, not a float (to avoid floating-point errors)."""
    usage = _usage(1000, 500)
    cost = pricing.calculate(provider="openai", model="gpt-4o-mini", usage=usage)
    assert isinstance(cost.input_cost_usd, Decimal)
    assert isinstance(cost.output_cost_usd, Decimal)
    assert isinstance(cost.total_cost_usd, Decimal)


def test_local_model_cost_defaults_to_zero(pricing: PricingService) -> None:
    """openai_compatible default entry has $0 pricing."""
    usage = _usage(10_000, 5_000)
    cost = pricing.calculate(provider="openai_compatible", model="mistral-7b-instruct", usage=usage)
    assert cost.total_cost_usd == Decimal("0")


def test_missing_pricing_marks_cost_estimated(pricing: PricingService) -> None:
    """An unknown provider/model must produce cost with estimated=True."""
    usage = _usage(1000, 1000)
    cost = pricing.calculate(
        provider="totally_unknown_provider", model="mystery-model-v99", usage=usage
    )
    assert cost.estimated is True


def test_embedding_cost_uses_input_tokens_only(pricing: PricingService) -> None:
    """For embedding models, output_tokens should be zero; only input is billed."""
    usage = ModelUsage(input_tokens=100_000, output_tokens=0, total_tokens=100_000, estimated=False)
    cost = pricing.calculate(provider="openai", model="text-embedding-3-small", usage=usage)
    assert cost.output_cost_usd == Decimal("0")
    assert cost.input_cost_usd >= Decimal("0")
