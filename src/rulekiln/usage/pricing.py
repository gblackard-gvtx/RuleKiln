"""Pricing service: calculates estimated cost from token usage and pricing config."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

from rulekiln.observability.logging import get_logger
from rulekiln.schemas.usage import ModelCallCost, ModelUsage

logger = get_logger(__name__)

_PRICING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_pricing.yaml"

_ZERO = Decimal("0")
_M = Decimal("1000000")


def _load_pricing_config(path: Path = _PRICING_CONFIG_PATH) -> dict[str, dict[str, dict[str, str]]]:
    """Load pricing config from YAML file."""
    if not path.exists():
        logger.warning("pricing_config_missing", path=str(path))
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw.get("pricing", {})  # pyright: ignore[reportReturnType]


class PricingService:
    """Calculates estimated USD cost from token usage and pricing configuration."""

    def __init__(self, config_path: Path = _PRICING_CONFIG_PATH) -> None:
        self._pricing: dict[str, dict[str, dict[str, str]]] = _load_pricing_config(config_path)

    def _lookup(self, provider: str, model: str) -> tuple[Decimal, Decimal, str]:
        """Return (input_per_1m, output_per_1m, pricing_source) for a provider/model pair."""
        provider_pricing = self._pricing.get(provider, {})

        # Try exact model match first
        model_entry = provider_pricing.get(model)
        if model_entry is None:
            # Fall back to default entry for the provider
            model_entry = provider_pricing.get("default")

        if model_entry is None:
            return _ZERO, _ZERO, "missing_pricing_config"

        input_rate = Decimal(model_entry.get("input_per_1m_tokens_usd", "0"))
        output_rate = Decimal(model_entry.get("output_per_1m_tokens_usd", "0"))
        source: str = model_entry.get("source", "config")
        return input_rate, output_rate, source

    def calculate(
        self,
        *,
        provider: str,
        model: str,
        usage: ModelUsage,
    ) -> ModelCallCost:
        """Calculate estimated cost for a model call."""
        input_rate, output_rate, pricing_source = self._lookup(provider, model)

        input_tokens = Decimal(usage.input_tokens or 0)
        output_tokens = Decimal(usage.output_tokens or 0)

        input_cost = (input_tokens / _M * input_rate).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        output_cost = (output_tokens / _M * output_rate).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        total_cost = input_cost + output_cost

        return ModelCallCost(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            pricing_source=pricing_source,
            estimated=True,
        )
