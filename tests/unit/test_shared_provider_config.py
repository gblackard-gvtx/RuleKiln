"""Regression test for item I.10 / F: generation settings must be identical across
baseline_scaffold, dbscan, and hdbscan — the same ProviderConfig instance is
resolved once per student role and reused for every strategy's evaluate_prompt()
call (see baseline_eval_service.py and distilled_eval_service.py), so no strategy
can silently diverge on temperature, max_tokens, or stop sequences.
"""

from __future__ import annotations

from rulekiln.providers.contracts import ProviderConfig


def test_provider_config_has_no_per_strategy_generation_fields() -> None:
    """ProviderConfig carries connection/rate-limit settings only — no temperature,
    top_p, max_tokens, or stop-sequence field exists to diverge per strategy."""
    field_names = set(ProviderConfig.model_fields)
    forbidden = {"temperature", "top_p", "max_tokens", "stop", "stop_sequences"}
    assert field_names.isdisjoint(forbidden)


def test_provider_config_is_strategy_agnostic() -> None:
    """resolve_provider_config() takes no `strategy` argument — a single
    ProviderConfig is built per role (student/teacher/embedding/judge), not per
    distillation strategy, so baseline_scaffold/dbscan/hdbscan structurally share it."""
    import inspect

    from rulekiln.providers.resolver import resolve_provider_config

    params = inspect.signature(resolve_provider_config).parameters
    assert "strategy" not in params
