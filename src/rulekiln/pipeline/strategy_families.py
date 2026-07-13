"""Strategy family classification.

Families keep the paper's comparison honest: prompt-level baselines, embedding
baselines, retrieval baselines, and distilled (clustered rule) strategies are
distinct categories, and an embedding baseline win must never be presented as a
prompt-level distillation win.
"""

from __future__ import annotations

DISTILLED_FAMILY = "distilled"
BASELINE_FAMILY = "baseline"
EMBEDDING_BASELINE_FAMILY = "embedding_baseline"
RETRIEVAL_BASELINE_FAMILY = "retrieval_baseline"

_DISTILLED_STRATEGIES: frozenset[str] = frozenset({"dbscan", "hdbscan"})


def strategy_family(strategy: str) -> str:
    """Classify a strategy id into its comparison family."""
    if strategy in _DISTILLED_STRATEGIES:
        return DISTILLED_FAMILY
    if strategy.startswith("embedding_"):
        return EMBEDDING_BASELINE_FAMILY
    if strategy.startswith("retrieval_"):
        return RETRIEVAL_BASELINE_FAMILY
    return BASELINE_FAMILY
