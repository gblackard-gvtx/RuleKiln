"""Strategy family classification and scoring utilities."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from rulekiln.pipeline.baseline_strategies import (
    EMBEDDING_CENTROID_STRATEGY,
    EMBEDDING_KNN_STRATEGY_TO_K,
)
from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.pipeline.strategy_families import strategy_family
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import EvalResult
from rulekiln.schemas.task_case import RuleKilnCase, TaskMode

DISTILLED_STRATEGIES: tuple[str, str] = ("dbscan", "hdbscan")
_EMBEDDING_KNN_STRATEGIES: tuple[str, ...] = tuple(EMBEDDING_KNN_STRATEGY_TO_K.keys())
EMBEDDING_BASELINE_STRATEGIES: tuple[str, ...] = (
    EMBEDDING_CENTROID_STRATEGY,
    *_EMBEDDING_KNN_STRATEGIES,
)

_KNOWN_TASK_MODES: frozenset[str] = frozenset(
    {
        "classification",
        "summarization",
        "extraction",
        "rubric_review",
        "routing",
        "tool_use",
        "freeform_generation",
        "agent_behavior",
    }
)


def _strategy_family(strategy: str) -> str:
    """Classify a strategy into baseline / embedding_baseline / retrieval_baseline / distilled.

    Delegates to rulekiln.pipeline.strategy_families so the pipeline layer and the
    worker layer agree; embedding and retrieval baselines are deliberately NOT part
    of the prompt-baseline family so they cannot be reported as prompt-level wins.
    """
    return strategy_family(strategy)


def _best_strategy_for_family(
    *,
    strategy_evals: dict[str, EvalResult],
    strategy_prompt_tokens: dict[str, int],
    primary_metric: str,
    family: str,
) -> str | None:
    candidates = [
        strategy_name
        for strategy_name in strategy_evals
        if _strategy_family(strategy_name) == family
    ]
    if not candidates:
        return None

    def sort_key(strategy_name: str) -> tuple[float, float, int, str]:
        strategy_eval = strategy_evals[strategy_name]
        fallback_token_count = (
            strategy_eval.prompt_token_count if strategy_eval.prompt_token_count is not None else 0
        )
        token_count = strategy_prompt_tokens.get(strategy_name, fallback_token_count)
        return (
            -_primary_score_for_metric(strategy_eval, primary_metric),
            strategy_eval.malformed_output_rate,
            token_count,
            strategy_name,
        )

    return sorted(candidates, key=sort_key)[0]


def _primary_score_for_metric(result: EvalResult | None, primary_metric: str) -> float:
    if result is None:
        return 0.0
    metric_name = primary_metric.strip().lower()
    if metric_name == "macro_f1":
        return float(result.macro_f1 or 0.0)
    if metric_name == "accuracy":
        return float(result.accuracy or 0.0)
    return float(result.weighted_case_score or 0.0)


def _delta_vs_baseline(
    result: EvalResult | None,
    baseline: EvalResult | None,
    primary_metric: str,
) -> float:
    if result is None or baseline is None:
        return 0.0
    return _primary_score_for_metric(result, primary_metric) - _primary_score_for_metric(
        baseline, primary_metric
    )


def _resolve_primary_metric(payload: DistillationRequest, task_mode: str) -> str:
    if payload.metric and payload.metric.strip():
        return payload.metric.strip()
    if task_mode in _KNOWN_TASK_MODES:
        return get_primary_metric(cast(TaskMode, task_mode))
    return "weighted_case_score"


def _build_dataset_identifier(cases: list[RuleKilnCase]) -> str:
    sorted_cases = sorted(cases, key=lambda case: case.id)
    case_payloads = [case.model_dump(mode="json") for case in sorted_cases]
    serialized = json.dumps(case_payloads, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"cases_sha256:{digest}"
