"""Strategy selection with tie-break logic (C005)."""

from __future__ import annotations

from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.pipeline.strategy_families import strategy_family
from rulekiln.schemas.pipeline import (
    EvalResult,
    QualityGateResult,
    StrategyComparison,
    StrategySelectionRow,
)
from rulekiln.schemas.task_case import TaskMode

_PRIMARY_METRIC_EPSILON = 0.005
_MALFORMED_EPSILON = 0.001


def _primary_score(result: EvalResult, task_mode: TaskMode) -> float:
    metric = get_primary_metric(task_mode)
    if metric == "macro_f1":
        return result.macro_f1 or 0.0
    return result.weighted_case_score or 0.0


def _select_lowest_golden_failures(
    contenders: list[str],
    strategy_gates: dict[str, QualityGateResult],
) -> list[str]:
    golden_by_strategy: dict[str, int] = {}
    for strategy in contenders:
        strategy_gate = strategy_gates.get(strategy)
        golden_by_strategy[strategy] = (
            strategy_gate.golden_failures if strategy_gate is not None else 0
        )
    min_golden = min(golden_by_strategy.values())
    return [
        strategy for strategy in contenders if golden_by_strategy.get(strategy, 0) == min_golden
    ]


def _select_lowest_malformed(
    contenders: list[str],
    strategy_evals: dict[str, EvalResult],
) -> list[str]:
    malformed_by_strategy = {
        strategy: strategy_evals[strategy].malformed_output_rate for strategy in contenders
    }
    min_malformed = min(malformed_by_strategy.values())
    return [
        strategy
        for strategy in contenders
        if abs(malformed_by_strategy[strategy] - min_malformed) <= _MALFORMED_EPSILON
    ]


def _select_lowest_prompt_tokens(
    contenders: list[str],
    strategy_prompt_tokens: dict[str, int],
) -> list[str]:
    token_by_strategy = {
        strategy: strategy_prompt_tokens.get(strategy, 0) for strategy in contenders
    }
    min_tokens = min(token_by_strategy.values())
    return [strategy for strategy in contenders if token_by_strategy.get(strategy, 0) == min_tokens]


def select_strategy_generic(
    *,
    task_mode: TaskMode,
    strategy_evals: dict[str, EvalResult],
    strategy_gates: dict[str, QualityGateResult],
    strategy_prompt_tokens: dict[str, int],
    baseline_strategy: str,
) -> tuple[str, str]:
    excluded_candidates: set[str] = {baseline_strategy}
    if baseline_strategy != "baseline":
        excluded_candidates.add("baseline")

    candidate_evals = {
        strategy: result
        for strategy, result in strategy_evals.items()
        if strategy not in excluded_candidates
    }
    if not candidate_evals:
        return (
            baseline_strategy,
            "No non-baseline strategies available; selecting baseline scaffold.",
        )

    passing_candidates = {
        strategy: result
        for strategy, result in candidate_evals.items()
        if strategy not in strategy_gates or strategy_gates[strategy].passed
    }
    if not passing_candidates:
        # Every candidate failed its quality gates. Ranking the failed pool would
        # promote a strategy that is strictly worse than the baseline (that is how
        # embedding_knn_k1 with macro_f1 0.61 beat baseline_scaffold at 0.71), so
        # fall back to the baseline scaffold and report the gate failures instead.
        failed_summary = "; ".join(
            f"{strategy}: {', '.join(strategy_gates[strategy].violations)}"
            for strategy in sorted(candidate_evals)
            if strategy in strategy_gates and strategy_gates[strategy].violations
        )
        return (
            baseline_strategy,
            f"All {len(candidate_evals)} candidate strategies failed quality gates; "
            f"selecting baseline scaffold. Gate failures — {failed_summary}",
        )
    ranked_pool = passing_candidates

    score_by_strategy = {
        strategy: _primary_score(result, task_mode) for strategy, result in ranked_pool.items()
    }
    top_score = max(score_by_strategy.values())
    contenders = [
        strategy
        for strategy, score in score_by_strategy.items()
        if (top_score - score) <= _PRIMARY_METRIC_EPSILON
    ]
    if len(contenders) == 1:
        selected = contenders[0]
        return (
            selected,
            f"Selected {selected} by primary metric ({score_by_strategy[selected]:.4f}).",
        )

    contenders = _select_lowest_golden_failures(contenders, strategy_gates)
    if len(contenders) == 1:
        selected = contenders[0]
        return selected, "Tied primary metric; selected strategy with fewer golden failures."

    contenders = _select_lowest_malformed(contenders, ranked_pool)
    if len(contenders) == 1:
        selected = contenders[0]
        return selected, "Tied primary metric; selected strategy with lower malformed output rate."

    contenders = _select_lowest_prompt_tokens(contenders, strategy_prompt_tokens)
    if len(contenders) == 1:
        selected = contenders[0]
        return selected, "Tied metrics; selected strategy with lower prompt token count."

    if "hdbscan" in contenders:
        return "hdbscan", "All tie-breaks equal; defaulting to HDBSCAN."
    selected = sorted(contenders)[0]
    return selected, "All tie-breaks equal; defaulting to lexicographically stable strategy order."


def select_strategy(
    task_mode: TaskMode,
    dbscan_eval: EvalResult | None,
    hdbscan_eval: EvalResult | None,
    dbscan_gate: QualityGateResult | None,
    hdbscan_gate: QualityGateResult | None,
    baseline_eval: EvalResult | None,
    dbscan_token_count: int = 0,
    hdbscan_token_count: int = 0,
) -> tuple[str, str]:
    """Legacy wrapper selecting among DBSCAN/HDBSCAN with baseline fallback semantics."""
    strategy_evals: dict[str, EvalResult] = {}
    if baseline_eval is not None:
        strategy_evals["baseline"] = baseline_eval
    if dbscan_eval is not None:
        strategy_evals["dbscan"] = dbscan_eval
    if hdbscan_eval is not None:
        strategy_evals["hdbscan"] = hdbscan_eval

    strategy_gates: dict[str, QualityGateResult] = {}
    if dbscan_gate is not None:
        strategy_gates["dbscan"] = dbscan_gate
    if hdbscan_gate is not None:
        strategy_gates["hdbscan"] = hdbscan_gate

    strategy_prompt_tokens = {
        "dbscan": dbscan_token_count,
        "hdbscan": hdbscan_token_count,
    }
    return select_strategy_generic(
        task_mode=task_mode,
        strategy_evals=strategy_evals,
        strategy_gates=strategy_gates,
        strategy_prompt_tokens=strategy_prompt_tokens,
        baseline_strategy="baseline",
    )


def build_selection_table(
    *,
    task_mode: TaskMode,
    strategy_evals: dict[str, EvalResult],
    strategy_gates: dict[str, QualityGateResult],
    selected_strategy: str,
    selection_reason: str | None,
    baseline_strategy: str,
    unavailable_strategies: dict[str, str] | None = None,
) -> list[StrategySelectionRow]:
    """Build per-strategy selection rows so the report explains every outcome.

    Eligibility: the baseline scaffold is always eligible (it is the fallback);
    any other strategy is eligible only when it has no gate or its gate passed.
    Ranking: eligible strategies are ranked by primary score (descending); the
    ineligible and unavailable strategies carry no rank.
    """
    rows: list[StrategySelectionRow] = []

    def _is_eligible(strategy: str) -> bool:
        if strategy == baseline_strategy or strategy == "baseline":
            return True
        gate = strategy_gates.get(strategy)
        return gate is None or gate.passed

    eligible_scores = {
        strategy: _primary_score(result, task_mode)
        for strategy, result in strategy_evals.items()
        if _is_eligible(strategy)
    }
    ranked = sorted(eligible_scores.items(), key=lambda item: (-item[1], item[0]))
    rank_by_strategy = {strategy: rank for rank, (strategy, _) in enumerate(ranked, start=1)}

    for strategy, result in sorted(strategy_evals.items()):
        gate = strategy_gates.get(strategy)
        eligible = _is_eligible(strategy)
        selected = strategy == selected_strategy
        rows.append(
            StrategySelectionRow(
                strategy=strategy,
                family=strategy_family(strategy),
                eligible=eligible,
                gate_failures=list(gate.violations) if gate is not None else [],
                selection_score=_primary_score(result, task_mode),
                selection_rank=rank_by_strategy.get(strategy),
                selected=selected,
                reason=selection_reason if selected else None,
            )
        )

    for strategy, reason in sorted((unavailable_strategies or {}).items()):
        if strategy in strategy_evals:
            continue
        rows.append(
            StrategySelectionRow(
                strategy=strategy,
                family=strategy_family(strategy),
                eligible=False,
                gate_failures=[reason],
                selection_score=None,
                selection_rank=None,
                selected=False,
                reason=None,
            )
        )

    return rows


def build_strategy_comparison(
    baseline_eval: EvalResult | None,
    dbscan_eval: EvalResult | None,
    hdbscan_eval: EvalResult | None,
    dbscan_gate: QualityGateResult | None,
    hdbscan_gate: QualityGateResult | None,
    task_mode: TaskMode,
    dbscan_token_count: int = 0,
    hdbscan_token_count: int = 0,
    strategy_evals: dict[str, EvalResult] | None = None,
    strategy_gates: dict[str, QualityGateResult] | None = None,
    strategy_prompt_tokens: dict[str, int] | None = None,
    baseline_strategy: str = "baseline",
    unavailable_strategies: dict[str, str] | None = None,
) -> StrategyComparison:
    all_evals: dict[str, EvalResult] = dict(strategy_evals or {})
    if baseline_eval is not None:
        baseline_result = baseline_eval
        if baseline_result.strategy != baseline_strategy:
            baseline_result = baseline_result.model_copy(update={"strategy": baseline_strategy})
        all_evals.setdefault(baseline_strategy, baseline_result)
    if dbscan_eval is not None:
        all_evals.setdefault("dbscan", dbscan_eval)
    if hdbscan_eval is not None:
        all_evals.setdefault("hdbscan", hdbscan_eval)

    all_gates: dict[str, QualityGateResult] = dict(strategy_gates or {})
    if dbscan_gate is not None:
        all_gates.setdefault("dbscan", dbscan_gate)
    if hdbscan_gate is not None:
        all_gates.setdefault("hdbscan", hdbscan_gate)

    all_prompt_tokens: dict[str, int] = dict(strategy_prompt_tokens or {})
    all_prompt_tokens.setdefault("dbscan", dbscan_token_count)
    all_prompt_tokens.setdefault("hdbscan", hdbscan_token_count)

    selected, reason = select_strategy_generic(
        task_mode=task_mode,
        strategy_evals=all_evals,
        strategy_gates=all_gates,
        strategy_prompt_tokens=all_prompt_tokens,
        baseline_strategy=baseline_strategy,
    )
    selection_table = build_selection_table(
        task_mode=task_mode,
        strategy_evals=all_evals,
        strategy_gates=all_gates,
        selected_strategy=selected,
        selection_reason=reason,
        baseline_strategy=baseline_strategy,
        unavailable_strategies=unavailable_strategies,
    )
    return StrategyComparison(
        baseline_eval=baseline_eval,
        dbscan_eval=dbscan_eval,
        hdbscan_eval=hdbscan_eval,
        dbscan_gate=dbscan_gate,
        hdbscan_gate=hdbscan_gate,
        strategy_evals=all_evals,
        strategy_gates=all_gates,
        strategy_prompt_tokens=all_prompt_tokens,
        selection_table=selection_table,
        selected_strategy=selected,
        selection_reason=reason,
    )
