"""Strategy selection with tie-break logic (C005)."""

from __future__ import annotations

from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.schemas.pipeline import EvalResult, QualityGateResult, StrategyComparison
from rulekiln.schemas.task_case import TaskMode


def _primary_score(result: EvalResult, task_mode: TaskMode) -> float:
    metric = get_primary_metric(task_mode)
    if metric == "macro_f1":
        return result.macro_f1 or 0.0
    return result.weighted_case_score or 0.0


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
    """Return (selected_strategy, reason) applying C005 tie-break rules."""
    dbscan_passes = dbscan_gate.passed if dbscan_gate else False
    hdbscan_passes = hdbscan_gate.passed if hdbscan_gate else False

    # Rule 7: if HDBSCAN fails and DBSCAN passes → select DBSCAN
    if hdbscan_eval is not None and dbscan_eval is not None:
        if not hdbscan_passes and dbscan_passes:
            return "dbscan", "HDBSCAN failed quality gates; DBSCAN passed."
        if hdbscan_passes and not dbscan_passes:
            return "hdbscan", "DBSCAN failed quality gates; HDBSCAN passed."
        if not hdbscan_passes and not dbscan_passes:
            # Neither passes; prefer higher primary score
            hs = _primary_score(hdbscan_eval, task_mode)
            ds = _primary_score(dbscan_eval, task_mode)
            if hs >= ds:
                return (
                    "hdbscan",
                    "Neither strategy passed gates; HDBSCAN has higher/equal primary metric.",
                )
            return "dbscan", "Neither strategy passed gates; DBSCAN has higher primary metric."

    # Both pass (or only one available)
    if hdbscan_eval is None and dbscan_eval is not None:
        return "dbscan", "Only DBSCAN available."
    if dbscan_eval is None and hdbscan_eval is not None:
        return "hdbscan", "Only HDBSCAN available."
    if hdbscan_eval is None and dbscan_eval is None:
        return "baseline", "No distilled strategy produced results; falling back to baseline."

    if hdbscan_eval is None or dbscan_eval is None:  # pragma: no cover
        return "baseline", "No distilled strategy produced results; falling back to baseline."
    hs = _primary_score(hdbscan_eval, task_mode)
    ds = _primary_score(dbscan_eval, task_mode)

    # Rule 2: prefer higher primary metric
    if abs(hs - ds) > 0.005:
        if hs > ds:
            return "hdbscan", f"HDBSCAN has higher primary metric ({hs:.4f} vs {ds:.4f})."
        return "dbscan", f"DBSCAN has higher primary metric ({ds:.4f} vs {hs:.4f})."

    # Rule 3: within 0.005 — prefer fewer golden failures
    hg = hdbscan_gate.golden_failures if hdbscan_gate else 0
    dg = dbscan_gate.golden_failures if dbscan_gate else 0
    if hg != dg:
        if hg < dg:
            return "hdbscan", "Tied primary metric; HDBSCAN has fewer golden failures."
        return "dbscan", "Tied primary metric; DBSCAN has fewer golden failures."

    # Rule 4: prefer lower malformed output rate
    hm = hdbscan_eval.malformed_output_rate
    dm = dbscan_eval.malformed_output_rate
    if abs(hm - dm) > 0.001:
        if hm < dm:
            return "hdbscan", "Tied; HDBSCAN has lower malformed output rate."
        return "dbscan", "Tied; DBSCAN has lower malformed output rate."

    # Rule 5: prefer lower prompt token count
    if hdbscan_token_count != dbscan_token_count:
        if hdbscan_token_count < dbscan_token_count:
            return "hdbscan", "Tied; HDBSCAN has lower token count."
        return "dbscan", "Tied; DBSCAN has lower token count."

    # Rule 6: default to HDBSCAN (production default)
    return "hdbscan", "All tie-breaks equal; defaulting to HDBSCAN (production default)."


def build_strategy_comparison(
    baseline_eval: EvalResult | None,
    dbscan_eval: EvalResult | None,
    hdbscan_eval: EvalResult | None,
    dbscan_gate: QualityGateResult | None,
    hdbscan_gate: QualityGateResult | None,
    task_mode: TaskMode,
    dbscan_token_count: int = 0,
    hdbscan_token_count: int = 0,
) -> StrategyComparison:
    selected, reason = select_strategy(
        task_mode=task_mode,
        dbscan_eval=dbscan_eval,
        hdbscan_eval=hdbscan_eval,
        dbscan_gate=dbscan_gate,
        hdbscan_gate=hdbscan_gate,
        baseline_eval=baseline_eval,
        dbscan_token_count=dbscan_token_count,
        hdbscan_token_count=hdbscan_token_count,
    )
    return StrategyComparison(
        baseline_eval=baseline_eval,
        dbscan_eval=dbscan_eval,
        hdbscan_eval=hdbscan_eval,
        dbscan_gate=dbscan_gate,
        hdbscan_gate=hdbscan_gate,
        selected_strategy=selected,
        selection_reason=reason,
    )
