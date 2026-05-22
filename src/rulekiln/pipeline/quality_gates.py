"""Quality gate checks (C006): metric delta, malformed output, golden failures,
regression rate, token budget."""

from __future__ import annotations

from rulekiln.config.settings import QualityGateDefaults
from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.schemas.pipeline import EvalResult, QualityGateResult
from rulekiln.schemas.task_case import RuleKilnCase, TaskMode


def _get_threshold(
    task_gates: dict[str, object],
    settings_defaults: QualityGateDefaults,
    key: str,
    hardcoded: object,
) -> object:
    """Resolve threshold: task > settings > hardcoded (C006)."""
    if key in task_gates:
        return task_gates[key]
    val = getattr(settings_defaults, key, hardcoded)
    return val if val is not None else hardcoded


def check_quality_gates(
    strategy: str,
    distilled_eval: EvalResult,
    baseline_eval: EvalResult | None,
    cases: list[RuleKilnCase],
    task_mode: TaskMode,
    task_gates: dict[str, object],
    settings_defaults: QualityGateDefaults,
    prompt_token_count: int = 0,
) -> QualityGateResult:
    violations: list[str] = []

    # Resolve thresholds
    min_metric_delta = float(_get_threshold(task_gates, settings_defaults, "min_metric_delta", 0.0))
    max_regression_rate = float(
        _get_threshold(task_gates, settings_defaults, "max_regression_rate", 0.10)
    )
    max_golden_failures = int(
        _get_threshold(task_gates, settings_defaults, "max_golden_failures", 0)
    )
    max_malformed = float(
        _get_threshold(task_gates, settings_defaults, "max_malformed_output_rate", 0.01)
    )
    max_tokens = int(_get_threshold(task_gates, settings_defaults, "max_prompt_tokens", 8000))

    primary_metric = get_primary_metric(task_mode)
    distilled_score = (
        distilled_eval.macro_f1
        if primary_metric == "macro_f1"
        else distilled_eval.weighted_case_score
    ) or 0.0

    # Gate 1: metric delta
    if baseline_eval is not None:
        baseline_score = (
            baseline_eval.macro_f1
            if primary_metric == "macro_f1"
            else baseline_eval.weighted_case_score
        ) or 0.0
        delta = distilled_score - baseline_score
        if delta < min_metric_delta:
            violations.append(f"metric_delta {delta:.4f} < min_metric_delta {min_metric_delta:.4f}")
    else:
        delta = None

    # Gate 2: malformed output rate
    if distilled_eval.malformed_output_rate > max_malformed:
        violations.append(
            f"malformed_output_rate {distilled_eval.malformed_output_rate:.4f}"
            f" > {max_malformed:.4f}"
        )

    # Gate 3: golden failures
    golden_ids = {c.id for c in cases if c.split == "golden"}
    golden_failures = sum(
        1 for r in distilled_eval.case_results if r.case_id in golden_ids and not r.passed
    )
    if golden_failures > max_golden_failures:
        violations.append(
            f"golden_failures {golden_failures} > max_golden_failures {max_golden_failures}"
        )

    # Gate 4: regression rate vs baseline
    regression_rate = 0.0
    if baseline_eval is not None:
        baseline_pass = {r.case_id for r in baseline_eval.case_results if r.passed}
        distilled_pass = {r.case_id for r in distilled_eval.case_results if r.passed}
        regressions = baseline_pass - distilled_pass
        total = len(baseline_pass)
        regression_rate = len(regressions) / total if total > 0 else 0.0
        if regression_rate > max_regression_rate:
            violations.append(
                f"regression_rate {regression_rate:.4f}"
                f" > max_regression_rate {max_regression_rate:.4f}"
            )

    # Gate 5: token budget
    if prompt_token_count > max_tokens:
        violations.append(f"prompt_tokens {prompt_token_count} > max_prompt_tokens {max_tokens}")

    return QualityGateResult(
        strategy=strategy,
        passed=len(violations) == 0,
        metric_delta=delta if baseline_eval is not None else None,
        regression_rate=regression_rate,
        golden_failures=golden_failures,
        malformed_output_rate=distilled_eval.malformed_output_rate,
        prompt_tokens=prompt_token_count,
        violations=violations,
    )
