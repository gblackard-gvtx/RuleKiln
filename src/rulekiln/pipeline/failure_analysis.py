"""Failure analysis: categorize cases as fixed, broken, or unchanged between baseline and distilled."""

from __future__ import annotations

import json

from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    CaseEvaluationFailure,
    EvalResult,
    SynthesizedRuleSchema,
)


class FailureAnalysisResult:
    def __init__(self) -> None:
        self.fixed: list[dict[str, object]] = []
        self.broken: list[dict[str, object]] = []
        self.unchanged_passing: list[dict[str, object]] = []
        self.unchanged_failing: list[dict[str, object]] = []
        self.structured_failures: list[CaseEvaluationFailure] = []

    def to_jsonl(self, category: str) -> str:
        items = {
            "fixed": self.fixed,
            "broken": self.broken,
            "unchanged_passing": self.unchanged_passing,
            "unchanged_failing": self.unchanged_failing,
        }[category]
        return "\n".join(json.dumps(item) for item in items)

    def violated_rule_summary(self) -> dict[str, dict[str, int]]:
        """Aggregate violation counts per rule ID across all failure classes."""
        summary: dict[str, dict[str, int]] = {}
        for failure in self.structured_failures:
            for rule_id in failure.violated_rule_ids:
                if rule_id not in summary:
                    summary[rule_id] = {
                        "violated_count": 0,
                        "broken_count": 0,
                        "unchanged_wrong_count": 0,
                    }
                summary[rule_id]["violated_count"] += 1
                if failure.failure_class == "broken":
                    summary[rule_id]["broken_count"] += 1
                elif failure.failure_class == "unchanged_wrong":
                    summary[rule_id]["unchanged_wrong_count"] += 1
        return summary


def analyze_failures(
    baseline_eval: EvalResult | None,
    distilled_eval: EvalResult,
    selected_rules: list[SynthesizedRuleSchema] | None = None,
) -> FailureAnalysisResult:
    """Compare baseline vs distilled per-case results and categorize changes.

    If selected_rules is provided, maps failed assertion paths to violated rule IDs
    and populates structured_failures with CaseEvaluationFailure records.
    """
    result = FailureAnalysisResult()

    # Build rule output-path → rule ID mapping for eval-to-rule mapping
    rule_output_path_index: dict[str, str] = {}
    if selected_rules:
        for rule in selected_rules:
            for oc in rule.outcome_conditions.values():
                # outcome_conditions may have paths embedded in the "when" conditions
                pass
            # Primary mapping: rule topic used as fallback key
            if rule.id:
                rule_output_path_index[rule.topic.lower()] = rule.id

    if baseline_eval is None:
        for r in distilled_eval.case_results:
            entry = _case_entry(r, None)
            if r.passed:
                result.unchanged_passing.append(entry)
            else:
                result.unchanged_failing.append(entry)
                _maybe_add_structured_failure(result, r, "unchanged_wrong", rule_output_path_index)
        return result

    baseline_map: dict[str, CaseEvalResult] = {r.case_id: r for r in baseline_eval.case_results}
    distilled_map: dict[str, CaseEvalResult] = {r.case_id: r for r in distilled_eval.case_results}

    all_ids = set(baseline_map) | set(distilled_map)

    for case_id in all_ids:
        b = baseline_map.get(case_id)
        d = distilled_map.get(case_id)
        entry = _case_entry(d or b, b)  # type: ignore[arg-type]

        if b is None:
            if d and d.passed:
                result.unchanged_passing.append(entry)
            else:
                result.unchanged_failing.append(entry)
                if d:
                    _maybe_add_structured_failure(result, d, "unchanged_wrong", rule_output_path_index)
        elif d is None:
            result.unchanged_failing.append(entry)
        elif not b.passed and d.passed:
            result.fixed.append(entry)
            _maybe_add_structured_failure(result, d, "fixed", rule_output_path_index)
        elif b.passed and not d.passed:
            result.broken.append(entry)
            _maybe_add_structured_failure(result, d, "broken", rule_output_path_index)
        elif d.passed:
            result.unchanged_passing.append(entry)
        else:
            result.unchanged_failing.append(entry)
            _maybe_add_structured_failure(result, d, "unchanged_wrong", rule_output_path_index)

    return result


def _maybe_add_structured_failure(
    result: FailureAnalysisResult,
    case_result: CaseEvalResult,
    failure_class: str,
    rule_output_path_index: dict[str, str],
) -> None:
    """Build a CaseEvaluationFailure by mapping failed assertion paths to rules."""
    failed_paths: list[str] = [
        path
        for path, score in case_result.assertion_scores.items()
        if score < 1.0
    ]
    failed_types: list[str] = []

    # Map failed paths to violated rules
    violated: list[str] = []
    for path in failed_paths:
        # Direct path match
        rule_id = rule_output_path_index.get(path)
        if rule_id and rule_id not in violated:
            violated.append(rule_id)

    from typing import Literal
    fc: Literal["fixed", "broken", "unchanged_wrong"] = failure_class  # type: ignore[assignment]

    result.structured_failures.append(
        CaseEvaluationFailure(
            case_id=case_result.case_id,
            split="",
            failure_class=fc,
            violated_rule_ids=violated,
            failed_assertion_paths=failed_paths,
            failed_assertion_types=failed_types,
        )
    )


def _case_entry(
    current: CaseEvalResult,
    baseline: CaseEvalResult | None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "case_id": current.case_id,
        "score": current.score,
        "passed": current.passed,
        "malformed": current.malformed,
    }
    if baseline is not None:
        entry["baseline_score"] = baseline.score
        entry["baseline_passed"] = baseline.passed
    return entry

