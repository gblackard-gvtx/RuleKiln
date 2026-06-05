"""Failure analysis: categorize cases as fixed, broken, or unchanged between
baseline and distilled."""

from __future__ import annotations

import json
import re

from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    CaseEvaluationFailure,
    EvalResult,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase

# Sentinel rule ID for failures that cannot be attributed to any rule.
UNATTRIBUTED_RULE_ID = "__unattributed__"

_ASSERTION_KEY_RE = re.compile(r"^assertion_(\d+)$")


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
        """Aggregate violation and fix counts per rule ID across all failure classes."""
        summary: dict[str, dict[str, int]] = {}
        for failure in self.structured_failures:
            if failure.failure_class == "fixed":
                for rule_id in failure.matched_rule_ids:
                    entry = summary.setdefault(
                        rule_id,
                        {
                            "violated_count": 0,
                            "broken_count": 0,
                            "unchanged_wrong_count": 0,
                            "fixed_count": 0,
                        },
                    )
                    entry["fixed_count"] += 1
            for rule_id in failure.violated_rule_ids:
                entry = summary.setdefault(
                    rule_id,
                    {
                        "violated_count": 0,
                        "broken_count": 0,
                        "unchanged_wrong_count": 0,
                        "fixed_count": 0,
                    },
                )
                entry["violated_count"] += 1
                if failure.failure_class == "broken":
                    entry["broken_count"] += 1
                elif failure.failure_class == "unchanged_wrong":
                    entry["unchanged_wrong_count"] += 1
        return summary

    def build_utility_signals(self) -> dict[str, tuple[int, int]]:
        """Build rule_id -> (fixed_count, broken_count) for use with prune_rules.

        Excludes the UNATTRIBUTED_RULE_ID sentinel.
        """
        summary = self.violated_rule_summary()
        return {
            rule_id: (counts.get("fixed_count", 0), counts.get("broken_count", 0))
            for rule_id, counts in summary.items()
            if rule_id != UNATTRIBUTED_RULE_ID
        }

    def unattributed_fraction(self) -> float:
        """Fraction of non-fixed failures with no real rule attribution."""
        non_fixed = [
            f for f in self.structured_failures if f.failure_class in ("broken", "unchanged_wrong")
        ]
        if not non_fixed:
            return 0.0
        unattributed = sum(
            1
            for f in non_fixed
            if not f.violated_rule_ids or f.violated_rule_ids == [UNATTRIBUTED_RULE_ID]
        )
        return unattributed / len(non_fixed)


def _build_outcome_to_rule_ids(
    rules: list[SynthesizedRuleSchema],
) -> dict[str, list[str]]:
    """Build outcome_label -> [rule_id, ...] from rule.outcome_conditions."""
    index: dict[str, list[str]] = {}
    for rule in rules:
        if not rule.id:
            continue
        for oc in rule.outcome_conditions.values():
            label = oc.outcome
            bucket = index.setdefault(label, [])
            if rule.id not in bucket:
                bucket.append(rule.id)
    return index


def analyze_failures(
    baseline_eval: EvalResult | None,
    distilled_eval: EvalResult,
    selected_rules: list[SynthesizedRuleSchema] | None = None,
    cases: list[RuleKilnCase] | None = None,
) -> FailureAnalysisResult:
    """Compare baseline vs distilled per-case results and categorize changes.

    If selected_rules is provided, maps failed assertion paths to violated rule IDs
    and populates structured_failures with CaseEvaluationFailure records.

    If cases is also provided, uses assertion definitions (type, expected value) to
    populate failed_assertion_types and perform outcome-based rule attribution via
    outcome_conditions. Without cases, only the raw assertion key paths are recorded.
    """
    result = FailureAnalysisResult()

    outcome_to_rule_ids: dict[str, list[str]] = {}
    if selected_rules:
        outcome_to_rule_ids = _build_outcome_to_rule_ids(selected_rules)

    case_map: dict[str, RuleKilnCase] = {}
    if cases:
        case_map = {c.id: c for c in cases}

    want_structured = selected_rules is not None

    if baseline_eval is None:
        for r in distilled_eval.case_results:
            entry = _case_entry(r, None)
            if r.passed:
                result.unchanged_passing.append(entry)
                if want_structured:
                    _add_structured_failure(
                        result, r, "unchanged_correct", outcome_to_rule_ids, case_map
                    )
            else:
                result.unchanged_failing.append(entry)
                if want_structured:
                    _add_structured_failure(
                        result, r, "unchanged_wrong", outcome_to_rule_ids, case_map
                    )
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
                if want_structured:
                    _add_structured_failure(
                        result, d, "unchanged_correct", outcome_to_rule_ids, case_map
                    )
            else:
                result.unchanged_failing.append(entry)
                if d and want_structured:
                    _add_structured_failure(
                        result, d, "unchanged_wrong", outcome_to_rule_ids, case_map
                    )
        elif d is None:
            result.unchanged_failing.append(entry)
        elif not b.passed and d.passed:
            result.fixed.append(entry)
            if want_structured:
                _add_structured_failure(result, d, "fixed", outcome_to_rule_ids, case_map)
        elif b.passed and not d.passed:
            result.broken.append(entry)
            if want_structured:
                _add_structured_failure(result, d, "broken", outcome_to_rule_ids, case_map)
        elif d.passed:
            result.unchanged_passing.append(entry)
            if want_structured:
                _add_structured_failure(
                    result, d, "unchanged_correct", outcome_to_rule_ids, case_map
                )
        else:
            result.unchanged_failing.append(entry)
            if want_structured:
                _add_structured_failure(result, d, "unchanged_wrong", outcome_to_rule_ids, case_map)

    return result


def _add_structured_failure(
    result: FailureAnalysisResult,
    case_result: CaseEvalResult,
    failure_class: str,
    outcome_to_rule_ids: dict[str, list[str]],
    case_map: dict[str, RuleKilnCase],
) -> None:
    """Build a CaseEvaluationFailure by mapping failed assertion paths to rules.

    For 'fixed' class: populates matched_rule_ids with rules governing expected outcomes.
    For 'broken'/'unchanged_wrong': populates violated_rule_ids and failed_assertion_types.
    Adds UNATTRIBUTED_RULE_ID sentinel when a non-fixed failure has no rule match.
    """
    from typing import Literal

    fc: Literal["fixed", "broken", "unchanged_wrong", "unchanged_correct"] = failure_class  # type: ignore[assignment]
    case = case_map.get(case_result.case_id)

    if failure_class in ("fixed", "unchanged_correct"):
        matched: list[str] = []
        if case is not None:
            for assertion in case.evaluation.assertions:
                label = str(assertion.value) if assertion.value is not None else ""
                for rule_id in outcome_to_rule_ids.get(label, []):
                    if rule_id not in matched:
                        matched.append(rule_id)
        result.structured_failures.append(
            CaseEvaluationFailure(
                case_id=case_result.case_id,
                split="",
                failure_class=fc,
                matched_rule_ids=matched,
                violated_rule_ids=[],
                failed_assertion_paths=[],
                failed_assertion_types=[],
            )
        )
        return

    failed_paths: list[str] = [
        path for path, score in case_result.assertion_scores.items() if score < 1.0
    ]
    failed_types: list[str] = []
    violated: list[str] = []

    for path in failed_paths:
        m = _ASSERTION_KEY_RE.match(path)
        if m and case is not None:
            idx = int(m.group(1))
            assertions = case.evaluation.assertions
            if idx < len(assertions):
                assertion = assertions[idx]
                atype = assertion.type
                if atype not in failed_types:
                    failed_types.append(atype)
                label = str(assertion.value) if assertion.value is not None else ""
                for rule_id in outcome_to_rule_ids.get(label, []):
                    if rule_id not in violated:
                        violated.append(rule_id)

    if not violated and failed_paths:
        violated = [UNATTRIBUTED_RULE_ID]

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
