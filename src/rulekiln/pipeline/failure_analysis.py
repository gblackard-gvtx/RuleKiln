"""Failure analysis: categorize cases as fixed, broken, or unchanged between baseline and distilled."""

from __future__ import annotations

import json

from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult


class FailureAnalysisResult:
    def __init__(self) -> None:
        self.fixed: list[dict[str, object]] = []
        self.broken: list[dict[str, object]] = []
        self.unchanged_passing: list[dict[str, object]] = []
        self.unchanged_failing: list[dict[str, object]] = []

    def to_jsonl(self, category: str) -> str:
        items = {
            "fixed": self.fixed,
            "broken": self.broken,
            "unchanged_passing": self.unchanged_passing,
            "unchanged_failing": self.unchanged_failing,
        }[category]
        return "\n".join(json.dumps(item) for item in items)


def analyze_failures(
    baseline_eval: EvalResult | None,
    distilled_eval: EvalResult,
) -> FailureAnalysisResult:
    """Compare baseline vs distilled per-case results and categorize changes."""
    result = FailureAnalysisResult()

    if baseline_eval is None:
        # No baseline: all distilled failures go to unchanged_failing
        for r in distilled_eval.case_results:
            entry = _case_entry(r, None)
            if r.passed:
                result.unchanged_passing.append(entry)
            else:
                result.unchanged_failing.append(entry)
        return result

    baseline_map: dict[str, CaseEvalResult] = {r.case_id: r for r in baseline_eval.case_results}
    distilled_map: dict[str, CaseEvalResult] = {r.case_id: r for r in distilled_eval.case_results}

    all_ids = set(baseline_map) | set(distilled_map)

    for case_id in all_ids:
        b = baseline_map.get(case_id)
        d = distilled_map.get(case_id)
        entry = _case_entry(d or b, b)  # type: ignore[arg-type]

        if b is None:
            # New case in distilled only
            if d and d.passed:
                result.unchanged_passing.append(entry)
            else:
                result.unchanged_failing.append(entry)
        elif d is None:
            # Case disappeared from distilled
            result.unchanged_failing.append(entry)
        elif not b.passed and d.passed:
            result.fixed.append(entry)
        elif b.passed and not d.passed:
            result.broken.append(entry)
        elif d.passed:
            result.unchanged_passing.append(entry)
        else:
            result.unchanged_failing.append(entry)

    return result


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
