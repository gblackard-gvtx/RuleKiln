"""Tests for per-strategy debug artifacts (item H / I.6, I.7).

These artifacts exist so a malformed or regressed case can be inspected directly
from disk (raw model output, parsed label, expected label, prompt) without
querying the database — see docs/investigations/contractnli_hdbscan_malformed_output.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from rulekiln.artifacts.writer import write_strategy_debug_artifacts
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import RuleKilnCase


def _case(case_id: str, label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="validation",
        task_mode="classification",
        input={"premise": case_id, "hypothesis": case_id},
        expected={"label": label},
    )


def _eval_result() -> EvalResult:
    return EvalResult(
        strategy="hdbscan",
        model="fake",
        split="validation",
        malformed_output_rate=0.25,
        confusion_matrix={"entailment": {"entailment": 2, "neutral": 1}},
        case_results=[
            CaseEvalResult(
                case_id="c1",
                score=1.0,
                passed=True,
                malformed=False,
                actual_output={"label": "entailment"},
                raw_content='{"label": "entailment"}',
            ),
            CaseEvalResult(
                case_id="c2",
                score=0.0,
                passed=False,
                malformed=False,
                actual_output={"label": "neutral"},
                raw_content='{"label": "neutral"}',
            ),
            CaseEvalResult(
                case_id="c3",
                score=0.0,
                passed=False,
                malformed=True,
                actual_output=None,
                raw_content="entailment_verbal_included",
                error="prediction_not_in_allowed_labels",
            ),
            CaseEvalResult(
                case_id="c4",
                score=1.0,
                passed=True,
                malformed=False,
                actual_output="contradiction",
                raw_content="contradiction.",
                recovered=True,
            ),
        ],
    )


def _baseline_eval() -> EvalResult:
    return EvalResult(
        strategy="baseline_scaffold",
        model="fake",
        split="validation",
        case_results=[
            CaseEvalResult(
                case_id="c1",
                score=1.0,
                passed=True,
                actual_output={"label": "entailment"},
                raw_content='{"label": "entailment"}',
            ),
            CaseEvalResult(
                case_id="c2",
                score=1.0,
                passed=True,
                actual_output={"label": "entailment"},
                raw_content='{"label": "entailment"}',
            ),
            CaseEvalResult(
                case_id="c3",
                score=1.0,
                passed=True,
                actual_output={"label": "neutral"},
                raw_content='{"label": "neutral"}',
            ),
            CaseEvalResult(
                case_id="c4",
                score=0.0,
                passed=False,
                actual_output={"label": "entailment"},
                raw_content='{"label": "entailment"}',
            ),
        ],
    )


def _cases_by_id() -> dict[str, RuleKilnCase]:
    return {
        "c1": _case("c1", "entailment"),
        "c2": _case("c2", "entailment"),
        "c3": _case("c3", "entailment"),
        "c4": _case("c4", "contradiction"),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_writes_all_eight_debug_artifact_types(tmp_path: Path) -> None:
    paths = write_strategy_debug_artifacts(
        tmp_path,
        "hdbscan",
        _eval_result(),
        _cases_by_id(),
        "compiled prompt text",
        "deadbeef",
        baseline_eval=_baseline_eval(),
    )
    names = {p.name for p in paths}
    assert names == {
        "hdbscan_raw_outputs.jsonl",
        "hdbscan_parsed_outputs.jsonl",
        "hdbscan_malformed_outputs.jsonl",
        "hdbscan_regressions_vs_baseline.jsonl",
        "hdbscan_prompt_preview.txt",
        "hdbscan_compiled_prompt.txt",
        "hdbscan_confusion_matrix.json",
        "hdbscan_classification_report.json",
    }
    for path in paths:
        assert path.exists()
        assert path.parent == tmp_path / "outputs" / "debug"


def test_raw_outputs_include_every_case_regardless_of_malformed(tmp_path: Path) -> None:
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), "prompt", "hash"
    )
    rows = _read_jsonl(tmp_path / "outputs" / "debug" / "hdbscan_raw_outputs.jsonl")
    assert {row["case_id"] for row in rows} == {"c1", "c2", "c3", "c4"}


def test_parsed_outputs_excludes_malformed_cases(tmp_path: Path) -> None:
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), "prompt", "hash"
    )
    rows = _read_jsonl(tmp_path / "outputs" / "debug" / "hdbscan_parsed_outputs.jsonl")
    assert {row["case_id"] for row in rows} == {"c1", "c2", "c4"}
    c4 = next(row for row in rows if row["case_id"] == "c4")
    assert c4["parsed_label"] == "contradiction"
    assert c4["expected_label"] == "contradiction"
    assert c4["recovered"] is True


def test_malformed_outputs_captures_raw_content_and_parse_error(tmp_path: Path) -> None:
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), "prompt", "hash"
    )
    rows = _read_jsonl(tmp_path / "outputs" / "debug" / "hdbscan_malformed_outputs.jsonl")
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c3"
    assert rows[0]["raw_content"] == "entailment_verbal_included"
    assert rows[0]["parse_error"] == "prediction_not_in_allowed_labels"
    assert rows[0]["expected_label"] == "entailment"


def test_regressions_jsonl_only_contains_baseline_passed_candidate_failed(
    tmp_path: Path,
) -> None:
    """Item I.7: the regressions file is exactly the paired-regression set — cases
    baseline got right that this strategy got wrong (either scored wrong or malformed).
    c1 (both pass) and c4 (baseline fails, candidate passes) must be excluded."""
    write_strategy_debug_artifacts(
        tmp_path,
        "hdbscan",
        _eval_result(),
        _cases_by_id(),
        "prompt",
        "hash",
        baseline_eval=_baseline_eval(),
    )
    rows = _read_jsonl(tmp_path / "outputs" / "debug" / "hdbscan_regressions_vs_baseline.jsonl")
    # baseline passed c1, c2, c3; candidate passed c1, c4 only → regressions are c2, c3
    assert {row["case_id"] for row in rows} == {"c2", "c3"}
    for row in rows:
        assert "baseline_raw_content" in row
        assert "candidate_raw_content" in row


def test_regressions_jsonl_empty_without_baseline_eval(tmp_path: Path) -> None:
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), "prompt", "hash", baseline_eval=None
    )
    rows = _read_jsonl(tmp_path / "outputs" / "debug" / "hdbscan_regressions_vs_baseline.jsonl")
    assert rows == []


def test_prompt_preview_is_truncated_and_compiled_prompt_has_hash_header(
    tmp_path: Path,
) -> None:
    long_prompt = "x" * 5000
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), long_prompt, "abc123"
    )
    debug_dir = tmp_path / "outputs" / "debug"
    preview = (debug_dir / "hdbscan_prompt_preview.txt").read_text(encoding="utf-8")
    assert len(preview) == 2000

    full = (debug_dir / "hdbscan_compiled_prompt.txt").read_text(encoding="utf-8")
    assert full.startswith("# prompt_hash: abc123")
    assert long_prompt in full


def test_confusion_matrix_and_classification_report_are_valid_json(tmp_path: Path) -> None:
    write_strategy_debug_artifacts(
        tmp_path, "hdbscan", _eval_result(), _cases_by_id(), "prompt", "hash"
    )
    debug_dir = tmp_path / "outputs" / "debug"
    confusion = json.loads((debug_dir / "hdbscan_confusion_matrix.json").read_text())
    assert confusion == {"entailment": {"entailment": 2, "neutral": 1}}

    report = json.loads((debug_dir / "hdbscan_classification_report.json").read_text())
    assert report == []
