"""Split-leakage detection tests: train/validation/test boundary enforcement."""

import json
from pathlib import Path
from typing import Literal

from rulekiln.artifacts.canonical_hash import hash_text
from rulekiln.artifacts.writer import (
    write_cases_normalized,
    write_cluster_assignments,
    write_paired_comparison_artifacts,
    write_rule_provenance_json,
    write_strategy_eval,
)
from rulekiln.pipeline.split_leakage_validation import (
    read_split_leakage_validation,
    validate_split_leakage,
    write_split_leakage_validation,
)
from rulekiln.pipeline.statistics import PairedComparisonArtifacts
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    PairedComparisonExample,
    PairedComparisonSummary,
    RuleProvenanceArtifact,
    RuleProvenanceRecord,
)
from rulekiln.schemas.provenance import ClusterAssignmentRecord
from rulekiln.schemas.task_case import RuleKilnCase


def _case(
    case_id: str,
    split: Literal["train", "validation", "test", "golden"],
    document_hash: str | None = None,
) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split=split,
        task_mode="classification",
        input={"text": f"input for {case_id}"},
        expected={"label": "entailment"},
        metadata={"document_hash": document_hash} if document_hash else {},
    )


def _assignment(
    rule_id: str, source_case_id: str | None, source_split: str
) -> ClusterAssignmentRecord:
    return ClusterAssignmentRecord(
        rule_id=rule_id,
        source_case_id=source_case_id,
        source_split=source_split,
        cluster_id=0,
        is_noise=False,
        label="entailment",
        text_hash=hash_text(f"rule {rule_id}"),
        embedding_row_index=0,
    )


def _rule_provenance(
    rule_id: str, source_case_ids: list[str], source_split: str
) -> RuleProvenanceArtifact:
    return RuleProvenanceArtifact(
        job_id="job-1",
        strategy_id="hdbscan",
        rules=[
            RuleProvenanceRecord(
                rule_id=rule_id,
                topic="entailment",
                source_case_ids=source_case_ids,
                source_split=source_split,
            )
        ],
    )


def _eval_result(strategy: str, split: str, case_ids: list[str]) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="test-model",
        split=split,
        case_results=[
            CaseEvalResult(case_id=case_id, score=1.0, passed=True) for case_id in case_ids
        ],
    )


def _paired_comparison(case_ids: dict[str, list[str]]) -> PairedComparisonArtifacts:
    def _rows(
        names: list[str], change_class: Literal["fixed", "broken", "unchanged"]
    ) -> list[PairedComparisonExample]:
        return [
            PairedComparisonExample(
                case_id=case_id,
                change_class=change_class,
                baseline_strategy_id="baseline",
                candidate_strategy_id="hdbscan",
                expected_label="entailment",
                baseline_prediction="entailment",
                candidate_prediction="entailment",
                baseline_correct=True,
                candidate_correct=True,
                input_text="text",
            )
            for case_id in names
        ]

    fixed = _rows(case_ids.get("fixed", []), "fixed")
    broken = _rows(case_ids.get("broken", []), "broken")
    unchanged = _rows(case_ids.get("unchanged", []), "unchanged")
    return PairedComparisonArtifacts(
        fixed_examples=fixed,
        broken_examples=broken,
        unchanged_examples=unchanged,
        summary=PairedComparisonSummary(
            baseline_strategy_id="baseline",
            candidate_strategy_id="hdbscan",
            fixed_count=len(fixed),
            broken_count=len(broken),
            unchanged_correct_count=0,
            unchanged_wrong_count=0,
            total_cases=len(fixed) + len(broken) + len(unchanged),
            net_fix_rate=None,
            net_fix_rate_status="no_changed_outcomes",
            overall_net_fix_rate=0.0,
        ),
    )


def _write_refinement_provenance(
    root: Path, iteration: int, refined_using_split: str, validation_tuned: bool
) -> Path:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"rule_refinement_provenance_{iteration}.json"
    path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "refined_using_split": refined_using_split,
                "validation_tuned": validation_tuned,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_clean_run_has_no_findings(tmp_path: Path) -> None:
    write_cases_normalized(
        tmp_path,
        [_case("c1", "train"), _case("c2", "validation"), _case("c3", "test")],
    )
    write_cluster_assignments(tmp_path, "hdbscan", [_assignment("r1", "c1", "train")])
    write_rule_provenance_json(tmp_path, _rule_provenance("r1", ["c1"], "train"))
    write_strategy_eval(tmp_path, "hdbscan", _eval_result("hdbscan", "validation", ["c2"]))
    write_paired_comparison_artifacts(
        tmp_path / "outputs" / "paired_comparison", _paired_comparison({"fixed": ["c2"]})
    )
    _write_refinement_provenance(tmp_path, 1, "validation", True)

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []
    assert not report.has_errors
    assert not report.has_warnings


def test_train_stage_nontrain_case_from_cluster_assignment(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    write_cluster_assignments(tmp_path, "hdbscan", [_assignment("r1", "c1", "validation")])

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TRAIN_STAGE_NONTRAIN_CASE" in codes
    assert report.has_errors


def test_train_stage_nontrain_case_from_rule_provenance(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    write_rule_provenance_json(tmp_path, _rule_provenance("r1", ["c1"], "validation"))

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TRAIN_STAGE_NONTRAIN_CASE" in codes
    assert report.has_errors


def test_split_provenance_mismatch_on_cluster_assignment(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    write_cluster_assignments(tmp_path, "hdbscan", [_assignment("r1", "c1", "train")])

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "SPLIT_PROVENANCE_MISMATCH" in codes
    assert report.has_errors


def test_split_provenance_mismatch_on_rule_provenance(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    write_rule_provenance_json(tmp_path, _rule_provenance("r1", ["c1"], "train"))

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "SPLIT_PROVENANCE_MISMATCH" in codes
    assert report.has_errors


def test_test_case_leakage_via_cluster_assignment(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "test")])
    write_cluster_assignments(tmp_path, "hdbscan", [_assignment("r1", "c1", "test")])

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TEST_CASE_LEAKAGE" in codes
    assert report.has_errors


def test_test_case_leakage_via_rule_provenance(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "test")])
    write_rule_provenance_json(tmp_path, _rule_provenance("r1", ["c1"], "test"))

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TEST_CASE_LEAKAGE" in codes


def test_test_case_leakage_via_eval_result(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "test")])
    write_strategy_eval(tmp_path, "hdbscan", _eval_result("hdbscan", "test", ["c1"]))

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TEST_CASE_LEAKAGE" in codes


def test_test_case_leakage_via_paired_comparison(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "test")])
    write_paired_comparison_artifacts(
        tmp_path / "outputs" / "paired_comparison", _paired_comparison({"fixed": ["c1"]})
    )

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "TEST_CASE_LEAKAGE" in codes


def test_validation_eval_nonvalidation_case(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "train")])
    write_strategy_eval(tmp_path, "hdbscan", _eval_result("hdbscan", "validation", ["c1"]))

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "VALIDATION_EVAL_NONVALIDATION_CASE" in codes
    assert report.has_errors


def test_validation_refinement_not_marked(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    _write_refinement_provenance(tmp_path, 1, "validation", validation_tuned=False)

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "VALIDATION_REFINEMENT_NOT_MARKED" in codes
    assert report.has_errors


def test_validation_refinement_correctly_marked_is_clean(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "train")])
    _write_refinement_provenance(tmp_path, 1, "train", validation_tuned=False)

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []


def test_source_document_split_overlap_is_warning_not_error(tmp_path: Path) -> None:
    write_cases_normalized(
        tmp_path,
        [
            _case("c1", "train", document_hash="doc-1"),
            _case("c2", "validation", document_hash="doc-1"),
        ],
    )

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    codes = {f.code for f in report.findings}
    assert "SOURCE_DOCUMENT_SPLIT_OVERLAP" in codes
    finding = next(f for f in report.findings if f.code == "SOURCE_DOCUMENT_SPLIT_OVERLAP")
    assert finding.severity == "warning"
    assert not report.has_errors
    assert report.has_warnings


def test_same_split_document_overlap_is_not_flagged(tmp_path: Path) -> None:
    write_cases_normalized(
        tmp_path,
        [
            _case("c1", "train", document_hash="doc-1"),
            _case("c2", "train", document_hash="doc-1"),
        ],
    )

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []


def test_legacy_run_missing_all_artifacts_is_tolerated(tmp_path: Path) -> None:
    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []
    assert report.job_id == "job-1"


def test_legacy_malformed_eval_artifact_is_skipped_not_raised(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "train")])
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "hdbscan_eval.json").write_text("not valid json", encoding="utf-8")

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []


def test_legacy_malformed_refinement_provenance_is_skipped_not_raised(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "train")])
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "rule_refinement_provenance_1.json").write_text("{not json", encoding="utf-8")

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []


def test_legacy_rule_provenance_without_source_split_defaults_to_train(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "train")])
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    legacy_payload = {
        "schema_version": "rulekiln.rule_provenance.v1",
        "job_id": "job-1",
        "strategy_id": "hdbscan",
        "rules": [
            {
                "rule_id": "r1",
                "topic": "entailment",
                "source_case_ids": ["c1"],
                # no source_split key: legacy artifact predating the field
            }
        ],
    }
    (outputs / "rule_provenance.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)

    assert report.findings == []


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    write_cases_normalized(tmp_path, [_case("c1", "validation")])
    write_cluster_assignments(tmp_path, "hdbscan", [_assignment("r1", "c1", "validation")])

    report = validate_split_leakage(job_id="job-1", artifact_root=tmp_path)
    written_path = write_split_leakage_validation(tmp_path, report)

    assert written_path == tmp_path / "metadata" / "split_leakage_validation.json"
    loaded = read_split_leakage_validation(tmp_path)
    assert loaded is not None
    assert loaded.job_id == report.job_id
    assert [f.code for f in loaded.findings] == [f.code for f in report.findings]


def test_read_split_leakage_validation_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_split_leakage_validation(tmp_path) is None
