"""Unit tests for rule provenance artifact builder."""

from __future__ import annotations

from rulekiln.pipeline.failure_analysis import FailureAnalysisResult
from rulekiln.pipeline.rule_provenance import (
    build_cluster_id_by_micro_rule,
    build_provenance_records,
)
from rulekiln.schemas.pipeline import (
    CaseEvaluationFailure,
    OutcomeCondition,
    RuleAblationArtifact,
    RuleAblationRecord,
    SynthesizedRuleSchema,
)


def _rule(
    rule_id: str,
    topic: str = "topic",
    support_count: int = 3,
    support_ratio: float = 0.5,
    source_case_ids: list[str] | None = None,
    source_micro_rule_ids: list[str] | None = None,
    conflict_summary: str | None = None,
) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=topic,
        applies_when=["some condition"],
        outcome_conditions={
            "out": OutcomeCondition(outcome="out", when=["cond"], confidence="high")
        },
        support_count=support_count,
        support_ratio=support_ratio,
        source_case_ids=source_case_ids or ["c1"],
        source_micro_rule_ids=source_micro_rule_ids or ["m1"],
        conflict_summary=conflict_summary,
    )


def test_basic_provenance_shape() -> None:
    rules = [_rule("r1"), _rule("r2")]
    artifact = build_provenance_records(
        job_id="job1",
        strategy_id="dbscan",
        selected_rules=rules,
    )
    assert artifact.job_id == "job1"
    assert artifact.strategy_id == "dbscan"
    assert len(artifact.rules) == 2
    assert artifact.rules[0].rule_id == "r1"
    assert artifact.schema_version == "rulekiln.rule_provenance.v1"


def test_source_case_ids_populated() -> None:
    rule = _rule("r1", source_case_ids=["c1", "c2", "c3"])
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
    )
    assert set(artifact.rules[0].source_case_ids) == {"c1", "c2", "c3"}


def test_zero_validation_impact_flag_set() -> None:
    rule = _rule("r1")
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        failure_analysis=None,
    )
    assert artifact.rules[0].zero_validation_impact is True


def test_examples_fixed_from_failure_analysis() -> None:
    rule = _rule("r1", source_case_ids=["c1", "c2", "c3"])
    fa = FailureAnalysisResult()
    fa.fixed = [{"case_id": "c1"}, {"case_id": "c99"}]
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        failure_analysis=fa,
    )
    rec = artifact.rules[0]
    # c1 is in both fixed cases and rule's source_case_ids
    assert "c1" in rec.examples_fixed
    # c99 is NOT in source_case_ids — should not appear
    assert "c99" not in rec.examples_fixed
    assert rec.zero_validation_impact is False


def test_examples_broken_from_structured_failures() -> None:
    rule = _rule("r1")
    fa = FailureAnalysisResult()
    fa.structured_failures = [
        CaseEvaluationFailure(
            case_id="c_broken",
            split="test",
            failure_class="broken",
            violated_rule_ids=["r1"],
        )
    ]
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        failure_analysis=fa,
    )
    rec = artifact.rules[0]
    assert "c_broken" in rec.examples_broken
    assert rec.regression_flag is True


def test_cluster_id_lookup() -> None:
    rule = _rule("r1", source_micro_rule_ids=["m1", "m2"])
    cluster_map = build_cluster_id_by_micro_rule([("cluster-abc", ["m1", "m3"])])
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        cluster_id_by_micro_rule=cluster_map,
    )
    assert artifact.rules[0].cluster_id == "cluster-abc"


def test_build_cluster_id_by_micro_rule_basic() -> None:
    mapping = build_cluster_id_by_micro_rule([("c1", ["m1", "m2"]), ("c2", ["m3"])])
    assert mapping["m1"] == "c1"
    assert mapping["m2"] == "c1"
    assert mapping["m3"] == "c2"
    assert "m99" not in mapping


def test_ablation_enrichment_causal() -> None:
    rule = _rule("r1")
    ablation = RuleAblationArtifact(
        job_id="j",
        strategy_id="dbscan",
        records=[
            RuleAblationRecord(
                rule_id="r1",
                topic="topic",
                classification="helpful",
                metric_delta_without_rule=-0.02,
                changed_cases=10,
                primary_metric="macro_f1",
            )
        ],
    )
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        ablation_artifact=ablation,
    )
    rec = artifact.rules[0]
    assert rec.ablation_classification == "helpful"
    assert rec.ablation_metric_delta == -0.02
    assert rec.attribution_method == "causal"


def test_ablation_harmful_sets_regression_flag() -> None:
    rule = _rule("r1")
    ablation = RuleAblationArtifact(
        job_id="j",
        strategy_id="dbscan",
        records=[
            RuleAblationRecord(
                rule_id="r1",
                topic="topic",
                classification="harmful",
                metric_delta_without_rule=0.03,
                changed_cases=8,
            )
        ],
    )
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
        ablation_artifact=ablation,
    )
    assert artifact.rules[0].regression_flag is True


def test_conflict_summary_in_notes() -> None:
    rule = _rule("r1", conflict_summary="Rule A conflicts with Rule B")
    artifact = build_provenance_records(
        job_id="j",
        strategy_id="dbscan",
        selected_rules=[rule],
    )
    assert any("Rule A conflicts" in n for n in artifact.rules[0].notes)
