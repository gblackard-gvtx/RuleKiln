"""Rule provenance builder: assembles per-rule attribution records from available evidence."""

from __future__ import annotations

from rulekiln.pipeline.failure_analysis import FailureAnalysisResult
from rulekiln.schemas.pipeline import (
    RuleAblationArtifact,
    RuleAblationRecord,
    RuleProvenanceArtifact,
    RuleProvenanceRecord,
    SynthesizedRuleSchema,
)


def build_provenance_records(
    *,
    job_id: str,
    strategy_id: str,
    selected_rules: list[SynthesizedRuleSchema],
    failure_analysis: FailureAnalysisResult | None = None,
    cluster_id_by_micro_rule: dict[str, str] | None = None,
    ablation_artifact: RuleAblationArtifact | None = None,
) -> RuleProvenanceArtifact:
    """Assemble provenance records for every selected rule.

    Always emits associative fields (source_case_ids, support counts, examples_fixed/broken).
    When ablation_artifact is supplied, adds causal fields and prefers them in classification.
    """
    ablation_by_rule_id: dict[str, RuleAblationRecord] = {}
    if ablation_artifact is not None:
        ablation_by_rule_id = {rec.rule_id: rec for rec in ablation_artifact.records}

    # Build fixed/broken case-id sets from failure analysis
    fixed_case_ids: set[str] = set()
    violated_rules_for_broken: dict[str, list[str]] = {}  # rule_id -> broken case_ids

    if failure_analysis is not None:
        fixed_case_ids = {str(entry.get("case_id", "")) for entry in failure_analysis.fixed}
        for sf in failure_analysis.structured_failures:
            if sf.failure_class == "broken":
                for rule_id in sf.violated_rule_ids:
                    violated_rules_for_broken.setdefault(rule_id, []).append(sf.case_id)

    records: list[RuleProvenanceRecord] = []
    for rule in selected_rules:
        # Associative: fixed cases overlap with rule's source_case_ids
        rule_source_ids = set(rule.source_case_ids)
        examples_fixed = sorted(fixed_case_ids & rule_source_ids)
        examples_broken = violated_rules_for_broken.get(rule.id, [])

        # Cluster lookup
        cluster_id: str | None = None
        if cluster_id_by_micro_rule:
            for micro_id in rule.source_micro_rule_ids:
                cid = cluster_id_by_micro_rule.get(micro_id)
                if cid is not None:
                    cluster_id = cid
                    break

        # Flags
        zero_validation_impact = not examples_fixed and not examples_broken
        regression_flag = bool(examples_broken)

        # Notes from conflict review
        notes: list[str] = []
        if rule.conflict_summary:
            notes.append(f"conflict_summary: {rule.conflict_summary}")

        # Causal enrichment from ablation
        ablation_rec = ablation_by_rule_id.get(rule.id)
        ablation_classification = None
        ablation_metric_delta: float | None = None
        ablation_changed_cases: int | None = None
        attribution_method: str = "associative"

        if ablation_rec is not None:
            ablation_classification = ablation_rec.classification
            ablation_metric_delta = ablation_rec.metric_delta_without_rule
            ablation_changed_cases = ablation_rec.changed_cases
            attribution_method = "causal"
            # Prefer causal regression signal over associative
            if ablation_rec.classification == "harmful":
                regression_flag = True

        records.append(
            RuleProvenanceRecord(
                rule_id=rule.id,
                topic=rule.topic,
                source_case_ids=sorted(rule_source_ids),
                cluster_id=cluster_id,
                support_count=rule.support_count,
                support_ratio=rule.support_ratio,
                examples_fixed=examples_fixed,
                examples_broken=sorted(examples_broken),
                attribution_method=attribution_method,  # type: ignore[arg-type]
                ablation_classification=ablation_classification,
                ablation_metric_delta=ablation_metric_delta,
                ablation_changed_cases=ablation_changed_cases,
                zero_validation_impact=zero_validation_impact,
                regression_flag=regression_flag,
                notes=notes,
            )
        )

    return RuleProvenanceArtifact(
        job_id=job_id,
        strategy_id=strategy_id,
        rules=records,
    )


def build_cluster_id_by_micro_rule(
    clusters: list[tuple[str, list[str]]],
) -> dict[str, str]:
    """Build micro_rule_id -> cluster_id mapping from (cluster_id, micro_rule_ids) pairs."""
    mapping: dict[str, str] = {}
    for cluster_id, micro_rule_ids in clusters:
        for micro_id in micro_rule_ids:
            mapping[micro_id] = cluster_id
    return mapping
