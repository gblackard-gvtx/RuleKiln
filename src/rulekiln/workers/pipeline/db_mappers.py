"""DB model ↔ schema conversion helpers.

Pure functions with no side effects — safe to unit-test without a database.
"""

from __future__ import annotations

import json
import uuid

from rulekiln.db.models import Case, EvalCaseResultRecord, EvalRun, RuleCluster, SynthesizedRule
from rulekiln.db.repositories.eval_case_results import EvalCaseResultUpsert
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    OutcomeCondition,
    RuleClusterSchema,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase

_CASE_ID_DELIMITER = "::"


def _db_case_id(job_id: str, payload_case_id: str) -> str:
    return f"{job_id}{_CASE_ID_DELIMITER}{payload_case_id}"


def _payload_case_id_from_db_case_id(job_id: str, db_case_id: str) -> str:
    prefix = f"{job_id}{_CASE_ID_DELIMITER}"
    if db_case_id.startswith(prefix):
        return db_case_id[len(prefix) :]
    return db_case_id


def _to_db_case(job_id: str, case: RuleKilnCase) -> Case:
    expected_json: dict | None = None  # pyright: ignore[reportMissingTypeArgument]
    expected_text: str | None = None
    if isinstance(case.expected, dict):
        expected_json = case.expected
    elif case.expected is not None:
        expected_text = str(case.expected)
    return Case(
        id=_db_case_id(job_id, case.id),
        job_id=job_id,
        task_mode=case.task_mode,
        split=case.split,
        input_json=case.input,
        expected_json=expected_json,
        expected_text=expected_text,
        evaluation_json=case.evaluation.model_dump(),
        metadata_json=case.metadata,
        weight=case.weight,
    )


def _to_db_cluster(job_id: str, c: RuleClusterSchema) -> RuleCluster:
    return RuleCluster(
        id=str(uuid.uuid4()),
        job_id=job_id,
        strategy=c.strategy,
        topic=c.topic,
        algorithm=c.algorithm,
        rule_ids=c.rule_ids,
        cluster_metadata=c.cluster_metadata,
    )


def _synth_to_db(job_id: str, strategy: str, rule: SynthesizedRuleSchema) -> SynthesizedRule:
    return SynthesizedRule(
        id=str(uuid.uuid4()),
        job_id=job_id,
        strategy=strategy,
        rule_type=rule.rule_type,
        topic=rule.topic,
        applies_when=rule.applies_when,
        outcome_conditions={k: v.model_dump() for k, v in rule.outcome_conditions.items()},
        tie_breakers=rule.tie_breakers,
        priority=rule.priority,
        source_case_ids=rule.source_case_ids,
        source_micro_rule_ids=rule.source_micro_rule_ids,
        has_conflicts=rule.has_conflicts,
        conflict_summary=rule.conflict_summary,
        conflicting_micro_rule_ids=rule.conflicting_micro_rule_ids,
        support_count=rule.support_count,
        support_ratio=rule.support_ratio,
        golden_case_backed=rule.golden_case_backed,
        estimated_token_count=rule.estimated_token_count,
    )


def _db_synth_to_schema(r: SynthesizedRule) -> SynthesizedRuleSchema:
    oc = {k: OutcomeCondition.model_validate(v) for k, v in (r.outcome_conditions or {}).items()}
    return SynthesizedRuleSchema(
        id=r.id,
        rule_type=r.rule_type,
        topic=r.topic,
        applies_when=list(r.applies_when or []),
        outcome_conditions=oc,
        tie_breakers=list(r.tie_breakers or []),
        priority=r.priority,
        source_case_ids=list(r.source_case_ids or []),
        source_micro_rule_ids=list(r.source_micro_rule_ids or []),
        has_conflicts=r.has_conflicts,
        conflict_summary=r.conflict_summary,
        conflicting_micro_rule_ids=list(r.conflicting_micro_rule_ids or []),
        support_count=r.support_count,
        support_ratio=r.support_ratio,
        golden_case_backed=r.golden_case_backed,
        estimated_token_count=r.estimated_token_count,
    )


def _eval_to_db(job_id: str, prompt_version_id: str | None, result: EvalResult) -> EvalRun:
    return EvalRun(
        id=str(uuid.uuid4()),
        job_id=job_id,
        prompt_version_id=prompt_version_id,
        strategy=result.strategy,
        model=result.model,
        split=result.split,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        weighted_case_score=result.weighted_case_score,
        per_outcome_precision=result.per_outcome_precision,
        per_outcome_recall=result.per_outcome_recall,
        malformed_output_rate=result.malformed_output_rate,
        confusion_matrix={k: dict(v) for k, v in result.confusion_matrix.items()},
    )


def _eval_case_record_to_schema(row: EvalCaseResultRecord) -> CaseEvalResult:
    actual_output = row.actual_json
    if actual_output is None and row.raw_output is not None:
        actual_output = row.raw_output

    return CaseEvalResult(
        case_id=row.case_id,
        score=row.case_score,
        passed=row.passed,
        malformed=row.malformed,
        assertion_scores=dict(row.assertion_scores or {}),
        actual_output=actual_output,
        error=row.error_message,
    )


def _actual_output_to_raw_text(
    actual_output: dict[str, str | int | float | bool | None] | str | None,
) -> str | None:
    if actual_output is None:
        return None
    if isinstance(actual_output, str):
        return actual_output
    return json.dumps(actual_output, ensure_ascii=False)


def _is_invalid_label(
    case: RuleKilnCase,
    actual_output: dict[str, str | int | float | bool | None] | str | None,
) -> bool:
    if case.task_mode not in {"classification", "routing"}:
        return False

    if actual_output is None:
        return True

    if isinstance(actual_output, dict):
        label = actual_output.get("label")
        return not isinstance(label, str) or label.strip() == ""

    return actual_output.strip() == ""


def _build_eval_case_upsert_payload(
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
    case: RuleKilnCase,
    result: CaseEvalResult,
) -> EvalCaseResultUpsert:
    expected_json: dict[str, object] | str | None = (
        case.expected if isinstance(case.expected, (dict, str)) else None
    )

    actual_json = result.actual_output
    raw_output = _actual_output_to_raw_text(result.actual_output)
    invalid_label = _is_invalid_label(case, result.actual_output)

    return EvalCaseResultUpsert(
        job_id=job_id,
        student_id=student_id,
        strategy=strategy,
        split=split,
        case_id=result.case_id,
        expected_json=expected_json,
        actual_json=actual_json,
        raw_output=raw_output,
        assertion_scores=result.assertion_scores,
        passed=result.passed,
        case_score=result.score,
        malformed=result.malformed,
        invalid_label=invalid_label,
        error_type="MalformedOutput" if result.malformed else None,
        error_message=result.error,
    )
