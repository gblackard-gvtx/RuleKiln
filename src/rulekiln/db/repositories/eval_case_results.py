"""Repository helpers for durable per-case evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import EvalCaseResultRecord


class EvalCaseResultUpsert(BaseModel):
    """Payload for idempotent per-case evaluation persistence."""

    job_id: str
    student_id: str
    strategy: str
    split: str
    case_id: str

    expected_json: dict[str, object] | str | None = None
    actual_json: dict[str, str | int | float | bool | None] | str | None = None
    raw_output: str | None = None
    assertion_scores: dict[str, float] = Field(default_factory=dict)

    passed: bool
    case_score: float

    malformed: bool = False
    invalid_label: bool = False

    error_type: str | None = None
    error_message: str | None = None


async def upsert_eval_case_result(
    session: AsyncSession,
    payload: EvalCaseResultUpsert,
) -> None:
    """Insert or update one case-level evaluation record and commit immediately."""
    existing = await session.execute(
        select(EvalCaseResultRecord).where(
            EvalCaseResultRecord.job_id == payload.job_id,
            EvalCaseResultRecord.student_id == payload.student_id,
            EvalCaseResultRecord.strategy == payload.strategy,
            EvalCaseResultRecord.split == payload.split,
            EvalCaseResultRecord.case_id == payload.case_id,
        )
    )
    row = existing.scalar_one_or_none()

    if row is None:
        row = EvalCaseResultRecord(
            job_id=payload.job_id,
            student_id=payload.student_id,
            strategy=payload.strategy,
            split=payload.split,
            case_id=payload.case_id,
            expected_json=payload.expected_json,
            actual_json=payload.actual_json,
            raw_output=payload.raw_output,
            assertion_scores=payload.assertion_scores,
            passed=payload.passed,
            case_score=payload.case_score,
            malformed=payload.malformed,
            invalid_label=payload.invalid_label,
            error_type=payload.error_type,
            error_message=payload.error_message,
        )
        session.add(row)
    else:
        row.expected_json = payload.expected_json
        row.actual_json = payload.actual_json
        row.raw_output = payload.raw_output
        row.assertion_scores = payload.assertion_scores
        row.passed = payload.passed
        row.case_score = payload.case_score
        row.malformed = payload.malformed
        row.invalid_label = payload.invalid_label
        row.error_type = payload.error_type
        row.error_message = payload.error_message

    await session.commit()


async def get_eval_case_results(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
) -> list[EvalCaseResultRecord]:
    """Load all persisted case results for one job/student/strategy/split."""
    result = await session.execute(
        select(EvalCaseResultRecord)
        .where(
            EvalCaseResultRecord.job_id == job_id,
            EvalCaseResultRecord.student_id == student_id,
            EvalCaseResultRecord.strategy == strategy,
            EvalCaseResultRecord.split == split,
        )
        .order_by(EvalCaseResultRecord.created_at.asc())
    )
    return list(result.scalars().all())


async def get_completed_eval_case_ids(
    session: AsyncSession,
    *,
    job_id: str,
    student_id: str,
    strategy: str,
    split: str,
) -> set[str]:
    """Return case IDs already persisted for one evaluation strategy run."""
    result = await session.execute(
        select(EvalCaseResultRecord.case_id).where(
            EvalCaseResultRecord.job_id == job_id,
            EvalCaseResultRecord.student_id == student_id,
            EvalCaseResultRecord.strategy == strategy,
            EvalCaseResultRecord.split == split,
        )
    )
    return set(result.scalars().all())
