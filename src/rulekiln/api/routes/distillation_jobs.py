"""Distillation jobs API routes."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.api.validators.request_shape import (
    RequestValidationError,
    validate_distillation_request,
)
from rulekiln.config.settings import AppSettings, get_settings
from rulekiln.db.models import DistillationJob
from rulekiln.db.repositories.jobs import create_job, get_job
from rulekiln.db.session import get_db_session
from rulekiln.observability.logging import get_logger
from rulekiln.schemas.job import (
    CreateJobResponse,
    DistillationRequest,
    JobStatusResponse,
    JobUsageSummary,
)
from rulekiln.workers.dbos_runtime import require_dbos_available

router = APIRouter(prefix="/jobs", tags=["jobs"])

logger = get_logger(__name__)


@router.post("/", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_distillation_job(
    payload: DistillationRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> CreateJobResponse:
    try:
        validate_distillation_request(payload, settings)
    except RequestValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    try:
        require_dbos_available()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    job_id = str(uuid.uuid4())
    job = DistillationJob(
        id=job_id,
        task_id=payload.task.task_id,
        task_name=payload.task.task_name,
        task_mode=payload.task.task_mode,
        status="created",
        stage=None,
        queue_status="pending",
        max_attempts=settings.worker_max_attempts,
        request_json=payload.model_dump(mode="json"),
    )
    await create_job(session, job)
    logger.info(
        "job_created",
        job_id=job_id,
        task_id=payload.task.task_id,
        execution_backend=settings.execution_backend,
    )

    return CreateJobResponse(job_id=job_id, status="pending")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_distillation_job(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> JobStatusResponse:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    usage: JobUsageSummary | None = None
    if job.total_tokens > 0 or job.estimated_total_cost_usd is not None:
        usage = JobUsageSummary(
            total_input_tokens=job.total_input_tokens,
            total_output_tokens=job.total_output_tokens,
            total_tokens=job.total_tokens,
            estimated_total_cost_usd=float(job.estimated_total_cost_usd)
            if job.estimated_total_cost_usd is not None
            else None,
            teacher_cost_usd=float(job.teacher_cost_usd)
            if job.teacher_cost_usd is not None
            else None,
            student_cost_usd=float(job.student_cost_usd)
            if job.student_cost_usd is not None
            else None,
            embedding_cost_usd=float(job.embedding_cost_usd)
            if job.embedding_cost_usd is not None
            else None,
            judge_cost_usd=float(job.judge_cost_usd) if job.judge_cost_usd is not None else None,
        )

    mlflow_run_url: str | None = None
    if job.mlflow_run_id and settings.mlflow_ui_base_url:
        mlflow_run_url = (
            f"{settings.mlflow_ui_base_url.rstrip('/')}/#/experiments/1/runs/{job.mlflow_run_id}"
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        error_message=job.error_message,
        usage=usage,
        mlflow_run_id=job.mlflow_run_id,
        mlflow_run_url=mlflow_run_url,
    )
