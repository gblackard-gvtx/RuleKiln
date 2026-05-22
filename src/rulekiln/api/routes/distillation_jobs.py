"""Distillation jobs API routes."""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
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
from rulekiln.schemas.job import CreateJobResponse, DistillationRequest, JobStatusResponse
from rulekiln.workers.distillation_worker import run_distillation_pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])

logger = get_logger(__name__)


@router.post("/", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_distillation_job(
    payload: DistillationRequest,
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> CreateJobResponse:
    try:
        validate_distillation_request(payload, settings)
    except RequestValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    job_id = str(uuid.uuid4())
    queue_status = "pending" if settings.execution_backend == "postgres_queue" else "created"
    job = DistillationJob(
        id=job_id,
        task_id=payload.task.task_id,
        task_name=payload.task.task_name,
        task_mode=payload.task.task_mode,
        status="created",
        stage=None,
        queue_status=queue_status,
        request_json=payload.model_dump(mode="json"),
    )
    await create_job(session, job)
    logger.info(
        "job_created",
        job_id=job_id,
        task_id=payload.task.task_id,
        execution_backend=settings.execution_backend,
    )

    if settings.execution_backend == "background_tasks":
        background_tasks.add_task(run_distillation_pipeline, job_id, payload)
        return CreateJobResponse(job_id=job_id, status="created")

    # postgres_queue: job is persisted with queue_status='pending'; worker picks it up
    return CreateJobResponse(job_id=job_id, status="pending")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_distillation_job(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> JobStatusResponse:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        error_message=job.error_message,
    )
