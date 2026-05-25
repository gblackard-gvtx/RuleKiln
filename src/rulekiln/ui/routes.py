"""Operator UI routes — all phases (4-12)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse, RedirectResponse, Response

from rulekiln.api.validators.request_shape import (
    RequestValidationError,
    validate_distillation_request,
)
from rulekiln.config.settings import AppSettings, get_settings
from rulekiln.db.models import DistillationJob
from rulekiln.db.repositories.jobs import (
    create_job,
    get_eval_runs_for_job,
    get_job,
    get_selected_prompt_version,
    get_synthesized_rules_for_job,
    list_recent_jobs,
    update_job_status,
)
from rulekiln.db.session import get_db_session
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask
from rulekiln.ui.forms import NewJobForm
from rulekiln.ui.view_models import (
    ArtifactFileView,
    ArtifactsView,
    JobDetailView,
    JobListItemView,
    PreviewView,
    ProviderRouteView,
    ResultsSummaryView,
)
from rulekiln.workers.distillation_worker import run_distillation_pipeline

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"])
logger = get_logger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml", ".jsonl"})
_CONTENT_TYPE_MAP: dict[str, str] = {
    ".md": "text/markdown",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".txt": "text/plain",
}
_KNOWN_ARTIFACT_PATTERNS: list[str] = [
    "task.yaml",
    "cases.normalized.jsonl",
    "outputs/distilled_prompt_dbscan.md",
    "outputs/distilled_prompt_hdbscan.md",
    "outputs/selected_distilled_prompt.md",
    "outputs/rules_dbscan.jsonl",
    "outputs/rules_hdbscan.jsonl",
    "outputs/eval_report.json",
    "outputs/strategy_comparison.json",
    "outputs/failures_fixed.jsonl",
    "outputs/failures_broken.jsonl",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_selected_strategy(session: AsyncSession, job: DistillationJob) -> str | None:
    """Derive selected strategy from the selected PromptVersion, falling back to the job column."""
    pv = await get_selected_prompt_version(session, job.id)
    if pv is not None:
        return pv.strategy
    return job.selected_strategy


def _safe_artifact_path(artifact_root: str, job_id: str, path_param: str) -> Path:
    """Resolve an artifact path; raise 400 on traversal or absolute paths."""
    if path_param.startswith("/") or path_param.startswith("\\"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Absolute paths are not permitted.",
        )
    if ".." in Path(path_param).parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not permitted.",
        )
    base_dir = (Path(artifact_root) / job_id).resolve()
    resolved = (base_dir / path_param).resolve()
    if not resolved.is_relative_to(base_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not permitted.",
        )
    return resolved


def _ext_ok(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


def _resolve_primary_metric(job: DistillationJob) -> str:
    """Resolve primary metric using task config first, then request override, then mode default."""
    req_json = job.request_json
    if isinstance(req_json, dict):
        task_obj = req_json.get("task")
        if isinstance(task_obj, dict):
            eval_obj = task_obj.get("evaluation")
            if isinstance(eval_obj, dict):
                metric = eval_obj.get("primary_metric")
                if isinstance(metric, str) and metric.strip():
                    return metric.strip()

        metric_override = req_json.get("metric")
        if isinstance(metric_override, str) and metric_override.strip():
            return metric_override.strip()

    return get_primary_metric(job.task_mode)


def _score_for_metric(run: object, primary_metric: str) -> float | None:
    """Map a primary metric name to the corresponding EvalRun value."""
    metric = primary_metric.strip().lower()
    if metric == "macro_f1":
        return getattr(run, "macro_f1", None)
    if metric == "accuracy":
        return getattr(run, "accuracy", None)
    return getattr(run, "weighted_case_score", None)


# ── Phase 4: Job list ─────────────────────────────────────────────────────────


@router.get("/", include_in_schema=False)
async def ui_root() -> RedirectResponse:
    return RedirectResponse(url="/ui/jobs", status_code=status.HTTP_302_FOUND)


@router.get("/jobs")
async def job_list(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    jobs = await list_recent_jobs(session, limit=50)
    items: list[JobListItemView] = [
        JobListItemView(
            job_id=j.id,
            task_name=j.task_name,
            task_mode=j.task_mode,
            status=j.status,
            stage=j.stage,
            selected_strategy=j.selected_strategy,
            primary_metric_delta=None,
            created_at=j.created_at,
            detail_url=f"/ui/jobs/{j.id}",
        )
        for j in jobs
    ]
    return templates.TemplateResponse(request, "jobs/index.html", {"jobs": items})


# ── Phase 5: New job form ──────────────────────────────────────────────────────


@router.get("/jobs/new")
async def new_job_form(
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    profile_names = sorted(settings.provider_profiles.keys())
    return templates.TemplateResponse(request, "jobs/new.html", {"profile_names": profile_names})


# ── Phase 6: Preview / validate ───────────────────────────────────────────────


@router.post("/jobs/preview")
async def preview_job(
    request: Request,
    form: Annotated[NewJobForm, Depends(NewJobForm)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    errors: list[str] = []
    warnings: list[str] = []

    # ── Extension whitelist ──
    if not _ext_ok(form.task_file.filename):
        errors.append(
            f"task_file: extension "
            f"'{Path(form.task_file.filename or '').suffix or '(none)'}' is not allowed. "
            "Use .yaml or .yml."
        )
    if not _ext_ok(form.cases_file.filename):
        errors.append(
            f"cases_file: extension "
            f"'{Path(form.cases_file.filename or '').suffix or '(none)'}' is not allowed. "
            "Use .jsonl."
        )

    if errors:
        return templates.TemplateResponse(
            request,
            "jobs/preview.html",
            {"preview": None, "errors": errors, "warnings": warnings, "draft_job_id": None},
            status_code=422,
        )

    task_bytes = await form.task_file.read()
    cases_bytes = await form.cases_file.read()

    if len(task_bytes) > settings.max_upload_size_bytes:
        errors.append(
            f"task_file exceeds maximum allowed size ({settings.max_upload_size_bytes} bytes)."
        )
    if len(cases_bytes) > settings.max_upload_size_bytes:
        errors.append(
            f"cases_file exceeds maximum allowed size ({settings.max_upload_size_bytes} bytes)."
        )
    if errors:
        return templates.TemplateResponse(
            request,
            "jobs/preview.html",
            {"preview": None, "errors": errors, "warnings": warnings, "draft_job_id": None},
            status_code=422,
        )

    # ── Parse task YAML ──
    task: RuleKilnTask | None = None
    try:
        raw_task = yaml.safe_load(task_bytes.decode("utf-8"))
        task = RuleKilnTask.model_validate(raw_task)
    except Exception as exc:
        errors.append(f"task.yaml parse error: {exc}")

    # ── Parse cases JSONL ──
    cases: list[RuleKilnCase] = []
    for line_no, raw_line in enumerate(cases_bytes.decode("utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            cases.append(RuleKilnCase.model_validate(json.loads(stripped)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"cases.jsonl line {line_no}: {exc}")
            if len(errors) >= 10:
                errors.append("Too many parse errors — stopping early.")
                break

    if not cases and not errors:
        errors.append("cases.jsonl contains no valid cases.")

    if errors or task is None:
        return templates.TemplateResponse(
            request,
            "jobs/preview.html",
            {"preview": None, "errors": errors, "warnings": warnings, "draft_job_id": None},
            status_code=422,
        )

    # ── Build DistillationRequest & cross-validate ──
    judge: ModelRoute | None = None
    if form.judge_profile and form.judge_model:
        judge = ModelRoute(provider_profile=form.judge_profile, model=form.judge_model)

    distillation_request = DistillationRequest(
        task=task,
        cases=cases,
        teacher=ModelRoute(provider_profile=form.teacher_profile, model=form.teacher_model),
        student=ModelRoute(provider_profile=form.student_profile, model=form.student_model),
        embedding=ModelRoute(provider_profile=form.embedding_profile, model=form.embedding_model),
        judge=judge,
        baseline_prompt=form.baseline_prompt or None,
    )
    try:
        validate_distillation_request(distillation_request, settings)
    except (RequestValidationError, ValidationError) as exc:
        errors.append(str(exc))

    # ── Count splits and detect evaluation methods ──
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0, "golden": 0}
    eval_method_set: set[str] = set()
    for case in cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
        for assertion in case.evaluation.assertions:
            eval_method_set.add(assertion.type)

    # ── Provider routes for display ──
    route_specs: list[tuple[str, str, str]] = [
        ("teacher", form.teacher_profile, form.teacher_model),
        ("student", form.student_profile, form.student_model),
        ("embedding", form.embedding_profile, form.embedding_model),
    ]
    if form.judge_profile and form.judge_model:
        route_specs.append(("judge", form.judge_profile, form.judge_model))

    provider_routes: list[ProviderRouteView] = []
    for _role, profile_name, model_id in route_specs:
        profile = settings.provider_profiles.get(profile_name)
        provider_routes.append(
            ProviderRouteView(
                profile_name=profile_name,
                model_id=model_id,
                supports_chat=profile.supports_chat if profile else False,
                supports_embeddings=profile.supports_embeddings if profile else False,
            )
        )

    train_count = split_counts.get("train", 0)
    validation_count = split_counts.get("validation", 0)
    test_count = split_counts.get("test", 0)
    golden_count = split_counts.get("golden", 0)

    preview = PreviewView(
        task_id=task.task_id,
        task_name=task.task_name,
        task_mode=task.task_mode,
        case_count=len(cases),
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        golden_count=golden_count,
        evaluation_methods=sorted(eval_method_set),
        output_schema_present=bool(task.output_schema),
        provider_routes=provider_routes,
        estimated_teacher_calls=train_count,
        estimated_student_eval_calls=(validation_count + test_count) * 2,
        estimated_embedding_calls=train_count * 5,
        warnings=warnings,
        errors=errors,
    )

    if errors:
        return templates.TemplateResponse(
            request,
            "jobs/preview.html",
            {"preview": preview, "errors": errors, "warnings": warnings, "draft_job_id": None},
            status_code=422,
        )

    # ── Persist draft ──
    draft_id = str(uuid.uuid4())
    draft = DistillationJob(
        id=draft_id,
        task_id=task.task_id,
        task_name=task.task_name,
        task_mode=task.task_mode,
        status="draft",
        stage=None,
        request_json=distillation_request.model_dump(mode="json"),
    )
    await create_job(session, draft)

    return templates.TemplateResponse(
        request,
        "jobs/preview.html",
        {"preview": preview, "errors": [], "warnings": warnings, "draft_job_id": draft_id},
    )


# ── Phase 7: Create job from UI ───────────────────────────────────────────────


@router.post("/jobs")
async def create_job_from_ui(
    request: Request,
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
    draft_job_id: Annotated[str, Form()],
) -> RedirectResponse:
    job = await get_job(session, draft_job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft job not found.",
        )
    if job.status != "draft":
        logger.info(
            "ui_job_already_submitted",
            job_id=job.id,
            task_id=job.task_id,
            status=job.status,
        )
        # Treat duplicate submits as idempotent and send the user to the job page.
        return RedirectResponse(
            url=f"/ui/jobs/{job.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    try:
        payload = DistillationRequest.model_validate(job.request_json)
        validate_distillation_request(payload, settings)
    except (RequestValidationError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {exc}",
        ) from exc

    await update_job_status(session, job.id, status="created")
    background_tasks.add_task(run_distillation_pipeline, job.id, payload)
    logger.info("ui_job_submitted", job_id=job.id, task_id=job.task_id)

    return RedirectResponse(
        url=f"/ui/jobs/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ── Phase 8: Job detail & HTMX polling ───────────────────────────────────────


@router.get("/jobs/{job_id}")
async def job_detail(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    selected_strategy = await _get_selected_strategy(session, job)

    mlflow_run_url: str | None = None
    if job.mlflow_run_id and settings.mlflow_ui_base_url:
        mlflow_run_url = (
            f"{settings.mlflow_ui_base_url.rstrip('/')}/#/experiments/1/runs/{job.mlflow_run_id}"
        )

    detail = JobDetailView(
        job_id=job.id,
        task_name=job.task_name,
        task_mode=job.task_mode,
        status=job.status,
        stage=job.stage,
        progress_completed=None,
        progress_total=None,
        selected_strategy=selected_strategy,
        quality_gates_passed=None,
        mlflow_run_id=job.mlflow_run_id,
        mlflow_run_url=mlflow_run_url,
        error_message=job.error_message,
    )
    return templates.TemplateResponse(request, "jobs/detail.html", {"job": detail})


@router.get("/jobs/{job_id}/status-fragment")
async def job_status_fragment(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    polling = job.status in {"queued", "running", "created"}
    return templates.TemplateResponse(
        request,
        "jobs/status_fragment.html",
        {
            "job_id": job_id,
            "status": job.status,
            "stage": job.stage,
            "error_message": job.error_message,
            "polling": polling,
        },
    )


# ── Phase 9: Results summary ───────────────────────────────────────────────────


@router.get("/jobs/{job_id}/results")
async def job_results(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    eval_runs = await get_eval_runs_for_job(session, job_id)
    selected_strategy = await _get_selected_strategy(session, job)
    primary_metric = _resolve_primary_metric(job)

    score_map: dict[str, float | None] = {}
    malformed_map: dict[str, float | None] = {}
    for run in eval_runs:
        if run.split in {"validation", "test"}:
            score_map[run.strategy] = _score_for_metric(run, primary_metric)
            malformed_map[run.strategy] = run.malformed_output_rate

    baseline_score = score_map.get("baseline")
    dbscan_score = score_map.get("dbscan")
    hdbscan_score = score_map.get("hdbscan")
    selected_score = score_map.get(selected_strategy or "") if selected_strategy else None

    metric_delta: float | None = None
    if selected_score is not None and baseline_score is not None:
        metric_delta = selected_score - baseline_score

    summary = ResultsSummaryView(
        job_id=job_id,
        primary_metric=primary_metric,
        baseline_score=baseline_score,
        dbscan_score=dbscan_score,
        hdbscan_score=hdbscan_score,
        selected_score=selected_score,
        selected_strategy=selected_strategy,
        metric_delta=metric_delta,
        golden_failures=None,
        malformed_output_rate=malformed_map.get(selected_strategy or "")
        if selected_strategy
        else None,
        prompt_token_count=None,
        fixed_count=None,
        broken_count=None,
        quality_gates_passed=None,
        estimated_total_cost_usd=float(job.estimated_total_cost_usd)
        if job.estimated_total_cost_usd is not None
        else None,
        teacher_cost_usd=float(job.teacher_cost_usd) if job.teacher_cost_usd is not None else None,
        student_cost_usd=float(job.student_cost_usd) if job.student_cost_usd is not None else None,
        embedding_cost_usd=float(job.embedding_cost_usd)
        if job.embedding_cost_usd is not None
        else None,
        judge_cost_usd=float(job.judge_cost_usd) if job.judge_cost_usd is not None else None,
        total_model_calls=None,
        total_tokens=job.total_tokens if job.total_tokens > 0 else None,
        has_estimated_usage=False,
    )
    return templates.TemplateResponse(request, "jobs/results.html", {"summary": summary})


# ── Phase 10: Prompt & rules ──────────────────────────────────────────────────


@router.get("/jobs/{job_id}/prompt")
async def job_prompt(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} has not completed (status: {job.status}).",
        )
    pv = await get_selected_prompt_version(session, job_id)
    if pv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No selected prompt version found for job {job_id!r}.",
        )
    return templates.TemplateResponse(request, "jobs/prompt.html", {"pv": pv})


@router.get("/jobs/{job_id}/rules")
async def job_rules(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    pv = await get_selected_prompt_version(session, job_id)
    strategy = pv.strategy if pv else (job.selected_strategy or "hdbscan")
    rules = await get_synthesized_rules_for_job(session, job_id, strategy)
    return templates.TemplateResponse(
        request,
        "jobs/rules.html",
        {"rules": rules, "strategy": strategy, "job_id": job_id},
    )


# ── Phase 11: Eval report & failures ──────────────────────────────────────────


@router.get("/jobs/{job_id}/eval-report")
async def job_eval_report(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    eval_runs = await get_eval_runs_for_job(session, job_id)
    primary_metric = _resolve_primary_metric(job)
    return templates.TemplateResponse(
        request,
        "jobs/eval_report.html",
        {
            "eval_runs": eval_runs,
            "job_id": job_id,
            "primary_metric": primary_metric,
        },
    )


@router.get("/jobs/{job_id}/failures")
async def job_failures(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
    failure_class: str | None = None,
    split: str | None = None,
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    artifact_base = Path(settings.artifact_root) / job_id / "outputs"
    fixed_path = artifact_base / "failures_fixed.jsonl"
    broken_path = artifact_base / "failures_broken.jsonl"
    artifacts_pending = not (fixed_path.exists() or broken_path.exists())

    failures: list[dict[str, object]] = []
    if not artifacts_pending:
        target = fixed_path if failure_class == "fixed" else broken_path
        if failure_class in {"fixed", "broken"} and target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry: dict[str, object] = json.loads(stripped)
                    if split and entry.get("split") != split:
                        continue
                    failures.append(entry)
                except json.JSONDecodeError:
                    continue

    return templates.TemplateResponse(
        request,
        "jobs/failures.html",
        {
            "job_id": job_id,
            "artifacts_pending": artifacts_pending,
            "failures": failures,
            "failure_class": failure_class or "fixed",
            "split_filter": split or "",
        },
    )


# ── Phase 12: Artifacts manifest & download ────────────────────────────────────


@router.get("/jobs/{job_id}/artifacts")
async def job_artifacts(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    artifact_root = Path(settings.artifact_root) / job_id
    files: list[ArtifactFileView] = []
    for rel_str in _KNOWN_ARTIFACT_PATTERNS:
        full = artifact_root / rel_str
        if full.exists():
            ext = full.suffix.lower()
            content_type = _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")
            files.append(
                ArtifactFileView(
                    filename=full.name,
                    relative_path=rel_str,
                    download_url=f"/ui/jobs/{job_id}/artifacts/download?path={rel_str}",
                    content_type=content_type,
                )
            )

    manifest = ArtifactsView(job_id=job_id, files=files)
    return templates.TemplateResponse(request, "jobs/artifacts.html", {"manifest": manifest})


@router.get("/jobs/{job_id}/artifacts/download")
async def download_artifact(
    job_id: str,
    path: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> FileResponse:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    resolved = _safe_artifact_path(settings.artifact_root, job_id, path)

    if not resolved.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

    ext = resolved.suffix.lower()
    media_type = _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")
    return FileResponse(path=str(resolved), media_type=media_type, filename=resolved.name)
