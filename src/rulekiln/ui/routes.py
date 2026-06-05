"""Operator UI routes — all phases (4-12)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse, RedirectResponse, Response

from rulekiln.api.validators.request_shape import (
    RequestValidationError,
    validate_distillation_request,
)
from rulekiln.config.settings import AppSettings, get_settings
from rulekiln.db.models import (
    Case,
    DistillationJob,
    EvalCaseResultRecord,
    MicroRule,
    StageMarker,
    SynthesizedRule,
)
from rulekiln.db.repositories.jobs import (
    cancel_job,
    create_job,
    get_eval_runs_for_job,
    get_job,
    get_selected_prompt_version,
    get_synthesized_rules_for_job,
    list_recent_jobs,
    retry_job,
    update_job_status,
)
from rulekiln.db.repositories.model_calls import summarize_model_call_events
from rulekiln.db.session import get_db_session
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.evaluator import get_primary_metric
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.schemas.classroom import PhaseTeacherConfig, TeacherConfig
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask, TaskMode
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
from rulekiln.workers.dbos_runtime import ensure_dbos_runtime_launched, require_dbos_available

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"])
logger = get_logger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml", ".jsonl"})
_CONTENT_TYPE_MAP: dict[str, str] = {
    ".csv": "text/csv",
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
    "outputs/baseline_scaffold_prompt.md",
    "outputs/baseline_scaffold_eval.json",
    "outputs/baseline_scaffold_prompt.md",
    "outputs/baseline_scaffold_eval.json",
    "outputs/selected_distilled_prompt.md",
    "outputs/rules_dbscan.jsonl",
    "outputs/rules_hdbscan.jsonl",
    "outputs/eval_report.json",
    "outputs/strategy_comparison.json",
    "outputs/confusion_matrix.csv",
    "outputs/per_label_metrics.csv",
    "outputs/top_confusions.md",
    "outputs/paired_comparison/fixed.jsonl",
    "outputs/paired_comparison/broken.jsonl",
    "outputs/paired_comparison/unchanged.jsonl",
    "outputs/paired_comparison/summary.json",
    "outputs/confusion_matrix.csv",
    "outputs/per_label_metrics.csv",
    "outputs/top_confusions.md",
    "outputs/paired_comparison/fixed.jsonl",
    "outputs/paired_comparison/broken.jsonl",
    "outputs/paired_comparison/unchanged.jsonl",
    "outputs/paired_comparison/summary.json",
    "outputs/failures_fixed.jsonl",
    "outputs/failures_broken.jsonl",
    "outputs/rule_provenance.json",
    "outputs/rule_provenance.md",
    "outputs/rule_ablation.json",
    "outputs/rule_provenance.json",
    "outputs/rule_provenance.md",
    "outputs/rule_ablation.json",
]
_DB_CASE_ID_DELIMITER = "::"
_EXTRACTION_CASE_MARKER_PREFIX = "extracting_case:"

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


def _auto_enable_extraction_batch(
    teacher_config: TeacherConfig | None,
    teacher_profile: str,
    teacher_model: str,
    settings: AppSettings,
) -> TeacherConfig | None:
    """Auto-enable extraction batch when the selected profile supports it."""
    provider_profile = settings.provider_profiles.get(teacher_profile)
    if provider_profile is None or not provider_profile.batch_enabled:
        return teacher_config

    if teacher_config is None:
        return TeacherConfig(
            default=PhaseTeacherConfig(provider=teacher_profile, model=teacher_model),
            instruction_extraction=PhaseTeacherConfig(
                provider=teacher_profile,
                model=teacher_model,
                batch_enabled=True,
            ),
        )

    # Auto-enable batch for the resolved extraction phase config when the selected
    # provider profile supports it.
    phase_cfg = teacher_config.for_phase("instruction_extraction")
    phase_provider = settings.provider_profiles.get(phase_cfg.provider)
    if phase_provider is None or not phase_provider.batch_enabled:
        return teacher_config

    if teacher_config.instruction_extraction is None:
        teacher_config.instruction_extraction = PhaseTeacherConfig(
            provider=phase_cfg.provider,
            model=phase_cfg.model,
            batch_enabled=True,
        )
    elif not teacher_config.instruction_extraction.batch_enabled:
        teacher_config.instruction_extraction.batch_enabled = True

    return teacher_config


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

    valid_task_modes = {
        "classification",
        "summarization",
        "extraction",
        "rubric_review",
        "routing",
        "tool_use",
        "freeform_generation",
        "agent_behavior",
    }
    if job.task_mode in valid_task_modes:
        return get_primary_metric(cast(TaskMode, job.task_mode))
    return "weighted_case_score"
    valid_task_modes = {
        "classification",
        "summarization",
        "extraction",
        "rubric_review",
        "routing",
        "tool_use",
        "freeform_generation",
        "agent_behavior",
    }
    if job.task_mode in valid_task_modes:
        return get_primary_metric(cast(TaskMode, job.task_mode))
    return "weighted_case_score"


def _score_for_metric(run: object, primary_metric: str) -> float | None:
    """Map a primary metric name to the corresponding EvalRun value."""
    metric = primary_metric.strip().lower()
    if metric == "macro_f1":
        return getattr(run, "macro_f1", None)
    if metric == "accuracy":
        return getattr(run, "accuracy", None)
    return getattr(run, "weighted_case_score", None)


def _select_summary_eval_run(eval_runs: Sequence[object], strategy: str) -> object | None:
    """Pick one EvalRun per strategy with split preference: validation -> test -> train."""
    strategy_runs = [run for run in eval_runs if getattr(run, "strategy", None) == strategy]
    if not strategy_runs:
        return None

    for split_name in ("validation", "test", "train"):
        split_runs = [run for run in strategy_runs if getattr(run, "split", None) == split_name]
        if split_runs:
            return split_runs[-1]
    return strategy_runs[-1]


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _count_non_empty_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_non_empty_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_non_empty_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _empty_split_counts() -> dict[str, int]:
    return {"train": 0, "validation": 0, "test": 0, "golden": 0}


def _split_policy_from_request_json(
    job: DistillationJob,
) -> tuple[dict[str, int], int, str, int, set[str]] | None:
    req_json = job.request_json
    if not isinstance(req_json, dict):
        return None

    try:
        payload = DistillationRequest.model_validate(req_json)
    except ValidationError:
        return None

    split_policy = resolve_split_policy(payload.cases)
    split_counts = _empty_split_counts()
    for split_name, count in split_policy.split_counts.items():
        split_counts[split_name] = int(count)
    train_case_ids = {case.id for case in split_policy.extraction_cases}

    return (
        split_counts,
        len(split_policy.extraction_cases),
        split_policy.evaluation_split,
        len(split_policy.evaluation_cases),
        train_case_ids,
    )


async def _load_db_case_split_counts(session: AsyncSession, job_id: str) -> dict[str, int]:
    counts = _empty_split_counts()
    result = await session.execute(
        select(Case.split, func.count()).where(Case.job_id == job_id).group_by(Case.split)
    )
    for split_name, count in result.all():
        split_key = str(split_name)
        counts[split_key] = int(count)
    return counts


def _payload_case_id_from_db_case_id(job_id: str, db_case_id: str) -> str:
    prefix = f"{job_id}{_DB_CASE_ID_DELIMITER}"
    if db_case_id.startswith(prefix):
        return db_case_id[len(prefix) :]
    return db_case_id


def _payload_case_id_from_extraction_marker(artifact_type: str | None) -> str | None:
    if artifact_type is None:
        return None
    if not artifact_type.startswith(_EXTRACTION_CASE_MARKER_PREFIX):
        return None
    return artifact_type[len(_EXTRACTION_CASE_MARKER_PREFIX) :]


async def _load_db_train_payload_case_ids(session: AsyncSession, job_id: str) -> set[str]:
    result = await session.execute(
        select(Case.id).where(Case.job_id == job_id, Case.split == "train")
    )
    payload_case_ids: set[str] = set()
    for db_case_id in result.scalars().all():
        payload_case_ids.add(_payload_case_id_from_db_case_id(job_id, str(db_case_id)))
    return payload_case_ids


def _resolve_eval_target_from_split_counts(split_counts: dict[str, int]) -> tuple[str | None, int]:
    for split_name in ("validation", "train", "test", "golden"):
        split_count = int(split_counts.get(split_name, 0))
        if split_count > 0:
            return split_name, split_count
    return None, 0


async def _load_teacher_extraction_completed_count(
    session: AsyncSession,
    job_id: str,
    *,
    allowed_payload_case_ids: set[str] | None,
) -> int:
    marker_result = await session.execute(
        select(func.distinct(StageMarker.artifact_type)).where(
            StageMarker.job_id == job_id,
            StageMarker.stage == "extracting_rules",
            StageMarker.artifact_type.like(f"{_EXTRACTION_CASE_MARKER_PREFIX}%"),
        )
    )
    marker_case_ids: set[str] = set()
    for artifact_type in marker_result.scalars().all():
        if not isinstance(artifact_type, str):
            continue
        payload_case_id = _payload_case_id_from_extraction_marker(artifact_type)
        if payload_case_id is not None:
            marker_case_ids.add(payload_case_id)

    micro_rule_result = await session.execute(
        select(func.distinct(MicroRule.case_id)).where(MicroRule.job_id == job_id)
    )
    micro_rule_case_ids: set[str] = set()
    for db_case_id in micro_rule_result.scalars().all():
        if not isinstance(db_case_id, str):
            continue
        micro_rule_case_ids.add(_payload_case_id_from_db_case_id(job_id, db_case_id))

    if allowed_payload_case_ids is not None:
        marker_case_ids = marker_case_ids.intersection(allowed_payload_case_ids)
        micro_rule_case_ids = micro_rule_case_ids.intersection(allowed_payload_case_ids)

    return max(len(marker_case_ids), len(micro_rule_case_ids))


async def _load_student_eval_completed_count(
    session: AsyncSession,
    *,
    job_id: str,
    strategy: str,
    split: str | None,
) -> int:
    if split is None:
        return 0

    completed_value = await session.scalar(
        select(func.count(func.distinct(EvalCaseResultRecord.case_id))).where(
            EvalCaseResultRecord.job_id == job_id,
            EvalCaseResultRecord.strategy == strategy,
            EvalCaseResultRecord.split == split,
        )
    )
    return int(completed_value or 0)


async def _load_student_eval_completed_counts(
    session: AsyncSession,
    *,
    job_id: str,
    split: str | None,
) -> dict[str, int]:
    if split is None:
        return {}

    result = await session.execute(
        select(
            EvalCaseResultRecord.strategy,
            func.count(func.distinct(EvalCaseResultRecord.case_id)),
        )
        .where(
            EvalCaseResultRecord.job_id == job_id,
            EvalCaseResultRecord.split == split,
        )
        .group_by(EvalCaseResultRecord.strategy)
        .order_by(EvalCaseResultRecord.strategy)
    )
    return {
        strategy: int(count)
        for strategy, count in result.all()
        if isinstance(strategy, str)
    }


async def _load_rule_counts(session: AsyncSession, job_id: str) -> tuple[int, int, int]:
    micro_rules_count_value = await session.scalar(
        select(func.count()).select_from(MicroRule).where(MicroRule.job_id == job_id)
    )
    synthesized_rules_count_value = await session.scalar(
        select(func.count()).select_from(SynthesizedRule).where(SynthesizedRule.job_id == job_id)
    )
    selected_rules_count_value = await session.scalar(
        select(func.count())
        .select_from(SynthesizedRule)
        .where(
            SynthesizedRule.job_id == job_id,
            SynthesizedRule.is_pruned == False,  # noqa: E712
        )
    )
    return (
        int(micro_rules_count_value or 0),
        int(synthesized_rules_count_value or 0),
        int(selected_rules_count_value or 0),
    )


def _model_call_count_for_role(by_role: object, role: str) -> int | None:
    if not isinstance(by_role, dict):
        return None
    role_bucket = by_role.get(role)
    if not isinstance(role_bucket, dict):
        return 0
    return _as_int(role_bucket.get("call_count")) or 0


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

    # ── Validate per-phase teacher overrides ──
    errors.extend(form.validate_phase_overrides())

    # ── Validate per-phase teacher overrides ──
    errors.extend(form.validate_phase_overrides())

    # ── Build DistillationRequest & cross-validate ──
    judge: ModelRoute | None = None
    if form.judge_profile and form.judge_model:
        judge = ModelRoute(provider_profile=form.judge_profile, model=form.judge_model)

    teacher_config = form.build_teacher_config()
    teacher_config = _auto_enable_extraction_batch(
        teacher_config,
        form.teacher_profile,
        form.teacher_model,
        settings,
    )
    classroom_config = form.build_classroom_config()
    anchor = classroom_config.anchor_student

    distillation_request = DistillationRequest(
        task=task,
        cases=cases,
        teacher=ModelRoute(provider_profile=form.teacher_profile, model=form.teacher_model),
        student=ModelRoute(provider_profile=anchor.provider, model=anchor.model),
        embedding=ModelRoute(provider_profile=form.embedding_profile, model=form.embedding_model),
        judge=judge,
        baseline_prompt=form.baseline_prompt or None,
        teacher_config=teacher_config,
        classroom_config=classroom_config if len(classroom_config.students) > 1 else None,
    )
    try:
        validate_distillation_request(distillation_request, settings)
    except (RequestValidationError, ValidationError) as exc:
        errors.append(str(exc))

    # ── Count splits and detect evaluation methods ──
    split_policy = resolve_split_policy(cases)
    split_counts: dict[str, int] = {
        "train": split_policy.split_counts.get("train", 0),
        "validation": split_policy.split_counts.get("validation", 0),
        "test": split_policy.split_counts.get("test", 0),
        "golden": split_policy.split_counts.get("golden", 0),
    }
    eval_method_set: set[str] = set()
    for case in cases:
        for assertion in case.evaluation.assertions:
            eval_method_set.add(assertion.type)

    if split_policy.fallback_warning is not None:
        warnings.append(split_policy.fallback_warning)

    # ── Provider routes for display ──
    student_specs = [
        (f"student_{i}" if i > 0 else "student", s.provider, s.model)
        for i, s in enumerate(classroom_config.students)
    ]
    route_specs: list[tuple[str, str, str]] = [
        ("teacher", form.teacher_profile, form.teacher_model),
        *student_specs,
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
        estimated_student_eval_calls=len(split_policy.evaluation_cases) * 3,
        estimated_embedding_calls=train_count * 5,
        warnings=warnings,
        errors=errors,
        teacher_routing=form.build_teacher_routing_view(),
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
        max_attempts=settings.worker_max_attempts,
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

    try:
        require_dbos_available()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # Ensure draft jobs (including older rows) use the current retry cap.
    job.max_attempts = settings.worker_max_attempts
    await session.commit()

    await update_job_status(session, job.id, status="pending")

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

    request_split_policy = _split_policy_from_request_json(job)
    db_split_counts = await _load_db_case_split_counts(session, job.id)
    db_case_total = sum(db_split_counts.values())

    split_counts = db_split_counts if db_case_total > 0 else _empty_split_counts()
    train_payload_case_ids: set[str] | None = None
    if db_case_total > 0:
        train_payload_case_ids = await _load_db_train_payload_case_ids(session, job.id)

    teacher_extraction_total = int(split_counts.get("train", 0))
    student_eval_split, student_eval_total = _resolve_eval_target_from_split_counts(split_counts)

    if db_case_total == 0 and request_split_policy is not None:
        (
            request_split_counts,
            request_teacher_total,
            request_eval_split,
            request_eval_total,
            request_train_case_ids,
        ) = request_split_policy
        split_counts = request_split_counts
        teacher_extraction_total = request_teacher_total
        student_eval_split = request_eval_split
        student_eval_total = request_eval_total
        train_payload_case_ids = request_train_case_ids

    teacher_extraction_completed = await _load_teacher_extraction_completed_count(
        session,
        job.id,
        allowed_payload_case_ids=train_payload_case_ids,
    )
    if teacher_extraction_total > 0:
        teacher_extraction_completed = min(teacher_extraction_completed, teacher_extraction_total)

    student_baseline_completed = await _load_student_eval_completed_count(
        session,
        job_id=job.id,
        strategy="baseline",
        split=student_eval_split,
    )
    student_dbscan_completed = await _load_student_eval_completed_count(
        session,
        job_id=job.id,
        strategy="dbscan",
        split=student_eval_split,
    )
    student_hdbscan_completed = await _load_student_eval_completed_count(
        session,
        job_id=job.id,
        strategy="hdbscan",
        split=student_eval_split,
    )
    student_eval_completed_counts = await _load_student_eval_completed_counts(
        session,
        job_id=job.id,
        split=student_eval_split,
    )

    usage_summary = await summarize_model_call_events(session, job.id)
    by_role = usage_summary.get("by_role")
    total_model_calls = _as_int(usage_summary.get("total_model_calls"))
    teacher_model_calls = _model_call_count_for_role(by_role, "teacher")
    student_model_calls = _model_call_count_for_role(by_role, "student")
    embedding_model_calls = _model_call_count_for_role(by_role, "embedding")
    judge_model_calls = _model_call_count_for_role(by_role, "judge")

    micro_rules_count, synthesized_rules_count, selected_rules_count = await _load_rule_counts(
        session, job.id
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
        total_cases=sum(split_counts.values()),
        train_cases=split_counts.get("train", 0),
        validation_cases=split_counts.get("validation", 0),
        test_cases=split_counts.get("test", 0),
        golden_cases=split_counts.get("golden", 0),
        teacher_extraction_completed=teacher_extraction_completed,
        teacher_extraction_total=teacher_extraction_total,
        student_eval_split=student_eval_split,
        student_eval_total=student_eval_total,
        student_baseline_completed=student_baseline_completed,
        student_dbscan_completed=student_dbscan_completed,
        student_hdbscan_completed=student_hdbscan_completed,
        total_model_calls=total_model_calls,
        teacher_model_calls=teacher_model_calls,
        student_model_calls=student_model_calls,
        embedding_model_calls=embedding_model_calls,
        judge_model_calls=judge_model_calls,
        micro_rules_count=micro_rules_count,
        synthesized_rules_count=synthesized_rules_count,
        selected_rules_count=selected_rules_count,
    )
    return templates.TemplateResponse(
        request,
        "jobs/detail.html",
        {
            "job": detail,
            "student_eval_completed_counts": student_eval_completed_counts,
        },
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job_from_ui(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> RedirectResponse:
    """Cancel a queued/running job from the operator UI.

    For DBOS backend, this performs a best-effort workflow cancellation.
    For all backends, it marks the job terminal with failed_terminal status.
    """
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    terminal_statuses = {"completed", "failed", "failed_terminal", "failed_retryable"}
    if job.status in terminal_statuses:
        return RedirectResponse(
            url=f"/ui/jobs/{job.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    try:
        ensure_dbos_runtime_launched(settings)
        from dbos import DBOS  # imported lazily to keep optional dependency behavior

        workflow_id = f"rulekiln-job-{job.id}"
        workflow_status = DBOS.get_workflow_status(workflow_id)
        if workflow_status is not None:
            DBOS.cancel_workflow(workflow_id)
            logger.info("ui_dbos_workflow_cancelled", job_id=job.id, workflow_id=workflow_id)
    except Exception as exc:
        logger.warning(
            "ui_dbos_workflow_cancel_failed",
            job_id=job.id,
            error=str(exc),
        )

    await cancel_job(session, job.id, error_message="Cancelled by operator.")
    logger.info("ui_job_cancelled", job_id=job.id, status="failed_terminal")
    return RedirectResponse(
        url=f"/ui/jobs/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/jobs/{job_id}/retry")
async def retry_job_from_ui(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> RedirectResponse:
    """Requeue a failed job and resume from existing persisted progress."""
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    retryable_statuses = {"failed", "failed_terminal", "failed_retryable"}
    if job.status not in retryable_statuses:
        return RedirectResponse(
            url=f"/ui/jobs/{job.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    try:
        require_dbos_available()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    resulting_status = await retry_job(session, job.id, queue_backed=True)
    logger.info(
        "ui_job_retried",
        job_id=job.id,
        resulting_status=resulting_status,
        execution_backend=settings.execution_backend,
        queued=True,
    )
    return RedirectResponse(
        url=f"/ui/jobs/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/jobs/{job_id}/status-fragment")
async def job_status_fragment(
    request: Request,
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    polling = job.status in {"queued", "running", "created", "waiting_for_retry"}
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
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    eval_runs = await get_eval_runs_for_job(session, job_id)
    selected_strategy = await _get_selected_strategy(session, job)
    primary_metric = _resolve_primary_metric(job)

    comparison_payload = _load_json_object(
        Path(settings.artifact_root) / job_id / "outputs" / "strategy_comparison.json"
    )
    if selected_strategy is None and comparison_payload is not None:
        payload_strategy = comparison_payload.get("selected_strategy")
        if isinstance(payload_strategy, str) and payload_strategy.strip():
            selected_strategy = payload_strategy.strip()

    score_map: dict[str, float | None] = {}
    malformed_map: dict[str, float | None] = {}
    run_map: dict[str, object] = {}

    strategy_names = {
        str(getattr(run, "strategy", ""))
        for run in eval_runs
        if isinstance(getattr(run, "strategy", None), str)
        and str(getattr(run, "strategy", "")).strip()
    }
    baseline_strategy = "baseline_scaffold" if "baseline_scaffold" in strategy_names else "baseline"

    for strategy in sorted(strategy_names):
        run = _select_summary_eval_run(eval_runs, strategy)
        if run is None:
            continue
        run_map[strategy] = run
        score_map[strategy] = _score_for_metric(run, primary_metric)
        malformed_map[strategy] = _as_float(getattr(run, "malformed_output_rate", None))

    baseline_score = score_map.get(baseline_strategy)
    dbscan_score = score_map.get("dbscan")
    hdbscan_score = score_map.get("hdbscan")
    selected_score = score_map.get(selected_strategy or "") if selected_strategy else None

    selected_malformed_output_rate = (
        malformed_map.get(selected_strategy or "") if selected_strategy else None
    )

    best_strategy = selected_strategy
    best_run = run_map.get(selected_strategy or "") if selected_strategy else None
    if best_run is None:
        ranked_runs = [
            (strategy, run)
            for strategy, run in run_map.items()
            if _as_float(getattr(run, "macro_f1", None)) is not None
        ]
        if ranked_runs:
            ranked_runs.sort(
                key=lambda item: _as_float(getattr(item[1], "macro_f1", None)) or 0.0,
                reverse=True,
            )
            best_strategy, best_run = ranked_runs[0]

    baseline_run = run_map.get(baseline_strategy)
    baseline_run = run_map.get(baseline_strategy)
    baseline_macro_f1 = _as_float(getattr(baseline_run, "macro_f1", None)) if baseline_run else None
    best_macro_f1 = _as_float(getattr(best_run, "macro_f1", None)) if best_run else None
    macro_f1_delta: float | None = None
    if baseline_macro_f1 is not None and best_macro_f1 is not None:
        macro_f1_delta = best_macro_f1 - baseline_macro_f1

    macro_f1_relative_lift_percent: float | None = None
    if macro_f1_delta is not None and baseline_macro_f1 is not None and baseline_macro_f1 != 0:
        macro_f1_relative_lift_percent = (macro_f1_delta / baseline_macro_f1) * 100.0

    baseline_accuracy = _as_float(getattr(baseline_run, "accuracy", None)) if baseline_run else None
    best_accuracy = _as_float(getattr(best_run, "accuracy", None)) if best_run else None
    accuracy_lift_percentage_points: float | None = None
    if baseline_accuracy is not None and best_accuracy is not None:
        accuracy_lift_percentage_points = (best_accuracy - baseline_accuracy) * 100.0

    best_malformed_output_rate = (
        _as_float(getattr(best_run, "malformed_output_rate", None)) if best_run else None
    )

    quality_gates_passed: bool | None = None
    golden_failures: int | None = None
    selected_prompt_token_count: int | None = None
    if comparison_payload is not None and selected_strategy:
        gate_payload: object | None = None
        strategy_gates_payload = comparison_payload.get("strategy_gates")
        if isinstance(strategy_gates_payload, dict):
            gate_payload = strategy_gates_payload.get(selected_strategy)

        if gate_payload is None:
            gate_key = f"{selected_strategy}_gate"
            gate_payload = comparison_payload.get(gate_key)

    selected_prompt_token_count: int | None = None
    if comparison_payload is not None and selected_strategy:
        gate_payload: object | None = None
        strategy_gates_payload = comparison_payload.get("strategy_gates")
        if isinstance(strategy_gates_payload, dict):
            gate_payload = strategy_gates_payload.get(selected_strategy)

        if gate_payload is None:
            gate_key = f"{selected_strategy}_gate"
            gate_payload = comparison_payload.get(gate_key)

        if isinstance(gate_payload, dict):
            quality_gates_passed = _as_bool(gate_payload.get("passed"))
            golden_failures = _as_int(gate_payload.get("golden_failures"))
            if selected_malformed_output_rate is None:
                selected_malformed_output_rate = _as_float(
                    gate_payload.get("malformed_output_rate")
                )

        prompt_tokens_payload = comparison_payload.get("strategy_prompt_tokens")
        if isinstance(prompt_tokens_payload, dict):
            selected_prompt_token_count = _as_int(prompt_tokens_payload.get(selected_strategy))

        prompt_tokens_payload = comparison_payload.get("strategy_prompt_tokens")
        if isinstance(prompt_tokens_payload, dict):
            selected_prompt_token_count = _as_int(prompt_tokens_payload.get(selected_strategy))

    outputs_dir = Path(settings.artifact_root) / job_id / "outputs"
    fixed_count = _count_non_empty_jsonl_rows(outputs_dir / "failures_fixed.jsonl")
    broken_count = _count_non_empty_jsonl_rows(outputs_dir / "failures_broken.jsonl")

    usage_summary = await summarize_model_call_events(session, job_id)
    total_model_calls = _as_int(usage_summary.get("total_model_calls"))
    has_persisted_usage = (total_model_calls or 0) > 0

    if has_persisted_usage:
        estimated_total_cost_usd = _as_float(usage_summary.get("estimated_total_cost_usd"))
        teacher_cost_usd = _as_float(usage_summary.get("teacher_cost_usd"))
        student_cost_usd = _as_float(usage_summary.get("student_cost_usd"))
        embedding_cost_usd = _as_float(usage_summary.get("embedding_cost_usd"))
        judge_cost_usd = _as_float(usage_summary.get("judge_cost_usd"))
        total_tokens = _as_int(usage_summary.get("total_tokens"))
        has_estimated_usage = bool(usage_summary.get("has_estimated_usage", False))
    else:
        estimated_total_cost_usd = (
            float(job.estimated_total_cost_usd)
            if job.estimated_total_cost_usd is not None
            else None
        )
        teacher_cost_usd = float(job.teacher_cost_usd) if job.teacher_cost_usd is not None else None
        student_cost_usd = float(job.student_cost_usd) if job.student_cost_usd is not None else None
        embedding_cost_usd = (
            float(job.embedding_cost_usd) if job.embedding_cost_usd is not None else None
        )
        judge_cost_usd = float(job.judge_cost_usd) if job.judge_cost_usd is not None else None
        total_tokens = job.total_tokens if job.total_tokens > 0 else None
        total_model_calls = None
        has_estimated_usage = False

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
        golden_failures=golden_failures,
        malformed_output_rate=selected_malformed_output_rate,
        prompt_token_count=selected_prompt_token_count,
        fixed_count=fixed_count,
        broken_count=broken_count,
        quality_gates_passed=quality_gates_passed,
        best_strategy=best_strategy,
        baseline_macro_f1=baseline_macro_f1,
        best_macro_f1=best_macro_f1,
        macro_f1_delta=macro_f1_delta,
        macro_f1_relative_lift_percent=macro_f1_relative_lift_percent,
        accuracy_lift_percentage_points=accuracy_lift_percentage_points,
        best_malformed_output_rate=best_malformed_output_rate,
        estimated_total_cost_usd=estimated_total_cost_usd,
        teacher_cost_usd=teacher_cost_usd,
        student_cost_usd=student_cost_usd,
        embedding_cost_usd=embedding_cost_usd,
        judge_cost_usd=judge_cost_usd,
        total_model_calls=total_model_calls,
        total_tokens=total_tokens,
        has_estimated_usage=has_estimated_usage,
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
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    eval_runs = await get_eval_runs_for_job(session, job_id)
    primary_metric = _resolve_primary_metric(job)
    comparison_payload = _load_json_object(
        Path(settings.artifact_root) / job_id / "outputs" / "strategy_comparison.json"
    )
    evaluation_split_warning: str | None = None
    selected_strategy_id: str | None = None
    selected_strategy_family: str | None = None
    best_baseline_strategy_id: str | None = None
    best_distilled_strategy_id: str | None = None
    paired_summary: dict[str, float | int | str | None] | None = None
    selected_strategy_id: str | None = None
    selected_strategy_family: str | None = None
    best_baseline_strategy_id: str | None = None
    best_distilled_strategy_id: str | None = None
    paired_summary: dict[str, float | int | str | None] | None = None
    if comparison_payload is not None:
        warning_value = comparison_payload.get("evaluation_split_warning")
        evaluation_split_warning = _as_non_empty_str(warning_value)

        selected_strategy_id = _as_non_empty_str(
            comparison_payload.get("selected_strategy_id")
        ) or _as_non_empty_str(comparison_payload.get("selected_strategy"))
        selected_strategy_family = _as_non_empty_str(
            comparison_payload.get("selected_strategy_family")
        )
        best_baseline_strategy_id = _as_non_empty_str(
            comparison_payload.get("best_baseline_strategy_id")
        )
        best_distilled_strategy_id = _as_non_empty_str(
            comparison_payload.get("best_distilled_strategy_id")
        )

        best_by_family_payload = comparison_payload.get("best_by_family")
        if isinstance(best_by_family_payload, dict):
            best_baseline_strategy_id = best_baseline_strategy_id or _as_non_empty_str(
                best_by_family_payload.get("baseline")
            )
            best_distilled_strategy_id = best_distilled_strategy_id or _as_non_empty_str(
                best_by_family_payload.get("distilled")
            )

        paired_payload = comparison_payload.get("paired_comparison")
        if isinstance(paired_payload, dict):
            paired_summary = {
                "fixed_count": _as_int(paired_payload.get("fixed_count")),
                "broken_count": _as_int(paired_payload.get("broken_count")),
                "unchanged_correct_count": _as_int(paired_payload.get("unchanged_correct_count")),
                "unchanged_wrong_count": _as_int(paired_payload.get("unchanged_wrong_count")),
                "net_fix_rate": _as_float(paired_payload.get("net_fix_rate")),
                "net_fix_rate_status": _as_non_empty_str(paired_payload.get("net_fix_rate_status")),
                "overall_net_fix_rate": _as_float(paired_payload.get("overall_net_fix_rate")),
            }

    if selected_strategy_family is None and selected_strategy_id is not None:
        selected_strategy_family = (
            "distilled" if selected_strategy_id in {"dbscan", "hdbscan"} else "baseline"
        )
        evaluation_split_warning = _as_non_empty_str(warning_value)

        selected_strategy_id = _as_non_empty_str(
            comparison_payload.get("selected_strategy_id")
        ) or _as_non_empty_str(comparison_payload.get("selected_strategy"))
        selected_strategy_family = _as_non_empty_str(
            comparison_payload.get("selected_strategy_family")
        )
        best_baseline_strategy_id = _as_non_empty_str(
            comparison_payload.get("best_baseline_strategy_id")
        )
        best_distilled_strategy_id = _as_non_empty_str(
            comparison_payload.get("best_distilled_strategy_id")
        )

        best_by_family_payload = comparison_payload.get("best_by_family")
        if isinstance(best_by_family_payload, dict):
            best_baseline_strategy_id = best_baseline_strategy_id or _as_non_empty_str(
                best_by_family_payload.get("baseline")
            )
            best_distilled_strategy_id = best_distilled_strategy_id or _as_non_empty_str(
                best_by_family_payload.get("distilled")
            )

        paired_payload = comparison_payload.get("paired_comparison")
        if isinstance(paired_payload, dict):
            paired_summary = {
                "fixed_count": _as_int(paired_payload.get("fixed_count")),
                "broken_count": _as_int(paired_payload.get("broken_count")),
                "unchanged_correct_count": _as_int(paired_payload.get("unchanged_correct_count")),
                "unchanged_wrong_count": _as_int(paired_payload.get("unchanged_wrong_count")),
                "net_fix_rate": _as_float(paired_payload.get("net_fix_rate")),
                "net_fix_rate_status": _as_non_empty_str(paired_payload.get("net_fix_rate_status")),
                "overall_net_fix_rate": _as_float(paired_payload.get("overall_net_fix_rate")),
            }

    if selected_strategy_family is None and selected_strategy_id is not None:
        selected_strategy_family = (
            "distilled" if selected_strategy_id in {"dbscan", "hdbscan"} else "baseline"
        )

    return templates.TemplateResponse(
        request,
        "jobs/eval_report.html",
        {
            "eval_runs": eval_runs,
            "job_id": job_id,
            "primary_metric": primary_metric,
            "evaluation_split_warning": evaluation_split_warning,
            "selected_strategy_id": selected_strategy_id,
            "selected_strategy_family": selected_strategy_family,
            "best_baseline_strategy_id": best_baseline_strategy_id,
            "best_distilled_strategy_id": best_distilled_strategy_id,
            "paired_summary": paired_summary,
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
