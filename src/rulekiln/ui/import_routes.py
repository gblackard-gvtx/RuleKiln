"""CSV import UI routes."""

from __future__ import annotations

import csv
import io
import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse, RedirectResponse, Response

from rulekiln.artifacts.writer import (
    write_cases_jsonl,
    write_import_mapping,
    write_import_preview,
    write_import_source_csv,
    write_task,
    write_validation_report,
)
from rulekiln.config.settings import AppSettings, get_settings
from rulekiln.db.models import DistillationJob
from rulekiln.db.repositories.jobs import create_job
from rulekiln.db.session import get_db_session
from rulekiln.importers.case_generator import generate_cases
from rulekiln.importers.column_mapping import CsvColumnMapping, CsvImportMapping, CsvImportPreview
from rulekiln.importers.csv_importer import parse_csv_preview
from rulekiln.importers.task_generator import generate_task
from rulekiln.importers.validator import validate_import_mapping
from rulekiln.observability.logging import get_logger
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask
from rulekiln.ui.forms import CsvMappingForm, CsvUploadForm
from rulekiln.ui.view_models import CsvGenerateView, CsvPreviewView

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/ui/import", tags=["import"])
logger = get_logger(__name__)

_TASK_MODES = [
    "classification",
    "summarization",
    "extraction",
    "rubric_review",
    "routing",
    "tool_use",
    "freeform_generation",
    "agent_behavior",
]


def _validated_import_id(import_id: str) -> str:
    try:
        return str(uuid.UUID(import_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid import identifier.",
        ) from exc


def _import_artifact_root(settings: AppSettings, import_id: str) -> Path:
    base_dir = Path(settings.import_artifact_root).resolve()
    safe_import_id = _validated_import_id(import_id)
    resolved = (base_dir / safe_import_id).resolve()
    if not resolved.is_relative_to(base_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not permitted.",
        )
    return resolved


def _normalize_row(row: dict[str | None, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[key] = "" if value is None else value
    return normalized


def _read_csv_rows(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return [_normalize_row(row) for row in reader]


async def _parse_csv_mapping_form(request: Request) -> CsvMappingForm:
    form = CsvMappingForm(request)
    await form.parse()
    return form


@router.get("/csv")
async def csv_upload_form(request: Request) -> Response:
    """Render the first-step CSV upload form."""
    return templates.TemplateResponse(
        request,
        "import/csv/upload.html",
        {"task_modes": _TASK_MODES},
    )


@router.post("/csv/preview")
async def csv_preview(
    request: Request,
    form: Annotated[CsvUploadForm, Depends(CsvUploadForm)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    """Parse an uploaded CSV file and show a preview with suggested mappings."""
    errors: list[str] = []

    content = await form.csv_file.read()
    if len(content) > settings.max_csv_upload_size_bytes:
        errors.append(
            f"CSV file exceeds maximum size ({settings.max_csv_upload_size_bytes} bytes)."
        )
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {"task_modes": _TASK_MODES, "errors": errors},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    preview = parse_csv_preview(content, file_name=form.csv_file.filename or "upload.csv")

    if preview.errors:
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {"task_modes": _TASK_MODES, "errors": preview.errors},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    if preview.row_count > settings.max_csv_rows:
        errors.append(
            f"CSV has {preview.row_count} rows which exceeds the limit of {settings.max_csv_rows}."
        )
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {"task_modes": _TASK_MODES, "errors": errors},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    import_id = str(uuid.uuid4())
    artifact_root = _import_artifact_root(settings, import_id)
    write_import_source_csv(artifact_root, content)
    write_import_preview(artifact_root, preview)

    suggested = [
        {
            "column_name": mapping.column_name,
            "role": mapping.role,
            "path": mapping.path or "",
            "assertion_type": mapping.assertion_type,
            "create_assertion": mapping.create_assertion,
        }
        for mapping in preview.suggested_mappings
    ]

    view = CsvPreviewView(
        import_id=import_id,
        file_name=preview.file_name,
        task_id=form.task_id,
        task_name=form.task_name,
        task_mode=form.task_mode,
        description=form.description,
        row_count=preview.row_count,
        columns=preview.columns,
        sample_rows=preview.sample_rows,
        inferred_types=preview.inferred_types,
        suggested_mappings=suggested,
        warnings=preview.warnings,
        errors=preview.errors,
    )

    return templates.TemplateResponse(
        request,
        "import/csv/preview.html",
        {"view": view, "task_modes": _TASK_MODES},
    )


@router.post("/csv/generate")
async def csv_generate(
    request: Request,
    mapping_form: Annotated[CsvMappingForm, Depends(_parse_csv_mapping_form)],
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> Response:
    """Accept column mappings and generate import artifacts."""
    form_data = await request.form()
    import_id = mapping_form.import_id.strip()
    task_id = str(form_data.get("task_id", "")).strip()
    task_name = str(form_data.get("task_name", "")).strip()
    task_mode = str(form_data.get("task_mode", "")).strip()
    description = str(form_data.get("description", "")).strip()

    if not import_id:
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {
                "task_modes": _TASK_MODES,
                "errors": ["Missing import session. Please restart the import."],
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    artifact_root = _import_artifact_root(settings, import_id)
    source_csv = artifact_root / "source.csv"
    if not source_csv.exists():
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {
                "task_modes": _TASK_MODES,
                "errors": ["Import session expired. Please re-upload your CSV."],
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    columns = [
        CsvColumnMapping.model_validate(column_mapping)
        for column_mapping in mapping_form.column_mappings
    ]

    mapping = CsvImportMapping(
        task_id=task_id,
        task_name=task_name,
        task_mode=task_mode,
        description=description,
        columns=columns,
    )

    rows = _read_csv_rows(source_csv.read_bytes())

    errors, warnings = validate_import_mapping(mapping, rows)
    if errors:
        preview_path = artifact_root / "import_preview.json"
        preview_data: CsvImportPreview | None = None
        if preview_path.exists():
            preview_data = CsvImportPreview.model_validate(
                json.loads(preview_path.read_text(encoding="utf-8"))
            )

        preview_columns = preview_data.columns if preview_data is not None else []
        if not preview_columns and rows:
            preview_columns = list(rows[0].keys())
        suggested = [
            {
                "column_name": column.column_name,
                "role": column.role,
                "path": column.path or "",
                "assertion_type": column.assertion_type,
                "create_assertion": column.create_assertion,
            }
            for column in columns
        ]
        view = CsvPreviewView(
            import_id=import_id,
            file_name=preview_data.file_name if preview_data is not None else source_csv.name,
            task_id=task_id,
            task_name=task_name,
            task_mode=task_mode,
            description=description,
            row_count=len(rows),
            columns=preview_columns,
            sample_rows=preview_data.sample_rows if preview_data is not None else rows[:5],
            inferred_types=preview_data.inferred_types if preview_data is not None else {},
            suggested_mappings=suggested,
            warnings=warnings,
            errors=errors,
        )
        return templates.TemplateResponse(
            request,
            "import/csv/preview.html",
            {"view": view, "task_modes": _TASK_MODES},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    cases, case_errors = generate_cases(rows, mapping)
    errors.extend(case_errors)

    if errors:
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {"task_modes": _TASK_MODES, "errors": errors},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    input_columns = [column for column in columns if column.role == "input"]
    expected_columns = [column for column in columns if column.role in ("expected", "assertion")]
    task = generate_task(mapping, input_columns, expected_columns)

    try:
        RuleKilnTask.model_validate(task.model_dump(mode="json"))
    except ValidationError as exc:
        logger.warning(
            "csv_import_task_schema_validation_failed",
            import_id=import_id,
            validation_errors=exc.errors(),
        )
        errors.append("Generated task failed schema validation. Please review your mapping.")

    for case in cases:
        try:
            RuleKilnCase.model_validate(case.model_dump(mode="json"))
        except ValidationError as exc:
            logger.warning(
                "csv_import_case_schema_validation_failed",
                import_id=import_id,
                case_id=case.id,
                validation_errors=exc.errors(),
            )
            errors.append(f"Case '{case.id}' failed schema validation. Please review your mapping.")
            break

    if errors:
        return templates.TemplateResponse(
            request,
            "import/csv/upload.html",
            {"task_modes": _TASK_MODES, "errors": errors},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    write_task(artifact_root, task)
    write_cases_jsonl(artifact_root, cases)
    write_import_mapping(artifact_root, mapping)
    write_validation_report(artifact_root, errors, warnings)

    view = CsvGenerateView(
        import_id=import_id,
        task_yaml_url=f"/ui/import/csv/{import_id}/download/task.yaml",
        cases_jsonl_url=f"/ui/import/csv/{import_id}/download/cases.jsonl",
        mapping_yaml_url=f"/ui/import/csv/{import_id}/download/import_mapping.yaml",
        errors=errors,
        warnings=warnings,
        case_count=len(cases),
    )

    logger.info("csv_import_generated", import_id=import_id, case_count=len(cases))

    return templates.TemplateResponse(request, "import/csv/generate.html", {"view": view})


@router.get("/csv/{import_id}/download/{filename}")
async def download_import_artifact(
    import_id: str,
    filename: str,
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> FileResponse:
    """Download a generated import artifact."""
    if filename == "task.yaml":
        safe_filename = "task.yaml"
    elif filename == "cases.jsonl":
        safe_filename = "cases.jsonl"
    elif filename == "import_mapping.yaml":
        safe_filename = "import_mapping.yaml"
    elif filename == "validation_report.json":
        safe_filename = "validation_report.json"
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

    artifact_root = _import_artifact_root(settings, import_id)
    path = artifact_root / safe_filename
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

    return FileResponse(str(path), filename=safe_filename)


@router.post("/csv/run")
async def csv_run(
    settings: Annotated[AppSettings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    import_id: Annotated[str, Form()],
) -> RedirectResponse:
    """Create a draft distillation job from the generated import artifacts."""
    artifact_root = _import_artifact_root(settings, import_id)
    task_path = artifact_root / "task.yaml"
    cases_path = artifact_root / "cases.jsonl"

    if not task_path.exists() or not cases_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Import artifacts not found. Please regenerate.",
        )

    task = RuleKilnTask.model_validate(yaml.safe_load(task_path.read_text(encoding="utf-8")))
    cases: list[RuleKilnCase] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            cases.append(RuleKilnCase.model_validate(json.loads(stripped)))

    profile_names = sorted(settings.provider_profiles.keys())
    if not profile_names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No provider profiles are configured.",
        )
    first_profile = profile_names[0]

    distillation_request = DistillationRequest(
        task=task,
        cases=cases,
        teacher=ModelRoute(provider_profile=first_profile, model=""),
        student=ModelRoute(provider_profile=first_profile, model=""),
        embedding=ModelRoute(provider_profile=first_profile, model=""),
    )

    job_id = str(uuid.uuid4())
    job = DistillationJob(
        id=job_id,
        task_id=task.task_id,
        task_name=task.task_name,
        task_mode=task.task_mode,
        status="draft",
        stage=None,
        request_json=distillation_request.model_dump(mode="json"),
    )
    await create_job(session, job)

    job_root = Path(settings.artifact_root) / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_path, job_root / "task.yaml")
    shutil.copy2(cases_path, job_root / "cases.jsonl")

    logger.info("csv_import_job_created", job_id=job_id, import_id=import_id)

    return RedirectResponse(
        url=f"/ui/jobs/{job_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
