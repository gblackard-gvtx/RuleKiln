---
description: "Task list for RuleKiln Minimal MVP Operator UI"
---

# Tasks: RuleKiln Minimal MVP Operator UI

**Input**: `Docs/plans/rulekiln_minimal_mvp_ui_spec.md`

**Prerequisites**: Existing FastAPI backend at `src/rulekiln/api/`, DB models, job creation flow, artifact writer, provider profiles in `AppSettings`.

**Tests**: Included — all UI tests must run offline using fake providers. No external model calls.

**Organization**: Grouped by phase so each phase is independently deliverable.

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Current API routes live at `/v1/jobs` — do **not** rename them. Add `/ui/` routes alongside.

---

## Phase 1: Dependencies & App Infrastructure

**Purpose**: Wire up all new packages and mount Jinja2 / static onto the existing FastAPI app.

Note: `python-multipart` is already in `pyproject.toml`. Only `jinja2` and `aiofiles` are new.

- [ ] T060 Add `jinja2` and `aiofiles` to `[project.dependencies]` in `pyproject.toml`
- [ ] T061 [P] Add `MLFLOW_UI_BASE_URL: str | None` field to `AppSettings` in `src/rulekiln/config/settings.py` (alias `MLFLOW_UI_BASE_URL`, default `None`)
- [ ] T061b [P] Add `selected_strategy: str | None` column to `DistillationJob` in `src/rulekiln/db/models.py`; add a corresponding Alembic migration in `migrations/versions/`
- [ ] T062 Mount `Jinja2Templates` from `src/rulekiln/templates/` and `StaticFiles` from `src/rulekiln/static/` in `src/rulekiln/api/app.py`; include the UI router (to be created in T072) under no prefix

**Checkpoint**: App starts, `/ui/` routes are registered, static files are served.

---

## Phase 2: Static Assets & Base Layout

**Purpose**: Create the shared HTML shell, CSS, and component partials that every page uses.

- [ ] T063 Create `src/rulekiln/static/app.css` — minimal Tailwind CDN link or hand-written utility CSS covering status badge colours (`queued`: gray, `running`: blue, `completed`: green, `failed`: red, `needs_review`: amber), card layout, table styles, and `<pre>` block styles
- [ ] T064 [P] Create `src/rulekiln/static/app.js` — empty placeholder (Alpine.js / HTMX loaded from CDN in base template)
- [ ] T065 Create `src/rulekiln/templates/base.html` — full-page shell with `<head>` (HTMX CDN, Alpine.js CDN, Tailwind CDN or `app.css`), top navigation (`Jobs`, `New Job`), flash message slot, and `{% block content %}` body slot
- [ ] T066 [P] Create `src/rulekiln/templates/components/flash.html` — success / error flash partial
- [ ] T067 [P] Create `src/rulekiln/templates/components/job_status_badge.html` — coloured inline badge for job status string
- [ ] T068 [P] Create `src/rulekiln/templates/components/metric_card.html` — small card displaying a label and numeric/text value; handles `None` as `—`
- [ ] T069 [P] Create `src/rulekiln/templates/components/artifact_link.html` — single artifact download anchor with filename and MIME hint

**Checkpoint**: `base.html` renders correctly; component partials are includable.

---

## Phase 3: View Models & UI Route Module

**Purpose**: Define Pydantic view models and the route module skeleton before building individual pages.

- [ ] T070 Create `src/rulekiln/ui/view_models.py` with the following models (all `BaseModel`, no bare dicts):
  - `JobListItemView` — `job_id`, `task_name`, `task_mode`, `status`, `stage | None`, `selected_strategy | None`, `primary_metric_delta: float | None`, `created_at: datetime`, `detail_url: str`
  - `JobDetailView` — core fields above plus split totals (`total_cases`, `train_cases`, `validation_cases`, `test_cases`, `golden_cases`), execution progress (`teacher_extraction_*`, `student_eval_split`, `student_eval_total`, per-strategy student completion counts), and diagnostics (`total_model_calls`, role-specific model calls, `micro_rules_count`, `synthesized_rules_count`, `selected_rules_count`)
  - `ResultsSummaryView` — core score fields above plus recommendation/cost fields (`best_strategy`, `baseline_macro_f1`, `best_macro_f1`, `macro_f1_delta`, `macro_f1_relative_lift_percent`, `accuracy_lift_percentage_points`, `best_malformed_output_rate`, `estimated_total_cost_usd`, per-role costs, `total_model_calls`, `total_tokens`, `has_estimated_usage`)
  - `ProviderRouteView` — `profile_name: str`, `model_id: str`, `supports_chat: bool`, `supports_embeddings: bool`
  - `PreviewView` — `task_id`, `task_name`, `task_mode`, `case_count: int`, `train_count: int`, `validation_count: int`, `test_count: int`, `golden_count: int`, `evaluation_methods: list[str]`, `output_schema_present: bool`, `provider_routes: list[ProviderRouteView]`, `estimated_teacher_calls: int`, `estimated_student_eval_calls: int`, `estimated_embedding_calls: int`, `warnings: list[str]`, `errors: list[str]`
  - `ArtifactFileView` — `filename: str`, `relative_path: str`, `download_url: str`, `content_type: str`
  - `ArtifactsView` — `job_id: str`, `files: list[ArtifactFileView]`

- [ ] T071 [P] Create `src/rulekiln/ui/forms.py` with `NewJobForm` — file upload fields (`task_file`, `cases_file`) and provider/model string fields (`teacher_profile`, `teacher_model`, `student_profile`, `student_model`, `embedding_profile`, `embedding_model`, `judge_profile | None`, `judge_model | None`, `baseline_prompt: str | None`); parse using FastAPI `Form(...)` and `UploadFile`

- [ ] T072 Create `src/rulekiln/ui/routes.py` — `APIRouter(prefix="/ui", tags=["ui"])` skeleton; import and wire all sub-handlers; template renderer helper using `Jinja2Templates`

**Checkpoint**: `view_models.py` and `forms.py` pass Pyright; `routes.py` registers with no errors.

---

## Phase 4: Job List Dashboard

**Purpose**: `GET /ui` and `GET /ui/jobs` — show recent jobs or an empty-state prompt.

- [ ] T073 Implement `GET /ui` redirect to `GET /ui/jobs` in `src/rulekiln/ui/routes.py`
- [ ] T074 Implement `GET /ui/jobs` handler — query `DistillationJob` table (latest 50, ordered by `created_at` desc), map to `list[JobListItemView]`, render `jobs/index.html`
- [ ] T075 Create `src/rulekiln/templates/jobs/index.html` — extends `base.html`; table of jobs with columns: job ID (link to detail), task name, mode, status badge, stage, selected strategy, metric delta, created time; empty-state message: _"No jobs yet. Create your first distillation job."_

**Checkpoint**: `GET /ui/jobs` returns 200 with an empty-state or populated table.

---

## Phase 5: New Job Form

**Purpose**: `GET /ui/jobs/new` — render the upload and model-selection form.

- [ ] T076 Implement `GET /ui/jobs/new` handler — load `AppSettings.provider_profiles`, build list of profile names, render `jobs/new.html`
- [ ] T077 Create `src/rulekiln/templates/jobs/new.html` — extends `base.html`; file input for `task.yaml`, file input for `cases.jsonl`, provider profile dropdowns (teacher / student / embedding / optional judge) populated from configured profiles, model ID text inputs, optional baseline prompt textarea, submit to `POST /ui/jobs/preview`; include security note that credentials are never in the form

**Checkpoint**: `GET /ui/jobs/new` renders a form with provider dropdowns from settings.

---

## Phase 6: Validate / Preview

**Purpose**: `POST /ui/jobs/preview` — parse and validate uploaded files before the user commits to running a job.

- [ ] T078 Implement `POST /ui/jobs/preview` handler in `src/rulekiln/ui/routes.py`:
  - Accept multipart form via `NewJobForm` fields
  - Read `task_file` bytes, parse as YAML — collect `task.yaml` errors
  - Read `cases_file` bytes, parse each line as JSON — collect `cases.jsonl` errors
  - Validate schema versions, task mode consistency, output schema presence, provider profile references using `validate_distillation_request` and custom preview checks
  - Count splits (train / validation / test / golden)
  - Resolve split policy and include fallback warning when evaluation is not on validation
  - Detect evaluation methods from case `evaluation_json`
  - Estimate model call counts
  - Build `PreviewView` with all errors and warnings
  - If validation passes, persist a **draft** `DistillationJob` row (status `draft`) containing the serialised `DistillationRequest` in `request_json`; pass the draft `job_id` to the template as a hidden field
  - If errors exist, render `jobs/preview.html` with errors and **no** submit button (do not persist a draft)
  - If clean, render `jobs/preview.html` with summary and **Run Pipeline** button whose form POSTs the draft `job_id` to `POST /ui/jobs`
  - Enforce upload extension whitelist: `.yaml`, `.yml`, `.jsonl` only
  - Enforce max upload size from settings (default 10 MB)
- [ ] T079 Create `src/rulekiln/templates/jobs/preview.html` — extends `base.html`; sections for task summary, case split counts, detected evaluation methods, provider route status, estimated call counts, warnings list, errors list; conditional **Run Pipeline** form with hidden `draft_job_id` field when no errors

**Checkpoint**: Uploading a malformed `task.yaml` shows an error and no submit button; valid files show split counts and a submit button.

---

## Phase 7: Create Job from UI

**Purpose**: `POST /ui/jobs` — promote the draft job row created by preview and start the pipeline.

- [ ] T080 Implement `POST /ui/jobs` handler in `src/rulekiln/ui/routes.py`:
  - Accept `draft_job_id` from the form body
  - Load the draft `DistillationJob` row; return 400 if not found or status is not `draft`
  - Transition the job status from `draft` → `created`
  - Deserialise `request_json` back into a `DistillationRequest` and re-run `validate_distillation_request(payload, settings)` as a safety check
  - Enqueue `run_distillation_pipeline` as a `BackgroundTask`
  - Redirect to `GET /ui/jobs/{job_id}` on success
  - Render `jobs/new.html` with flash error on failure
  - Do **not** duplicate any pipeline-starting logic already in `POST /v1/jobs`

**Checkpoint**: Submitting a valid form creates a DB row and redirects to the job detail page.

---

## Phase 8: Job Detail & HTMX Progress Polling

**Purpose**: `GET /ui/jobs/{job_id}` and `GET /ui/jobs/{job_id}/status-fragment` — show live progress.

- [ ] T081 Implement `GET /ui/jobs/{job_id}` handler — load `DistillationJob`, latest `StageMarker`, `EvalRun` for best strategy; build `JobDetailView` (include `mlflow_run_url` if `MLFLOW_UI_BASE_URL` is set); render `jobs/detail.html`
- [ ] T082 Create `src/rulekiln/templates/jobs/detail.html` — extends `base.html`; status badge/stage and HTMX polling fragment, split metric cards, execution progress (teacher extraction + per-strategy student eval counts), pipeline diagnostics (model calls by role + rule counts), MLflow run link, and links to results / prompt / rules / eval-report / failures / artifacts sub-pages
- [ ] T083 Implement `GET /ui/jobs/{job_id}/status-fragment` handler — return `jobs/status_fragment.html` partial only; omit polling attributes when status is `completed` or `failed`
- [ ] T084 Create `src/rulekiln/templates/jobs/status_fragment.html` — status badge, current stage, optional progress bar (`progress_completed / progress_total`), last error if failed, last-updated timestamp; includes `hx-get`, `hx-trigger="every 2s"`, `hx-swap="outerHTML"` only while status is `queued` or `running`

**Checkpoint**: Job detail page auto-refreshes progress without full page reload; stops polling on terminal status.

---

## Phase 9: Results Summary

**Purpose**: `GET /ui/jobs/{job_id}/results` — show metric comparison and quality gate outcome.

- [ ] T085 Implement `GET /ui/jobs/{job_id}/results` handler — load `EvalRun` rows for the job; build `ResultsSummaryView`; render `jobs/results.html`
- [ ] T086 Create `src/rulekiln/templates/jobs/results.html` — extends `base.html`; metric cards for baseline / DBSCAN / HDBSCAN / selected scores and delta; details table (selected strategy, quality gate, golden failures, malformed output, fixed/broken counts); recommendation section (best strategy, baseline macro_f1, relative lift, accuracy lift); run cost and usage summary; links to eval report and artifacts; no charts

**Checkpoint**: `GET /ui/jobs/{job_id}/results` renders scores and selected strategy for a completed job.

---

## Phase 10: Selected Prompt & Rules Views

**Purpose**: Read-only display of the distilled prompt and synthesized rules for a completed job.

- [ ] T087 Implement `GET /ui/jobs/{job_id}/prompt` handler — call `get_selected_prompt_version(session, job_id)`; render `jobs/prompt.html`; return 404 with useful message if job not completed
- [ ] T088 Create `src/rulekiln/templates/jobs/prompt.html` — extends `base.html`; shows version ID, strategy, prompt hash, compiler version, token estimate; prompt text inside an escaped `<pre>` block; copy-to-clipboard button (Alpine.js); download link to artifact
- [ ] T089 [P] Implement `GET /ui/jobs/{job_id}/rules` handler — call `get_synthesized_rules_for_job(session, job_id)`; render `jobs/rules.html`
- [ ] T090 [P] Create `src/rulekiln/templates/jobs/rules.html` — extends `base.html`; table with columns: rule ID, type, topic, applies_when, priority, source case count; each row expandable (Alpine.js `x-show`) to show full conditions and tie-breakers; read-only

**Checkpoint**: Prompt and rules pages render without errors for a completed job.

---

## Phase 11: Eval Report & Failures Views

**Purpose**: Detailed evaluation breakdown and fixed/broken case analysis.

- [ ] T091 Implement `GET /ui/jobs/{job_id}/eval-report` handler — load all `EvalRun` rows, read `outputs/strategy_comparison.json` for `evaluation_split_warning`, and render `jobs/eval_report.html`
- [ ] T092 Create `src/rulekiln/templates/jobs/eval_report.html` — extends `base.html`; table rows per strategy/split with primary metric score, accuracy, macro_f1, malformed rate, model; show fallback warning banner when `evaluation_split_warning` is present
- [ ] T093 [P] Implement `GET /ui/jobs/{job_id}/failures` handler — attempt to load `failures_fixed.jsonl` / `failures_broken.jsonl` from the artifact root; if files are absent (job still running or incomplete), pass an empty list and an `artifacts_pending: True` flag; support query params `?failure_class=fixed|broken&split=validation|golden|test`; render `jobs/failures.html`
- [ ] T094 [P] Create `src/rulekiln/templates/jobs/failures.html` — extends `base.html`; if `artifacts_pending` is true, show _"Artifact files are not yet available — the job must complete before failures can be reviewed."_ and hide the table; otherwise show filter controls (failure class, split) using HTMX `hx-get` on change; table with columns: case_id, split, failure class, expected (truncated), baseline output (truncated), distilled output (truncated), failed assertion, matched rules; expandable row details for long content

**Checkpoint**: Failures page renders with filter controls; filtering changes table content via HTMX.

---

## Phase 12: Artifacts Manifest & Download

**Purpose**: `GET /ui/jobs/{job_id}/artifacts` and `GET /ui/jobs/{job_id}/artifacts/download` — list and serve job output files.

- [ ] T095 Implement `GET /ui/jobs/{job_id}/artifacts` handler — scan `.rulekiln/runs/{job_id}/` for all known artifact files (see spec §8.12); build `ArtifactsView` with download URLs; render `jobs/artifacts.html`
- [ ] T096 Create `src/rulekiln/templates/jobs/artifacts.html` — extends `base.html`; sections for outputs / exports / metadata; each file shown with filename, size, and download link via `artifact_link.html` component
- [ ] T097 Implement `GET /ui/jobs/{job_id}/artifacts/download` handler with `?path=` query param:
  - Resolve path relative to `.rulekiln/runs/{job_id}/` only — reject absolute paths and any `..` components
  - Match file extension to safe `Content-Type`: `.md` → `text/markdown`, `.json` → `application/json`, `.jsonl` → `application/x-ndjson`, `.yaml` → `application/yaml`, `.txt` → `text/plain`
  - Use `FileResponse` or stream bytes with `aiofiles`
  - Return 404 if file not found, 400 if path is traversal attempt

**Checkpoint**: Artifact list shows files; download route streams files; path traversal returns 400.

---

## Phase 13: UI Tests

**Purpose**: Verify all UI routes work offline using fake providers — no external model calls.

- [ ] T098 Create `tests/ui/__init__.py`
- [ ] T099 [P] Create `tests/ui/test_job_list.py`:
  - `GET /ui/jobs` returns 200 with empty-state message when no jobs exist
  - `GET /ui/jobs` returns 200 with a table row after a job is seeded
- [ ] T100 [P] Create `tests/ui/test_new_job_form.py`:
  - `GET /ui/jobs/new` returns 200 and renders provider dropdowns from fake settings
- [ ] T101 [P] Create `tests/ui/test_preview.py`:
  - `POST /ui/jobs/preview` rejects an invalid `task.yaml` — response 200, errors section present, no submit button
  - `POST /ui/jobs/preview` rejects an invalid `cases.jsonl` — response 200, errors section present
  - `POST /ui/jobs/preview` rejects disallowed file extension (`.txt`)
  - `POST /ui/jobs/preview` accepts valid files — response 200, submit button present, split counts correct
- [ ] T102 [P] Create `tests/ui/test_create_job.py`:
  - `POST /ui/jobs` creates a job row and redirects to `/ui/jobs/{job_id}` with valid inputs using fake providers
- [ ] T103 [P] Create `tests/ui/test_job_detail.py`:
  - `GET /ui/jobs/{job_id}` returns 200 with status badge for a queued/running/completed/failed job
  - `GET /ui/jobs/{job_id}/status-fragment` renders without polling attributes when status is `completed`
  - `GET /ui/jobs/{job_id}/status-fragment` includes `hx-trigger` attribute when status is `running`
- [ ] T104 [P] Create `tests/ui/test_results.py`:
  - `GET /ui/jobs/{job_id}/results` renders selected strategy and metric delta for a completed job
- [ ] T105 [P] Create `tests/ui/test_artifacts.py`:
  - `GET /ui/jobs/{job_id}/artifacts/download` rejects `..` path traversal with 400
  - `GET /ui/jobs/{job_id}/artifacts/download` rejects absolute path with 400
  - `GET /ui/jobs/{job_id}/artifacts/download` streams a known artifact with correct `Content-Type`

**Checkpoint**: All UI tests pass via `pytest tests/ui/` with `fake` provider profiles and no internet access.

---

## Phase 14: Documentation & Cleanup

**Purpose**: Update the README and remove any temporary dev artefacts.

- [ ] T106 Update `README.md` with a **UI Usage** section covering: how to start the server, how to open `/ui/jobs/new`, required environment variables (`DATABASE_URL`, `MLFLOW_TRACKING_URI`, `PROVIDER_PROFILES`, optional `MLFLOW_UI_BASE_URL`), and how to run UI tests
- [ ] T107 Verify no hardcoded secrets, no raw stack traces rendered, no provider API keys in hidden form fields — fix any violations found
- [ ] T108 Remove any temporary scripts or test files created during development per the cleanup checklist in `AGENTS.md`

**Checkpoint**: README documents the UI; all tests pass; no temporary files remain.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Infrastructure)**: No dependencies — start immediately
- **Phase 2 (Static & Layout)**: Depends on Phase 1 (templates dir must exist)
- **Phase 3 (View Models & Routes)**: Can start in parallel with Phase 2 (different files)
- **Phases 4–12 (Feature pages)**: Each depends on Phases 1–3; pages within Phase 7–12 can progress in parallel once Phase 3 is done
- **Phase 13 (Tests)**: Each test file is independent; can begin as soon as the corresponding feature phase is done
- **Phase 14 (Docs & Cleanup)**: Final — after all tests pass

### Parallel Opportunities

All tasks marked `[P]` within a phase share no file dependencies and can be worked in parallel.

Phases 9–12 (results, prompt/rules, eval/failures, artifacts) are independent feature pages — can be tackled simultaneously once Phase 8 (job detail) is complete, since they share the same `base.html` and view model conventions.

---

## Acceptance Criteria Reference

See `Docs/plans/rulekiln_minimal_mvp_ui_spec.md` §20 for the full list.
Summary:
1. `/ui/jobs/new` renders with provider dropdowns
2. Files validate before submission; errors shown clearly
3. Valid job submission redirects and enqueues pipeline
4. Job detail auto-polls progress via HTMX
5. Results page shows strategy comparison and metric delta
6. Prompt, rules, eval-report, failures all render read-only
7. Artifacts downloadable; path traversal rejected
8. MLflow run ID and link visible when available
9. All UI tests pass offline with fake providers
10. No Next.js, Node build step, or external credentials required
