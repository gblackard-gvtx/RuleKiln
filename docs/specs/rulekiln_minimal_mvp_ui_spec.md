# RuleKiln Minimal MVP UI Spec

## 1. Purpose

The RuleKiln MVP UI is an **operator console** for running and reviewing prompt-hardening jobs.

It should make the existing RuleKiln pipeline easier to use without creating a second frontend application or duplicating backend state.

The UI should answer:

- What task and cases did I upload?
- Did the files validate?
- Which provider/model routes will be used?
- Is the job running?
- What stage is it in?
- Did the hardened prompt improve over baseline?
- Which strategy won: DBSCAN or HDBSCAN?
- What broke?
- What prompt was selected?
- Where are the artifacts?
- Where is the MLflow audit run?

This UI is not intended to be a full prompt IDE, hosted SaaS dashboard, or collaborative review workspace.

---

## 2. Senior Engineering Decision

Use:

- **FastAPI**
- **Jinja2 templates**
- **HTMX** for polling and partial page updates
- **Tailwind CSS** for styling
- optional **Alpine.js** only for small client-side interactions

Do **not** use Next.js for the MVP.

Reasoning:

- RuleKiln already has a FastAPI backend.
- The MVP UI is operational, not consumer-grade.
- A separate Next.js app would add routing, deployment, auth, CORS/session handling, build tooling, and duplicated frontend state.
- The main MVP risk is the pipeline, evaluation quality, artifact clarity, and review flow — not frontend richness.

Next.js can be reconsidered later if RuleKiln becomes a hosted multi-user product with complex review workflows, interactive diffing, RBAC, team workspaces, or rich dashboard requirements.

---

## 3. UX Principle

Build an **operator console**, not a prompt IDE.

The MVP UI should optimize for:

- fast job submission
- clear validation feedback
- visible progress
- understandable results
- artifact access
- MLflow handoff

Avoid building:

- drag-and-drop prompt editors
- collaborative review
- custom charting dashboards
- complex filters
- authentication/RBAC
- live WebSockets
- visual workflow builders
- prompt marketplace features

---

## 4. Relationship to MLflow

MLflow remains a separate UI.

RuleKiln UI owns the operator workflow:

```text
Upload task/cases
Select providers/models
Run pipeline
Watch progress
Review result
Download artifacts
Open MLflow run
```

MLflow UI owns the engineering/audit workflow:

```text
Compare runs
Inspect params
Inspect metrics
Browse artifacts
Review experiment history
Review prompt registry entries when available
```

The RuleKiln UI must show the MLflow run ID and provide an **Open MLflow Run** link when available.

RuleKiln should not rebuild MLflow's experiment comparison UI.

MVP MLflow behavior:

- Required: MLflow Tracking run, metrics, params, artifacts
- Optional: MLflow Prompt Registry registration
- If Prompt Registry registration fails or is unavailable, the UI should show `Prompt Registry: skipped` or `Prompt Registry: unavailable`, while still linking to the MLflow run and artifacts.

---

## 5. Scope

### In Scope

- Upload `task.yaml`
- Upload `cases.jsonl`
- Select teacher/student/embedding provider routes
- Optional judge model selection
- Optional baseline prompt
- Validate uploaded files before running
- Create distillation job
- Show job list
- Show job detail/progress
- Poll job status
- Show results summary
- Show selected prompt
- Show synthesized rules
- Show eval report
- Show fixed/broken failures
- Show artifact download links
- Show MLflow run link

### Out of Scope

- Authentication
- RBAC
- Multi-tenant workspaces
- Hosted SaaS billing
- Prompt editing workflow
- Rule editing workflow
- Manual approval workflow beyond read-only review
- Real-time WebSocket updates
- Advanced charts
- MLflow replacement UI
- External provider smoke-test UI
- Dataset management UI
- Case-by-case annotation UI

---

## 6. Dependencies

Add dependencies:

```text
jinja2
python-multipart
aiofiles
```

Optional:

```text
htmx via CDN or static vendored file
tailwind via compiled CSS, CDN for early dev, or static build later
alpinejs via CDN or static vendored file
```

Preferred MVP approach:

- Use HTMX from CDN or vendored static file.
- Use prebuilt Tailwind CSS from `src/rulekiln/static/app.css`.
- Avoid adding a Node build step for the MVP.

If Tailwind build tooling is added later, keep it isolated from the Python backend release path.

---

## 7. Directory Layout

```text
src/rulekiln/
  ui/
    routes.py
    forms.py
    view_models.py

  templates/
    base.html

    components/
      flash.html
      job_status_badge.html
      metric_card.html
      artifact_link.html

    jobs/
      index.html
      new.html
      preview.html
      detail.html
      status_fragment.html
      results.html
      prompt.html
      rules.html
      eval_report.html
      failures.html
      artifacts.html

  static/
    app.css
    app.js
```

Keep human-facing HTML routes separate from machine-facing API routes.

```text
/api/... = JSON API
/ui/...  = HTML UI
```

If the current API routes are not prefixed with `/api`, do not rename them in the MVP. Instead, add `/ui` routes alongside the existing endpoints.

---

## 8. UI Routes

### 8.1 Dashboard / Job List

```http
GET /ui
GET /ui/jobs
```

Purpose:

Show recent distillation jobs.

Display:

- job ID
- task name
- task mode
- status
- stage
- selected strategy
- primary metric delta
- created time
- link to details

Actions:

- New job
- View completed job
- Open MLflow run when available

---

### 8.2 New Job Form

```http
GET /ui/jobs/new
```

Purpose:

Upload a RuleKiln task and cases, select models, and optionally include a baseline prompt.

Fields:

```text
task.yaml file
cases.jsonl file

teacher provider profile
teacher model

student provider profile
student model

embedding provider profile
embedding model

optional judge provider profile
optional judge model

optional baseline prompt textarea
```

Provider profile dropdowns should be populated from configured `AppSettings.provider_profiles`.

The form should not ask for provider credentials. Credentials must come from environment/runtime configuration only.

---

### 8.3 Validate Upload / Preview

```http
POST /ui/jobs/preview
```

Purpose:

Validate files before creating a job.

Input:

- `task.yaml`
- `cases.jsonl`
- selected model routes
- optional baseline prompt

Server-side validation:

- parse `task.yaml`
- parse `cases.jsonl`
- validate schema versions
- validate task mode consistency
- validate output schema presence
- validate provider profile references
- validate required model routes
- count splits
- resolve split policy (extraction uses `train`; evaluation prefers `validation`, then falls back to `train`/`test`/`golden`)
- detect evaluation methods
- estimate model calls

Display:

```text
task_id
task_name
task_mode
case count
train count
validation count
test count
golden count
detected evaluation methods
output schema status
provider route status
estimated teacher calls
estimated student eval calls
estimated embedding calls
split fallback warning (when evaluation is not on validation)
warnings
errors
```

If validation fails, show errors and do not allow submission.

If validation passes, show a **Run Pipeline** button.

Implementation note:

For MVP, the preview can store parsed data temporarily in a signed server-side token, temporary file, or database draft record. Prefer a database draft record if job creation and preview state become complex. For a minimal first pass, preview can re-submit the files to the final create route.

---

### 8.4 Create Job From UI

```http
POST /ui/jobs
```

Purpose:

Create a distillation job using the existing backend job creation path.

Behavior:

- construct canonical `DistillationRequest`
- call the same service used by `POST /distillation-jobs`
- create job row
- enqueue background task
- redirect to `/ui/jobs/{job_id}`

Do not duplicate pipeline-starting logic in the UI route.

---

### 8.5 Job Detail

```http
GET /ui/jobs/{job_id}
```

Purpose:

Show current status, progress, and links to outputs.

Display:

- status
- stage
- case split counts (total / train / validation / test / golden)
- execution progress:
  - teacher extraction completed vs total train cases
  - student evaluation split
  - student baseline/DBSCAN/HDBSCAN completed vs eval total
- pipeline diagnostics:
  - total model calls and calls by role (teacher/student/embedding/judge)
  - micro rules, synthesized rules, selected rules
- task name
- task mode
- selected strategy when available
- quality gate status when available
- MLflow run link when available
- links to results / prompt / rules / eval-report / failures / artifacts

If job is running, include HTMX polling fragment.

---

### 8.6 Status Fragment

```http
GET /ui/jobs/{job_id}/status-fragment
```

Purpose:

HTMX-polled partial HTML fragment for job progress.

Example usage:

```html
<div
  hx-get="/ui/jobs/{{ job_id }}/status-fragment"
  hx-trigger="every 2s"
  hx-swap="outerHTML">
  {% include "jobs/status_fragment.html" %}
</div>
```

Display:

- status badge
- current stage
- progress bar if progress totals are known
- latest error if failed
- timestamp of last update

Polling behavior:

- poll every 2 seconds while status is `queued`, `running`, `created`, or `waiting_for_retry`
- stop polling once status is terminal (`completed`, `failed`, `failed_terminal`, `failed_retryable`)

HTMX can do this by rendering a fragment without the polling attributes for terminal states.

---

### 8.7 Results Summary

```http
GET /ui/jobs/{job_id}/results
```

Purpose:

Show the high-level job result.

Display:

- baseline score
- DBSCAN score
- HDBSCAN score
- selected strategy
- metric delta
- recommendation metrics:
  - best strategy
  - baseline macro_f1
  - best macro_f1
  - relative lift
  - accuracy lift (percentage points)
- quality gate result
- golden failures
- malformed output rate
- fixed count
- broken count
- cost/usage summary (total and per-role cost, total tokens, model-call count)
- links to full eval report and artifact downloads

Do not build advanced charts yet. Use simple metric cards and tables.

---

### 8.8 Selected Prompt

```http
GET /ui/jobs/{job_id}/prompt
```

Purpose:

Show selected distilled prompt.

Display:

- prompt version ID
- strategy
- prompt hash
- compiler version
- token estimate
- read-only prompt text
- copy button
- download link

MVP prompt view is read-only.

No prompt editing in MVP.

---

### 8.9 Rules

```http
GET /ui/jobs/{job_id}/rules
```

Purpose:

Show synthesized rules for the selected strategy.

Display:

- rule ID
- rule type
- topic
- applies_when
- outcome conditions
- priority
- source case count
- source case IDs
- tie-breakers

MVP rules view is read-only.

No rule editing in MVP.

---

### 8.10 Eval Report

```http
GET /ui/jobs/{job_id}/eval-report
```

Purpose:

Show detailed evaluation summary.

Display:

- primary metric
- per-run rows including strategy and split
- score for selected primary metric
- accuracy
- macro_f1
- malformed output rate
- model identifier
- evaluation split fallback warning banner when non-validation evaluation is used

---

### 8.11 Failures

```http
GET /ui/jobs/{job_id}/failures
```

Purpose:

Show fixed and broken failures.

MVP filters:

- failure class: fixed / broken
- split: validation / golden / test
- output path
- assertion type

Table columns:

```text
case_id
split
failure_class
expected
baseline_output
distilled_output
failed_assertion
matched_rules
```

For large outputs, truncate in table and provide expandable details.

No annotation or editing in MVP.

---

### 8.12 Artifacts

```http
GET /ui/jobs/{job_id}/artifacts
```

Purpose:

Show artifact manifest and download links.

Display files from:

```text
.rulekiln/runs/{job_id}/outputs/
.rulekiln/runs/{job_id}/exports/
.rulekiln/runs/{job_id}/metadata/
```

Required links:

```text
selected_distilled_prompt.md
rules_dbscan.jsonl
rules_hdbscan.jsonl
eval_report.json
strategy_comparison.json
failures_fixed.jsonl
failures_broken.jsonl
cases.normalized.jsonl
promptfoo.yaml
mlflow_run_id.txt
settings_snapshot.json
```

---

## 9. View Models

Do not pass raw database models directly into templates.

Create view models in:

```text
src/rulekiln/ui/view_models.py
```

Suggested models:

```python
class JobListItemView(BaseModel):
    job_id: str
    task_name: str
    task_mode: str
    status: str
    stage: str | None
    selected_strategy: str | None
    primary_metric_delta: float | None
    created_at: datetime
    detail_url: str


class JobDetailView(BaseModel):
    job_id: str
    task_name: str
    task_mode: str
    status: str
    stage: str | None
    progress_completed: int | None
    progress_total: int | None
    selected_strategy: str | None
    quality_gates_passed: bool | None
    mlflow_run_id: str | None
    mlflow_run_url: str | None
    error_message: str | None
    total_cases: int | None
    train_cases: int | None
    validation_cases: int | None
    test_cases: int | None
    golden_cases: int | None
    teacher_extraction_completed: int | None
    teacher_extraction_total: int | None
    student_eval_split: str | None
    student_eval_total: int | None
    student_baseline_completed: int | None
    student_dbscan_completed: int | None
    student_hdbscan_completed: int | None
    total_model_calls: int | None
    teacher_model_calls: int | None
    student_model_calls: int | None
    embedding_model_calls: int | None
    judge_model_calls: int | None
    micro_rules_count: int | None
    synthesized_rules_count: int | None
    selected_rules_count: int | None


class ResultsSummaryView(BaseModel):
    job_id: str
    primary_metric: str
    baseline_score: float | None
    dbscan_score: float | None
    hdbscan_score: float | None
    selected_score: float | None
    selected_strategy: str | None
    metric_delta: float | None
    golden_failures: int | None
    malformed_output_rate: float | None
    prompt_token_count: int | None
    fixed_count: int | None
    broken_count: int | None
    quality_gates_passed: bool | None
    best_strategy: str | None
    baseline_macro_f1: float | None
    best_macro_f1: float | None
    macro_f1_delta: float | None
    macro_f1_relative_lift_percent: float | None
    accuracy_lift_percentage_points: float | None
    best_malformed_output_rate: float | None
    estimated_total_cost_usd: float | None
    teacher_cost_usd: float | None
    student_cost_usd: float | None
    embedding_cost_usd: float | None
    judge_cost_usd: float | None
    total_model_calls: int | None
    total_tokens: int | None
    has_estimated_usage: bool
```

---

## 10. Upload Validation Rules

The UI should validate early and fail clearly.

Validation errors should be grouped by:

```text
task.yaml errors
cases.jsonl errors
provider route errors
evaluation config errors
safety/cost warnings
```

Examples:

```text
task.yaml is missing task_id
cases.jsonl line 14 is invalid JSON
case_042 task_mode does not match task.task_mode
provider profile bedrock-primary does not support embeddings
output_schema is missing required fields
golden cases exist but no assertions were provided
```

Warnings should not block submission unless they indicate invalid execution.

Warnings:

```text
No golden cases provided
No validation cases detected. Evaluation fell back to split=train.
No baseline prompt provided
No judge model configured for rubric_judge assertions
Large case count may exceed configured model call limits
```

---

## 11. UX States

Every page should handle:

```text
empty state
loading/running state
success state
failed state
partial artifacts state
```

Examples:

- Job list empty state: "No jobs yet. Create your first distillation job."
- Job detail failed state: show stage, error message, and logs/artifacts if available.
- Results partial state: if MLflow registration failed but artifacts exist, still show artifacts.
- Prompt unavailable state: if job is not completed, show current stage and status.

---

## 12. Error Handling

UI routes should not expose raw stack traces.

Error display should include:

```text
human-readable message
job ID if available
stage if available
correlation/request ID if available
suggested next action
```

For validation errors, show precise file/line references when possible.

For failed jobs, show:

- failed stage
- error summary
- whether artifacts are available
- whether MLflow run was created
- whether retry is supported

Retry is optional and out of scope for MVP unless the backend already supports it.

---

## 13. Security and Privacy

MVP assumptions:

- UI is intended for local/internal use.
- No authentication in MVP unless already provided by deployment environment.
- Do not expose provider secrets.
- Do not render secrets from settings snapshots.
- Do not include raw provider API keys in hidden form fields.
- Use server-side provider profile resolution only.

Upload safety:

- Restrict upload extensions to `.yaml`, `.yml`, and `.jsonl`.
- Enforce max upload sizes from settings.
- Do not execute uploaded content.
- Treat uploaded task/case content as data.
- Escape all rendered content in templates.
- Render prompt and JSON artifacts inside escaped `<pre>` blocks.

---

## 14. Styling

Use a minimal, functional design.

Recommended layout:

```text
left/top navigation:
  Jobs
  New Job
  docs/Artifacts link later

main content:
  cards for status and metrics
  tables for jobs/failures/rules
  pre blocks for prompt/artifacts
```

Status colors:

```text
queued: gray
running: blue
completed: green
failed: red
needs_review: amber
```

Keep CSS simple.

Do not spend significant time on visual polish until the pipeline workflow is proven.

---

## 15. MLflow Link Construction

Settings should include:

```text
MLFLOW_TRACKING_URI
MLFLOW_UI_BASE_URL optional
```

If `MLFLOW_UI_BASE_URL` is set, construct run links from it.

If not set, show the run ID and tracking URI, but do not attempt to guess a run URL.

Example display:

```text
MLflow Run ID: abc123
Tracking URI: http://localhost:5000
[Open MLflow Run]
```

If direct run URL construction is unreliable, link to the MLflow root UI and show the run ID for search.

---

## 16. Artifact Downloading

Artifact download routes should stream files from the configured artifact root.

Suggested route:

```http
GET /ui/jobs/{job_id}/artifacts/download?path=outputs/selected_distilled_prompt.md
```

Rules:

- Only allow paths under `.rulekiln/runs/{job_id}/`.
- Reject absolute paths.
- Reject `..` path traversal.
- Use a whitelist or manifest lookup when possible.
- Set safe content type:
  - `.md`: `text/markdown`
  - `.json`: `application/json`
  - `.jsonl`: `application/x-ndjson`
  - `.yaml`: `application/yaml`
  - `.txt`: `text/plain`

---

## 17. Backend Reuse

The UI must reuse existing services.

Do not duplicate:

- job creation logic
- provider resolution
- task/case validation
- pipeline start logic
- artifact lookup logic
- MLflow URL/run lookup logic

Preferred pattern:

```text
UI route
  -> form parser
  -> existing validation service
  -> existing job service
  -> redirect/render
```

---

## 18. Testing Strategy

Add tests for:

```text
GET /ui/jobs renders empty state
GET /ui/jobs/new renders upload form
POST /ui/jobs/preview rejects invalid task.yaml
POST /ui/jobs/preview rejects invalid cases.jsonl
POST /ui/jobs creates job with fake providers
GET /ui/jobs/{job_id} renders running status
GET /ui/jobs/{job_id}/status-fragment renders terminal completed state without polling
GET /ui/jobs/{job_id}/results renders selected strategy and metric delta
GET /ui/jobs/{job_id}/artifacts rejects path traversal
```

All UI tests must run offline with fake providers.

No UI test should call external model providers.

---

## 19. Implementation Tasks

Add a new phase to the MVP task file.

### Phase 4.5: Minimal Operator UI

Purpose:

Make RuleKiln usable through a small server-rendered UI without introducing a separate frontend application.

Tasks:

```text
T060 Add Jinja2, python-multipart, and aiofiles dependencies
T061 Add template/static mounting in FastAPI app
T062 Create UI route module in src/rulekiln/ui/routes.py
T063 Create base.html and basic layout templates
T064 Create /ui/jobs page listing recent jobs
T065 Create /ui/jobs/new upload form for task.yaml and cases.jsonl
T066 Implement provider/model selection using configured provider profiles
T067 Implement /ui/jobs/preview validation flow
T068 Submit validated upload into existing DistillationRequest and create job
T069 Create /ui/jobs/{job_id} detail page
T070 Add HTMX polling status fragment
T071 Create results summary page with selected strategy and quality gates
T072 Create read-only selected prompt view
T073 Create read-only synthesized rules view
T074 Create eval report view
T075 Create failures fixed/broken view
T076 Create artifact manifest and download links
T077 Add MLflow run link display
T078 Add UI tests using fake providers
T079 Update README with UI usage instructions
```

---

## 20. Acceptance Criteria

The MVP UI is acceptable when:

1. A user can open `/ui/jobs/new`.
2. A user can upload `task.yaml` and `cases.jsonl`.
3. The UI validates the files before job submission.
4. The UI shows split counts and provider route validation.
5. A user can submit a valid job.
6. A user can watch job progress without refreshing manually.
7. A user can see whether the selected prompt improved over baseline.
8. A user can see DBSCAN vs HDBSCAN scores.
9. A user can view the selected prompt.
10. A user can view synthesized rules.
11. A user can view fixed and broken failures.
12. A user can download artifacts.
13. A user can open the MLflow run or see the MLflow run ID.
14. All UI tests pass offline with fake providers.
15. The UI does not require Next.js, Node, or external model credentials.

---

## 21. Future UI Enhancements

Do not implement these in MVP, but keep the architecture open for them:

- authentication
- user accounts
- workspaces
- prompt diff viewer
- rule diff viewer
- case-level annotation
- manual rule editing
- manual prompt approval workflow
- retry failed job from UI
- run comparison UI
- richer charts
- WebSocket progress updates
- Next.js/React frontend
- hosted SaaS dashboard
- RBAC and audit logs
- scheduled regression runs

---

## 22. Final Recommendation

Build the minimal UI with FastAPI, Jinja, HTMX, and Tailwind.

Keep MLflow as a separate audit/experiment UI.

The MVP UI should make RuleKiln easy to operate, not turn it into a full product dashboard.

The first useful version is:

```text
Upload -> Validate -> Run -> Watch -> Review -> Download -> Open MLflow
```

That is enough to reduce friction without creating a second application to maintain.
