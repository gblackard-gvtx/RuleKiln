<p align="center">
  <img src="rulekilnlogo.png" alt="RuleKiln" width="320" />
</p>

# RuleKiln

> **⚠️ Work in Progress** — This project is under active development. The `main` branch is not yet stable or fully functional. APIs, configuration, and behaviour may change without notice. Not recommended for production use.

A prompt compiler that turns labelled cases into tested, versioned, auditable system prompts.

RuleKiln runs a full distillation pipeline: it extracts micro-rules from your cases, clusters them into coherent groups, synthesises rule sets, reviews each rule for logical conflicts, prunes the rule set to fit token and quality budgets, compiles deterministic prompts, evaluates both DBSCAN and HDBSCAN strategies against your cases, applies quality gates, selects the best strategy, and surfaces the winner as a ready-to-use system prompt with a full MLflow audit trail.

---

## Quick start

### Option A — Native Python

**Requirements**: Python 3.13+, [uv](https://docs.astral.sh/uv/), PostgreSQL 14+, MLflow server

```bash
# 1. Clone and install
git clone https://github.com/your-org/rulekiln.git
cd rulekiln
uv sync --extra dev

# 2. Configure environment
cp .env.example .env
# Edit .env — set DATABASE_URL, MLFLOW_TRACKING_URI, and provider profiles

# 3. Run database migrations
uv run alembic upgrade head

# 4. Start the API
uv run uvicorn src.rulekiln.api.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Start the queue worker (separate terminal — required for postgres_queue mode)
uv run python -m rulekiln.workers.queue_worker
# or, if installed as a script:
uv run rulekiln-worker
```

API docs available at <http://localhost:8000/docs>.

### Option B — Docker Compose

**Requirements**: Docker Desktop or Docker Engine with Compose v2

```bash
# 1. Clone
git clone https://github.com/your-org/rulekiln.git
cd rulekiln

# 2. Start the stack (copies .env.example → .env automatically on first run)
./scripts/dev-up.sh
```

This starts:
| Service | URL |
|---------|-----|
| RuleKiln API | <http://localhost:8000> |
| OpenAPI docs | <http://localhost:8000/docs> |
| MLflow UI | <http://localhost:5000> |
| PostgreSQL | `localhost:5432` |
| Queue worker | *(background process, no UI)* |

```bash
# Stop the stack
./scripts/dev-down.sh
```

---

## Environment configuration

Copy `.env.example` to `.env` and adjust values. The key variables are:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL async DSN (`postgresql+asyncpg://...`) |
| `MLFLOW_TRACKING_URI` | MLflow server URI or `file:///path/to/local` |
| `ARTIFACT_ROOT` | Local path for job artifact output (default: `.rulekiln/runs`) |
| `ENABLE_PGVECTOR` | Enable pgvector for embedding storage (default: `false`) |
| `EXECUTION_BACKEND` | `postgres_queue` (default) or `background_tasks` — see [Execution backends](#execution-backends) |
| `WORKER_POLL_INTERVAL_SECONDS` | How often the queue worker polls for new jobs (default: `2`) |
| `WORKER_LEASE_SECONDS` | Job lease duration in seconds before a crashed worker's job is reclaimed (default: `1800`) |
| `DEFAULT_PROVIDER_MAX_CONCURRENCY` | Max concurrent in-flight calls per provider config (default: `3`) |
| `DEFAULT_PROVIDER_RATE_LIMIT_RPM` | Global default requests-per-minute cap (default: unset) |
| `DEFAULT_PROVIDER_RATE_LIMIT_TPM` | Global default tokens-per-minute cap (default: unset) |
| `DEFAULT_MAX_RULES` | Maximum synthesised rules included in a compiled prompt (default: `40`) |
| `DEFAULT_MIN_RULE_SUPPORT_COUNT` | Minimum case-support threshold for a rule to survive pruning (default: `2`) |
| `DEFAULT_MAX_PROMPT_TOKENS` | Approximate token budget for the compiled rule policy section (default: `8000`) |

### Execution backends

RuleKiln supports two job execution modes, controlled by `EXECUTION_BACKEND`:

| Mode | Description |
|------|-------------|
| `postgres_queue` | **(default)** Jobs are queued in PostgreSQL. The `worker` process polls, claims, and runs them with lease-based crash recovery and automatic retry (up to `max_attempts`). Scale by running multiple workers. |
| `background_tasks` | Jobs run in the FastAPI process via `BackgroundTasks`. Simpler for local dev or single-server deploys. No separate worker needed, but no crash recovery or retry. |

In `postgres_queue` mode, `POST /v1/jobs/` returns `status: "pending"` and the API returns immediately. The worker picks the job up within `WORKER_POLL_INTERVAL_SECONDS`.

In `background_tasks` mode, the job starts in the same process and `POST /v1/jobs/` returns `status: "created"`.

### Provider profiles

Provider profiles are configured via environment variables with the `PROVIDER_PROFILES__<NAME>__` prefix. Each profile exposes a named provider route used by teacher, student, and embedding roles in the distillation request.

**Fake provider** (offline / CI — no credentials needed):
```env
PROVIDER_PROFILES__FAKE__PROVIDER=fake
PROVIDER_PROFILES__FAKE__SUPPORTS_CHAT=true
PROVIDER_PROFILES__FAKE__SUPPORTS_EMBEDDINGS=true
```

**OpenAI**:
```env
PROVIDER_PROFILES__OPENAI_DEFAULT__PROVIDER=openai
PROVIDER_PROFILES__OPENAI_DEFAULT__SUPPORTS_CHAT=true
PROVIDER_PROFILES__OPENAI_DEFAULT__SUPPORTS_EMBEDDINGS=true
OPENAI_API_KEY=sk-...
```

Per-profile rate limiting (applied after `DEFAULT_PROVIDER_RATE_LIMIT_*` defaults, and overridden by per-request `ModelRoute` values):
```env
PROVIDER_PROFILES__OPENAI_DEFAULT__RATE_LIMIT_RPM=60
PROVIDER_PROFILES__OPENAI_DEFAULT__RATE_LIMIT_TPM=100000
PROVIDER_PROFILES__OPENAI_DEFAULT__MAX_CONCURRENCY=5
```

**OpenAI-compatible** (Ollama, vLLM, LiteLLM proxy):
```env
PROVIDER_PROFILES__LOCAL__PROVIDER=openai_compatible
PROVIDER_PROFILES__LOCAL__BASE_URL=http://localhost:11434/v1
PROVIDER_PROFILES__LOCAL__SUPPORTS_CHAT=true
PROVIDER_PROFILES__LOCAL__SUPPORTS_EMBEDDINGS=true
```

**Amazon Bedrock**:
```env
PROVIDER_PROFILES__BEDROCK_PRIMARY__PROVIDER=bedrock
PROVIDER_PROFILES__BEDROCK_PRIMARY__REGION=us-east-1
PROVIDER_PROFILES__BEDROCK_PRIMARY__SUPPORTS_CHAT=true
```

**Anthropic**:
```env
PROVIDER_PROFILES__ANTHROPIC_DEFAULT__PROVIDER=anthropic
PROVIDER_PROFILES__ANTHROPIC_DEFAULT__SUPPORTS_CHAT=true
ANTHROPIC_API_KEY=sk-ant-...
```

**Google Vertex Gemini** *(stub — not yet implemented)*:
```env
PROVIDER_PROFILES__VERTEX_DEFAULT__PROVIDER=vertex_gemini
PROVIDER_PROFILES__VERTEX_DEFAULT__SUPPORTS_CHAT=true
```

**Azure OpenAI** *(stub — not yet implemented)*:
```env
PROVIDER_PROFILES__AZURE_DEFAULT__PROVIDER=azure_openai
PROVIDER_PROFILES__AZURE_DEFAULT__BASE_URL=https://<resource>.openai.azure.com/
PROVIDER_PROFILES__AZURE_DEFAULT__SUPPORTS_CHAT=true
```

Stub providers raise `ProviderNotImplementedError` immediately — they never silently fall back to another provider.

### Default quality gates

Override any gate threshold via environment:

```env
DEFAULT_QUALITY_GATE__MIN_METRIC_DELTA=0.0
DEFAULT_QUALITY_GATE__MAX_REGRESSION_RATE=0.10
DEFAULT_QUALITY_GATE__MAX_GOLDEN_FAILURES=0
DEFAULT_QUALITY_GATE__MAX_MALFORMED_OUTPUT_RATE=0.01
DEFAULT_QUALITY_GATE__MAX_PROMPT_TOKENS=8000
DEFAULT_QUALITY_GATE__REQUIRE_HUMAN_APPROVAL=true
```

Per-task overrides in `task.quality_gates` take precedence. Per-task rule budget settings (`max_rules`, `max_prompt_tokens`, `min_rule_support_count`, `preserve_golden_rules`) are set inside the `task` object of the distillation request and override the corresponding `DEFAULT_*` environment values.

---

## API

### Submit a distillation job

```http
POST /v1/jobs/
Content-Type: application/json

{
  "task": {
    "task_id": "intent-router",
    "task_name": "Intent Router",
    "task_mode": "classification"
  },
  "cases": [
    {
      "case_id": "c1",
      "input": {"user_message": "Book a flight"},
      "expected_output": {"label": "travel"}
    }
  ],
  "teacher": {"provider_profile": "openai_default", "model": "gpt-4o"},
  "student": {"provider_profile": "openai_default", "model": "gpt-4o-mini"},
  "embedding": {"provider_profile": "openai_default", "model": "text-embedding-3-small"}
}
```

Response `202 Accepted` (`postgres_queue` mode):
```json
{"job_id": "...", "status": "pending"}
```

Response `202 Accepted` (`background_tasks` mode):
```json
{"job_id": "...", "status": "created"}
```

### Poll job status

```http
GET /v1/jobs/{job_id}
```

```json
{"job_id": "...", "status": "running", "stage": "reviewing_rule_conflicts", "error_message": null}
```

Pipeline stages in order: `validating_project` → `extracting_rules` → `embedding_rules` → `clustering_rules` → `synthesizing_rules` → `reviewing_rule_conflicts` → `pruning_rules` → `compiling_prompts` → `evaluating_baseline` → `evaluating_distilled` → `selecting_strategy` → `analyzing_failures` → `checking_quality_gates` → `logging_artifacts` → `exporting_artifacts` → `completed`.

### Retrieve outputs (once `status == "completed"`)

```http
GET /v1/jobs/{job_id}/prompt        # selected system prompt
GET /v1/jobs/{job_id}/rules         # synthesised rules
GET /v1/jobs/{job_id}/eval-report   # evaluation metrics
```

---

## Operator UI

A lightweight server-rendered UI for managing distillation jobs without the API. Start the server as usual, then open:

```
http://localhost:8000/ui/jobs/new
```

### Workflow

1. **New Job** — upload a `task.yaml` and `cases.jsonl`, choose provider profiles and model IDs.
2. **Preview** — validate files, review split counts, estimated API calls, and provider routes before committing.
3. **Run Pipeline** — submit the validated job; the pipeline runs as a background task.
4. **Monitor** — the job detail page polls live status every 2 seconds via HTMX until the job finishes.
5. **Review results** — navigate to Results, Prompt, Rules, Eval Report, Failures, or Artifacts from the detail page.

### Environment variables for the UI

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Required — same as the API |
| `MLFLOW_TRACKING_URI` | Required — same as the API |
| `PROVIDER_PROFILES__*` | Required — define at least one chat and one embedding profile |
| `MLFLOW_UI_BASE_URL` | Optional — e.g. `http://localhost:5000`. Enables direct links to MLflow runs from the job detail page. |

### Running UI tests

```bash
DATABASE_URL="sqlite+aiosqlite://" \
MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" \
uv run pytest tests/ui/ --tb=short -q
```

---

## Tests

```bash
# All offline tests (default CI path — no external credentials)
DATABASE_URL="sqlite+aiosqlite://" \
MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" \
uv run pytest -m "not external" --tb=short -q

# External provider smoke tests (requires real credentials)
uv run pytest -m external
```

---

## Artifact layout

Each completed job writes its outputs under `.rulekiln/runs/{job_id}/`:

```
.rulekiln/runs/{job_id}/
  task.yaml
  cases.normalized.jsonl
  outputs/
    distilled_prompt_dbscan.md
    distilled_prompt_hdbscan.md
    selected_distilled_prompt.md
    rules_dbscan.jsonl
    rules_hdbscan.jsonl
    eval_report.json
    strategy_comparison.json
    failures_fixed.jsonl
    failures_broken.jsonl
    failures_unchanged.jsonl
  exports/
    promptfoo.yaml
    mlflow_run_id.txt
  metadata/
    settings_snapshot.json
    manifest.json
```

---

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run pyright

# Run migrations (native)
uv run alembic upgrade head
```

See [Docs/dev/docker.md](Docs/dev/docker.md) for the full Docker Compose development guide.
See [Docs/reference/python_module_reference.md](Docs/reference/python_module_reference.md) for a full module-by-module model/function reference.

---

## Citation

RuleKiln implements the **Prompt-Level Distillation (PLD)** method introduced in:

> Sanket Badhe, Deep Shah.  
> **Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning.**  
> arXiv:2602.21103 [cs.CL], February 2026.  
> <https://doi.org/10.48550/arXiv.2602.21103>

```bibtex
@misc{badhe2026promptleveldistillationnonparametric,
  title   = {Prompt-Level Distillation: A Non-Parametric Alternative to
             Model Fine-Tuning for Efficient Reasoning},
  author  = {Sanket Badhe and Deep Shah},
  year    = {2026},
  eprint  = {2602.21103},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://doi.org/10.48550/arXiv.2602.21103}
}
```
