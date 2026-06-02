<p align="center">
  <img src="rulekilnlogo.png" alt="RuleKiln" width="320" />
</p>

# RuleKiln

> **Stable Alpha** — RuleKiln has reached a stable alpha milestone. Core pipeline and Operator UI workflows are stable for internal pilots and evaluation. APIs, configuration, and behaviour may still evolve before beta. Not recommended for production use.

A prompt compiler that turns labelled cases into tested, versioned, auditable system prompts.

RuleKiln runs a full distillation pipeline: it extracts micro-rules from your cases, clusters them into coherent groups, synthesises rule sets, reviews each rule for logical conflicts, prunes the rule set to fit token and quality budgets, compiles deterministic prompts, evaluates both DBSCAN and HDBSCAN strategies against your cases, applies quality gates, selects the best strategy, and surfaces the winner as a ready-to-use system prompt with a full MLflow audit trail.

---


## Why RuleKiln Exists

RuleKiln is built around a simple idea:

> Use an expensive teacher model during prompt-hardening, then let cheaper student models run the task in production.

The teacher model may still be an online API such as Claude, GPT, Gemini, or a Bedrock-hosted model. The key distinction is **when** that model is used.

Instead of calling the expensive model for every production request, RuleKiln uses it during a build-time distillation process:

1. The teacher reviews task cases.
2. It extracts reusable task-policy rules.
3. RuleKiln clusters, resolves, and prunes those rules.
4. RuleKiln compiles them into prompt candidates.
5. One or more student models are graded against the same cases.
6. The best prompt/student combination is selected only if it passes quality gates.

The result is a reusable prompt artifact — a “lesson plan” — that can be deployed with a smaller, cheaper, or local student model.

This does not try to make a 4B or 8B model generally as capable as a frontier model. Instead, RuleKiln asks a narrower and more practical question:

> Can this smaller model perform this specific task well enough when given a distilled, task-specific prompt?

That makes RuleKiln useful for workflows such as transcript review, support-ticket routing, structured extraction, policy checks, summarization, and other repeatable tasks with measurable evaluation cases.

The economic thesis is:

> Pay the expensive model to teach during prompt hardening. Let cheaper students execute many times.

For high-volume workflows, this can turn expensive runtime reasoning into a reusable, auditable prompt-building step.

---

## Edge and On-Device Models

One of RuleKiln's strongest deployment targets is small models running locally, offline, or at the edge.

Many teams want AI tasks to run on phones, laptops, embedded devices, or private local servers, but smaller models often struggle with instruction-following, structured output, and task-specific edge cases. RuleKiln is designed to help with that gap.

The workflow is:

1. Use a stronger cloud teacher model during prompt hardening.
2. Extract and synthesize task-specific rules from labelled cases.
3. Prune those rules to fit a prompt/token budget.
4. Compile a compact prompt for a smaller student model.
5. Evaluate the student before and after hardening.
6. Promote the prompt only if it passes quality gates.

This makes RuleKiln especially useful for edge-oriented use cases such as:

- on-device intent classification
- offline form or document extraction
- local transcript tagging after speech-to-text
- support-ticket routing
- field technician troubleshooting
- device diagnostics
- private summarization and checklist review
- structured extraction where data should not leave the device

RuleKiln does not make a small model generally as capable as a frontier model. Instead, it asks a narrower deployment question:

> Can this edge model perform this specific task well enough when given a compact, distilled prompt?

For edge deployments, the best prompt is not always the highest-scoring prompt. It is the prompt that satisfies the full deployment budget:

- quality score is high enough
- malformed output rate is low enough
- regression rate is acceptable
- prompt length fits the target context window
- latency is acceptable for the device
- the model can run locally without calling a cloud LLM at runtime

Future RuleKiln deployment profiles may make this explicit:

```yaml
deployment_profile:
  type: edge
  target_runtime: phone
  max_prompt_tokens: 1200
  max_context_tokens: 4096
  max_rules: 20
  max_latency_ms: 1500
  require_json_output: true
  max_malformed_output_rate: 0.01
  max_regression_rate: 0.05
```

The long-term goal is not only to produce a better prompt, but to produce a deployable prompt:

```text
frontier teacher at build time
small student at runtime
auditable rules
compact prompt
before/after evals
quality gates
```

That makes RuleKiln a practical prompt-hardening layer for local and edge AI systems.

---

## Quick start

Canonical setup path for both local workflows: [docs/dev/docker.md](docs/dev/docker.md).

### Option A — Native Python

**Requirements**: Python 3.13+, [uv](https://docs.astral.sh/uv/), PostgreSQL 14+, MLflow server

```bash
# 1. Clone and install
git clone https://github.com/gblackard-gvtx/RuleKiln.git
cd RuleKiln
uv sync --extra dev

# 2. Configure environment
cp .env.example .env
# Edit .env — set DATABASE_URL, MLFLOW_TRACKING_URI, and provider profiles

# 3. Run database migrations
uv run alembic upgrade head

# 4. Start the API
uv run uvicorn src.rulekiln.api.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Start the worker (separate terminal — required)
uv run python -m rulekiln.workers.dbos_worker
# or, if installed as a script:
uv run rulekiln-worker
```

API docs available at <http://localhost:8000/docs>.

### Option B — Docker Compose

**Requirements**: Docker Desktop or Docker Engine with Compose v2

```bash
# 1. Clone
git clone https://github.com/gblackard-gvtx/RuleKiln.git
cd RuleKiln

# 2. Start the stack (copies .env.example → .env automatically on first run)
./scripts/dev-up.sh
```

This starts:
| Service | URL |
|---------|-----|
| RuleKiln API | <http://localhost:8010> |
| OpenAPI docs | <http://localhost:8010/docs> |
| MLflow UI | <http://localhost:5000> |
| PostgreSQL | `localhost:5432` |
| DBOS worker | *(background process, no UI)* |

```bash
# Stop the stack
./scripts/dev-down.sh
```

### Post-setup smoke test (both options)

```bash
DATABASE_URL="sqlite+aiosqlite://" \
MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" \
uv run pytest -m "not external" --tb=short -q
```

---

## Environment configuration

Copy `.env.example` to `.env` and adjust values. The key variables are:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL async DSN (`postgresql+asyncpg://...`) |
| `MLFLOW_TRACKING_URI` | MLflow server URI or `file:///path/to/local` |
| `MLFLOW_ALLOWED_HOSTS` | Optional MLflow server host allowlist. Include browser Host header forms (for example, `localhost:5000`, `127.0.0.1:5000`) to avoid MLflow "Invalid Host header" rejections. |
| `ARTIFACT_ROOT` | Local path for job artifact output (default: `.rulekiln/runs`) |
| `ENABLE_PGVECTOR` | Enable pgvector for embedding storage (default: `false`) |
| `EXECUTION_BACKEND` | `dbos` (only supported value) — see [Execution backend](#execution-backend) |
| `WORKER_POLL_INTERVAL_SECONDS` | How often the queue worker polls for new jobs (default: `2`) |
| `WORKER_LEASE_SECONDS` | Job lease duration in seconds before a crashed worker's job is reclaimed (default: `1800`) |
| `WORKER_RETRY_BACKOFF_SECONDS` | Backoff delay before retrying a retryable failed job (default: `30`) |
| `WORKER_MAX_ATTEMPTS` | Maximum queue claim attempts before a retryable job is marked `failed_retryable` (default: `2`) |
| `DEFAULT_PROVIDER_MAX_CONCURRENCY` | Max concurrent in-flight calls per provider config (default: `3`) |
| `DEFAULT_PROVIDER_RATE_LIMIT_RPM` | Global default requests-per-minute cap (default: unset) |
| `DEFAULT_PROVIDER_RATE_LIMIT_TPM` | Global default tokens-per-minute cap (default: unset) |
| `DEFAULT_MAX_RULES` | Maximum synthesised rules included in a compiled prompt (default: `40`) |
| `DEFAULT_MIN_RULE_SUPPORT_COUNT` | Minimum case-support threshold for a rule to survive pruning (default: `2`) |
| `DEFAULT_MAX_PROMPT_TOKENS` | Approximate token budget for the compiled rule policy section (default: `8000`) |

### Execution backend

RuleKiln supports a single execution mode, controlled by `EXECUTION_BACKEND`:

| Backend | Default | Separate worker | `POST /v1/jobs/` status | Retry semantics | Stage coverage (current) | Worker command |
|---------|---------|-----------------|--------------------------|-----------------|--------------------------|----------------|
| `dbos` | Yes | Yes | `pending` | Classified retry policy with `waiting_for_retry`, then `failed_retryable` or `failed_terminal` when attempts are exhausted or error is terminal | Full pipeline stage chain | `uv run rulekiln-worker` (or `uv run rulekiln-dbos-worker`) |

Manual retry from the Operator UI (`Retry Pipeline`) requeues the same job record for failed statuses (`failed`, `failed_terminal`, `failed_retryable`) and resumes from persisted progress.

Granular resume coverage for costly stages:

- `extracting_rules`: per case
- `synthesizing_rules`: per cluster
- `reviewing_rule_conflicts`: per synthesized rule
- `evaluating_baseline` / `evaluating_distilled`: per case

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

Per-profile timeout and provider-call retries (defaults: `timeout_seconds=120`, `max_retries=3`):
```env
PROVIDER_PROFILES__OPENAI_DEFAULT__TIMEOUT_SECONDS=120
PROVIDER_PROFILES__OPENAI_DEFAULT__MAX_RETRIES=3
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
    "schema_version": "rulekiln.task.v1",
    "task_id": "intent-router",
    "task_name": "Intent Router",
    "task_mode": "classification",
    "description": "Route user intent to one label.",
    "input_template": "{{input.user_message}}"
  },
  "cases": [
    {
      "schema_version": "rulekiln.case.v1",
      "id": "c1",
      "split": "train",
      "task_mode": "classification",
      "input": {"user_message": "Book a flight"},
      "expected": {"label": "travel"}
    },
    {
      "schema_version": "rulekiln.case.v1",
      "id": "c2",
      "split": "validation",
      "task_mode": "classification",
      "input": {"user_message": "I need to change my reservation"},
      "expected": {"label": "travel"}
    }
  ],
  "teacher": {"provider_profile": "openai_default", "model": "gpt-4o"},
  "student": {"provider_profile": "openai_default", "model": "gpt-4o-mini"},
  "embedding": {"provider_profile": "openai_default", "model": "text-embedding-3-small"}
}
```

Response `202 Accepted` (`dbos` mode):
```json
{"job_id": "...", "status": "pending"}
```

### Poll job status

```http
GET /v1/jobs/{job_id}
```

```json
{"job_id": "...", "status": "running", "stage": "reviewing_rule_conflicts", "error_message": null}
```

Common status values: `pending`, `running`, `waiting_for_retry`, `failed_retryable`, `failed_terminal`, `completed`.

Full pipeline stage order (`dbos`): `validating_project` → `extracting_rules` → `embedding_rules` → `clustering_rules` → `synthesizing_rules` → `reviewing_rule_conflicts` → `pruning_rules` → `compiling_prompts` → `evaluating_baseline` → `evaluating_distilled` → `selecting_strategy` → `analyzing_failures` → `checking_quality_gates` → `logging_artifacts` → `exporting_artifacts` → `completed`.

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
  - Split policy is centralized: extraction uses `train`; evaluation prefers `validation`, then falls back to `train`, `test`, or `golden`.
  - When fallback is used, preview surfaces a warning before submission.
3. **Run Pipeline** — submit the validated job; execution is delegated by `EXECUTION_BACKEND` (`dbos` queue + worker).
4. **Monitor** — the job detail page polls live status every 2 seconds via HTMX until the job finishes.
  - Job detail includes split totals, execution progress (`teacher extraction`, `student eval` per strategy), and pipeline diagnostics (model-call counts and rule counts).
5. **Review results** — navigate to Results, Prompt, Rules, Eval Report, Failures, or Artifacts from the detail page.
  - Results includes recommendation metrics: **Best strategy**, **Baseline macro_f1**, **Relative lift**, and **Accuracy lift**.
  - Eval Report displays an evaluation-split fallback banner when non-validation evaluation was used.
6. **Retry failed jobs** — use **Retry Pipeline** on the job detail page to requeue and resume from persisted progress.

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

## Phase 2 strategy expansion

RuleKiln now evaluates a wider strategy set beyond the legacy baseline plus DBSCAN and HDBSCAN.

- Baseline scaffold: `baseline_scaffold`
- Deterministic few-shot baselines: `baseline_few_shot_k3`, `baseline_few_shot_k5`
- Embedding-only baselines: `embedding_centroid`, `embedding_knn_k1`, `embedding_knn_k3`, `embedding_knn_k5`
- Retrieval few-shot baseline: `retrieval_few_shot_k5`
- Distilled strategies: `dbscan`, `hdbscan`

Few-shot prompts are assembled deterministically by pipeline code from training examples. They are not authored by teacher or student models at runtime.

For implementation and operator details, see [docs/dev/phase2.md](docs/dev/phase2.md).

---

## Artifact layout

Each completed job writes its outputs under `.rulekiln/runs/{job_id}/`:

```
.rulekiln/runs/{job_id}/
  task.yaml
  cases.normalized.jsonl
  outputs/
    baseline_prompt.md
    baseline_scaffold_prompt.md
    baseline_few_shot_k3_prompt.md
    baseline_few_shot_k5_prompt.md
    distilled_prompt_dbscan.md
    distilled_prompt_hdbscan.md
    selected_distilled_prompt.md
    baseline_scaffold_eval.json
    baseline_few_shot_k3_eval.json
    baseline_few_shot_k5_eval.json
    embedding_centroid_eval.json
    embedding_knn_k1_eval.json
    embedding_knn_k3_eval.json
    embedding_knn_k5_eval.json
    retrieval_few_shot_k5_eval.json
    dbscan_eval.json
    hdbscan_eval.json
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

`strategy_comparison.json` is the source of truth for the evaluated strategy set (`strategy_evals`, `strategy_gates`, `strategy_prompt_tokens`, `strategy_metadata`) and selected winner.

---

## Benchmark examples

- [Examples index](examples/README.md) — overview of non-runtime benchmark assets.
- [BANKING77 benchmark README](examples/datasets/banking77/README.md) — benchmark setup, initial results, reporting template, and root README snapshot format.

---

## Development

Use Make command aliases for a single command surface, or run the raw commands directly.

| Target | Runs | Typical use |
|--------|------|-------------|
| `make lint` | `uv run ruff check src/ tests/` | Verify lint before commit/CI |
| `make format` | `uv run ruff format src/ tests/` | Apply repo formatting |
| `make typecheck` | `uv run pyright` | Verify static typing |
| `make test` | Offline non-external test suite | Main local validation path |
| `make test-ui` | Offline UI test subset | Validate UI routes/templates |
| `make ci-local` | `lint + typecheck + test` | One-command pre-push check |
| `make docker-up` | `./scripts/dev-up.sh` | Start local Docker stack |
| `make docker-down` | `./scripts/dev-down.sh` | Stop local Docker stack |
| `make benchmark-smoke` | Dataset presence checks | Quick benchmark fixture sanity |

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

See [docs/dev/docker.md](docs/dev/docker.md) for the full Docker Compose development guide.
See [docs/dev/phase2.md](docs/dev/phase2.md) for detailed Phase 2 strategy and artifact documentation.
See [docs/reference/python_module_reference.md](docs/reference/python_module_reference.md) for a full module-by-module model/function reference.
See [docs/README.md](docs/README.md) for the full documentation map and canonical task/spec pointers.

---

## Contributing

See `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` before contributing.

---

## Research Inspiration

RuleKiln is inspired by recent work on prompt-level distillation, including ["Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning"](https://doi.org/10.48550/arXiv.2602.21103) by Sanket Badhe and Deep Shah.

RuleKiln is an independent implementation and product design. It does not copy or include the paper's text, figures, datasets, prompts, or code. RuleKiln extends the core idea with a case-first workflow, provider-neutral teacher/student routing, classroom-style multi-student grading, rule provenance, conflict review, pruning, eval-to-rule mapping, artifact exports, and operational tooling.
