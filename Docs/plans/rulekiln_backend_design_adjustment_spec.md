# RuleKiln Backend Design Adjustment Spec

> **Status:** Historical/archival design document. It references legacy `BackgroundTasks` and `postgres_queue` paths that are no longer runtime-supported.

## 1. Purpose

This spec defines a targeted backend adjustment for RuleKiln after the initial MVP backend has already been implemented.

The goal is to harden the existing backend without rewriting the system.

The adjustment addresses five engineering risks:

1. In-process `BackgroundTasks` are unsafe for long-running distillation jobs.
2. Synthesized rules may contain unresolved contradictions.
3. Compiled prompts may become too large and degrade student-model performance.
4. Evaluation artifacts need to map failures back to rules and assertions.
5. Provider calls need rate limiting to avoid enterprise quota failures.

This is an incremental design patch. It should preserve the existing API and UI behavior wherever possible.

---

## 2. Senior Engineering Decision Summary

Adopt the following changes:

| Area | Decision |
|---|---|
| Job execution | Replace FastAPI `BackgroundTasks` with a lightweight Postgres-backed job queue using `FOR UPDATE SKIP LOCKED`. |
| Conflict handling | Add one-pass conflict detection and resolution before prompt compilation. Full iterative conflict loop is deferred. |
| Rule pruning | Add rule prioritization/pruning before prompt compilation to control prompt bloat. |
| Eval traceability | Map evaluation failures to `matched_rule_ids`, `violated_rule_ids`, and `failed_assertion_paths`. |
| Provider safety | Add per-provider RPM/TPM/concurrency limits and enforce them in worker execution. |

Do **not** introduce Celery, Redis, Temporal, or a full orchestration platform for this adjustment unless the Postgres queue becomes insufficient.

---

## 3. Non-Goals

This adjustment does not include:

- replacing the full pipeline architecture
- changing the public `POST /distillation-jobs` request shape
- requiring a separate queue service
- adding Redis
- adding Celery
- adding Temporal
- adding DBOS
- adding a full iterative conflict-resolution loop
- adding rule editing UI
- adding failure-review database tables unless later needed
- requiring pgvector
- requiring MLflow Prompt Registry
- building a distributed multi-worker scheduler beyond a simple Postgres queue

---

## 4. Current Backend Assumptions

The current backend already has most or all of:

- FastAPI app
- `POST /distillation-jobs`
- `GET /distillation-jobs/{job_id}`
- task/case schemas
- provider abstraction
- fake/offline providers
- rule extraction
- embeddings
- DBSCAN/HDBSCAN clustering
- synthesis
- prompt compilation
- evaluation
- quality gates
- artifacts
- MLflow logging
- Docker/local dev setup
- optional UI layer

This spec changes internals but should keep the operator-facing workflow the same:

```text
Upload / submit task and cases
  -> create job
  -> watch progress
  -> review prompt/rules/eval/failures
  -> download artifacts
  -> open MLflow run
```

---

## 5. Migration Strategy

Use a phased migration:

```text
Phase A: Add schema columns and queue worker while keeping existing API.
Phase B: Route new jobs through Postgres queue instead of BackgroundTasks.
Phase C: Add provider rate limiting.
Phase D: Add conflict review and rule pruning.
Phase E: Add eval-to-rule failure mapping.
Phase F: Remove or disable BackgroundTasks path after stability.
```

During migration, keep the old path behind a feature flag:

```text
EXECUTION_BACKEND=background_tasks | postgres_queue
```

Default after this change:

```text
EXECUTION_BACKEND=postgres_queue
```

---

## 6. Execution Backend: Postgres Job Queue

### 6.1 Problem

FastAPI `BackgroundTasks` runs in the API process. A pod restart, process crash, deploy, OOM, or worker timeout can orphan long-running distillation jobs.

RuleKiln jobs can run for minutes and make many external provider calls. They need a durable execution path.

### 6.2 Decision

Use a lightweight Postgres-backed job queue.

The API creates job records. A separate worker process polls for pending jobs, claims one transactionally, and runs the pipeline.

Execution guarantee:

```text
At-least-once execution
```

Therefore every pipeline stage must remain idempotent.

### 6.3 Job Lifecycle Status

Use job lifecycle status separately from pipeline stage.

Recommended statuses:

```text
draft
pending
running
completed
failed
cancelled
```

Recommended stages:

```text
created
validating_project
extracting_rules
embedding_rules
clustering_rules
synthesizing_rules
reviewing_rule_conflicts
pruning_rules
compiling_prompts
evaluating_baseline
evaluating_distilled
selecting_strategy
analyzing_failures
checking_quality_gates
logging_artifacts
exporting_artifacts
completed
failed
```

Notes:

- `draft` is used by UI preview flows.
- `pending` means queued and claimable.
- `running` means claimed by a worker.
- `completed`, `failed`, and `cancelled` are terminal.
- `stage` records where the pipeline is within a running job.

### 6.4 Database Changes

Add queue-related columns to `distillation_jobs`.

```sql
alter table distillation_jobs
add column if not exists queue_status text not null default 'pending',
add column if not exists locked_by text,
add column if not exists locked_at timestamptz,
add column if not exists lease_expires_at timestamptz,
add column if not exists attempt_count int not null default 0,
add column if not exists max_attempts int not null default 3,
add column if not exists next_run_at timestamptz not null default now(),
add column if not exists selected_strategy text,
add column if not exists selected_prompt_version_id uuid,
add column if not exists primary_metric text,
add column if not exists baseline_score double precision,
add column if not exists selected_score double precision,
add column if not exists metric_delta double precision,
add column if not exists quality_gates_passed boolean;
```

Add indexes:

```sql
create index if not exists idx_distillation_jobs_queue_claim
on distillation_jobs (queue_status, next_run_at, created_at);

create index if not exists idx_distillation_jobs_lease
on distillation_jobs (queue_status, lease_expires_at);

create index if not exists idx_distillation_jobs_selected_prompt
on distillation_jobs (selected_prompt_version_id);
```

Optional stage-completion table:

```sql
create table if not exists job_stage_markers (
    id uuid primary key,
    job_id uuid not null references distillation_jobs(id),
    stage text not null,
    strategy text,
    artifact_type text,
    artifact_ref text,
    completed_at timestamptz not null default now(),
    metadata jsonb not null default '{}',
    unique (job_id, stage, strategy, artifact_type)
);
```

If the backend already has artifact records with uniqueness constraints, this table can be skipped. The important requirement is durable stage completion markers.

### 6.5 Job Claim Query

Worker claims one job at a time using `FOR UPDATE SKIP LOCKED`.

```sql
with next_job as (
    select id
    from distillation_jobs
    where queue_status = 'pending'
      and next_run_at <= now()
    order by created_at
    for update skip locked
    limit 1
)
update distillation_jobs j
set queue_status = 'running',
    status = 'running',
    locked_by = :worker_id,
    locked_at = now(),
    lease_expires_at = now() + (:lease_seconds || ' seconds')::interval,
    attempt_count = attempt_count + 1,
    updated_at = now()
from next_job
where j.id = next_job.id
returning j.*;
```

Default lease:

```text
30 minutes
```

For long jobs, the worker must renew the lease periodically.

### 6.6 Lease Renewal

```sql
update distillation_jobs
set lease_expires_at = now() + (:lease_seconds || ' seconds')::interval,
    updated_at = now()
where id = :job_id
  and locked_by = :worker_id
  and queue_status = 'running';
```

Renew every:

```text
lease_seconds / 3
```

For a 30-minute lease, renew every 10 minutes.

### 6.7 Expired Lease Recovery

```sql
update distillation_jobs
set queue_status = 'pending',
    status = 'pending',
    locked_by = null,
    locked_at = null,
    lease_expires_at = null,
    updated_at = now()
where queue_status = 'running'
  and lease_expires_at < now()
  and attempt_count < max_attempts;
```

Jobs that exceed `max_attempts`:

```sql
update distillation_jobs
set queue_status = 'failed',
    status = 'failed',
    error_message = coalesce(error_message, 'Job exceeded maximum retry attempts.'),
    updated_at = now()
where queue_status = 'running'
  and lease_expires_at < now()
  and attempt_count >= max_attempts;
```

### 6.8 Worker Process

Add a separate worker command:

```text
rulekiln-worker
```

Recommended module:

```text
src/rulekiln/workers/queue_worker.py
```

Worker loop:

```python
async def worker_loop(worker_id: str) -> None:
    while True:
        job = await jobs_repo.claim_next_job(worker_id=worker_id)

        if job is None:
            await asyncio.sleep(settings.worker_poll_interval_seconds)
            continue

        try:
            await run_distillation_job(job.id, worker_id=worker_id)
            await jobs_repo.mark_completed(job.id, worker_id=worker_id)
        except Exception as exc:
            await jobs_repo.mark_failed_or_retry(job.id, worker_id=worker_id, exc=exc)
```

CLI entrypoint:

```text
python -m rulekiln.workers.queue_worker
```

or console script:

```text
rulekiln-worker
```

### 6.9 API Behavior

`POST /distillation-jobs` should no longer enqueue an in-process background task.

Instead:

```text
validate request
create job with queue_status='pending'
return 202 with job_id
```

UI behavior remains unchanged.

### 6.10 Docker Compose

Add a worker service:

```yaml
worker:
  build: .
  env_file:
    - .env
  depends_on:
    - postgres
    - mlflow
  command: rulekiln-worker
  volumes:
    - .:/app
    - rulekiln_runs:/app/.rulekiln/runs
```

Keep the API service separate:

```yaml
api:
  command: uvicorn rulekiln.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Idempotency Requirements

Postgres queue gives at-least-once execution. Stages may run again after crash recovery.

Every stage must be safe to rerun.

### 7.1 Stage Idempotency Keys

Use keys like:

```text
job_id
stage
strategy
artifact_type
case_id optional
provider_model optional
schema_version optional
```

Examples:

```text
(job_id, extracting_rules, null, micro_rules)
(job_id, clustering_rules, dbscan, rule_clusters)
(job_id, synthesizing_rules, hdbscan, synthesized_rules)
(job_id, compiling_prompts, dbscan, prompt_version)
(job_id, analyzing_failures, hdbscan, failures_broken)
```

### 7.2 Model Call Cache Keys

Cache extraction calls by:

```text
case_hash
teacher_provider_profile
teacher_model
extraction_prompt_version
output_schema_version
```

Cache synthesis calls by:

```text
cluster_hash
synthesis_provider_profile
synthesis_model
synthesis_prompt_version
output_schema_version
```

Cache evaluation calls by:

```text
case_hash
student_provider_profile
student_model
compiled_prompt_hash
output_schema_version
```

### 7.3 Artifact Writes

Artifact writes should be atomic:

```text
write to temporary file
fsync/close
rename into final artifact path
```

Do not partially overwrite existing successful artifacts unless explicitly rerunning that stage.

---

## 8. Provider Rate Limiting

### 8.1 Problem

Rule extraction and evaluation can generate hundreds or thousands of provider calls. Bedrock, Azure OpenAI, OpenAI, and local gateways can rate-limit or throttle the job.

### 8.2 Schema Changes

Update `ProviderProfile`.

```python
class ProviderProfile(BaseModel):
    provider: ProviderKind
    region: str | None = None
    base_url: str | None = None
    api_key_env_var: str | None = None

    supports_chat: bool = True
    supports_embeddings: bool = False

    timeout_seconds: int = 60
    max_retries: int = 3

    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    max_concurrency: int = 3
```

Update `ModelRoute` to allow optional overrides.

```python
class ModelRoute(BaseModel):
    provider_profile: str
    model: str
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    max_concurrency: int | None = None
```

Effective limit precedence:

```text
ModelRoute override > ProviderProfile > AppSettings default
```

### 8.3 MVP Enforcement

MVP must enforce:

```text
max_concurrency
basic requests per minute
```

MVP may approximate or defer exact TPM enforcement if tokenizer support is not yet reliable across providers.

### 8.4 Rate Limiter Interface

```python
class ProviderRateLimiter:
    async def acquire(
        self,
        *,
        provider_profile: str,
        model: str,
        estimated_tokens: int | None = None,
    ) -> None:
        ...
```

Use this inside every provider call path:

- extraction agent
- synthesis agent
- conflict review agent
- embedding batches
- student evaluation
- judge/rubric scoring

### 8.5 Suggested Implementation

For single-worker MVP:

- use per-provider `asyncio.Semaphore` for concurrency
- use a simple sliding-window or token-bucket limiter for RPM
- log wait time as observability metadata

For multi-worker later:

- move rate limits into Postgres advisory locks, Redis, or provider-specific throttling service

Do not overbuild distributed throttling in the MVP.

---

## 9. Conflict Detection and Resolution

### 9.1 Problem

Teacher extraction may produce contradictory micro-rules across cases. Passing contradictions directly into the student prompt can reduce reliability.

### 9.2 Decision

Add one explicit conflict-review pass after synthesis and before pruning/compilation.

Do not implement a full iterative conflict-resolution loop in this adjustment.

### 9.3 Schema Changes

Update `SynthesizedRule`.

```python
class SynthesizedRule(BaseModel):
    id: str
    rule_type: str
    topic: str
    applies_when: list[str]
    outcome_conditions: dict[str, list[str]]
    tie_breakers: list[str] = []
    priority: int = 100

    has_conflicts: bool = False
    conflict_summary: str | None = None
    conflicting_micro_rule_ids: list[str] = []

    support_count: int = 0
    support_ratio: float = 0.0
    source_case_ids: list[str]
    source_micro_rule_ids: list[str]
```

Add conflict review output:

```python
class RuleConflictReview(BaseModel):
    synthesized_rule_id: str
    has_conflicts: bool
    conflict_summary: str | None = None
    conflicting_micro_rule_ids: list[str] = []
    resolution: Literal["keep", "modify", "split", "discard"]
    resolved_rules: list[SynthesizedRule] = []
```

### 9.4 Conflict Review Agent

Add:

```text
src/rulekiln/agents/rule_conflict_review.py
```

Agent purpose:

```text
Given one synthesized rule and the micro-rules that support it, identify contradictions, ambiguous outcomes, or incompatible conditions. Produce a resolved rule, split rules, or discard recommendation.
```

Agent instruction:

```text
You review synthesized task-policy rules for contradictions.

Given:
- a synthesized rule
- supporting micro-rules
- task output schema
- task mode

Determine whether the synthesized rule contains conflicts.

A conflict exists when:
- similar conditions map to incompatible outcomes
- the rule contains mutually exclusive instructions
- exceptions contradict the main condition
- the output path is ambiguous
- the rule would cause inconsistent student behavior

Return one of:
- keep: no material conflict
- modify: rewrite into one clean resolved rule
- split: split into multiple clean rules
- discard: conflicts cannot be safely resolved
```

### 9.5 Conflict Handling Rules

```text
keep:
  keep rule as-is

modify:
  replace synthesized rule with resolved rule

split:
  replace synthesized rule with resolved rules

discard:
  exclude rule from prompt compilation
```

Unresolved conflict rules must not be compiled into the prompt.

### 9.6 Artifacts

Add:

```text
rule_conflicts_dbscan.jsonl
rule_conflicts_hdbscan.jsonl
rules_discarded_conflicts.jsonl
```

---

## 10. Rule Pruning and Prompt Bloat Control

### 10.1 Problem

Too many rules can create prompt bloat and degrade small student model performance.

### 10.2 Task Config Changes

Add to `RuleKilnTask`.

```python
class RuleKilnTask(BaseModel):
    ...
    max_rules: int = 40
    max_prompt_tokens: int = 8000
    min_rule_support_count: int = 2
    preserve_golden_rules: bool = True
```

If these already exist in `limits`, allow both forms but normalize into a single internal `PromptBudgetConfig`.

### 10.3 Rule Metadata

Each synthesized rule should track:

```text
support_count
support_ratio
source_case_ids
source_micro_rule_ids
golden_case_backed
estimated_token_count
```

### 10.4 Pruning Order

Before prompt compilation:

1. Remove unresolved-conflict rules.
2. Preserve rules backed by golden cases unless they conflict.
3. Remove rules below `min_rule_support_count`, unless golden-backed.
4. Sort by `priority` ascending.
5. Sort by `support_count` descending.
6. Sort by `support_ratio` descending.
7. Keep rules until `max_rules` is reached.
8. Keep rules until `max_prompt_tokens` budget is reached.
9. Record all pruned rules with reason.

### 10.5 Pruning Reasons

Use explicit reasons:

```text
unresolved_conflict
below_min_support
max_rules_exceeded
prompt_token_budget_exceeded
duplicate_or_subsumed
```

### 10.6 Artifacts

Add:

```text
rules_selected_dbscan.jsonl
rules_selected_hdbscan.jsonl
rules_pruned_dbscan.jsonl
rules_pruned_hdbscan.jsonl
rule_pruning_report.json
```

### 10.7 Prompt Compiler Behavior

Prompt compiler receives only selected rules.

Prompt compiler should also include a short summary section:

```text
# Distilled Rule Policy

The following rules are selected from a larger rule set based on support, priority, conflict review, and prompt budget.
```

Do not include pruned rules in the final prompt.

---

## 11. Evaluation-to-Rule Failure Mapping

### 11.1 Problem

RuleKiln’s product thesis depends on an auditable rule layer. Evaluation needs to show which rules were matched, ignored, or violated.

### 11.2 Schema Changes

Add granular failure record.

```python
class CaseEvaluationFailure(BaseModel):
    case_id: str
    split: str
    failure_class: Literal["fixed", "broken", "unchanged_wrong"]

    expected: dict | str | None
    baseline_output: dict | str | None
    distilled_output: dict | str | None

    matched_rule_ids: list[str] = []
    violated_rule_ids: list[str] = []
    failed_assertion_paths: list[str] = []
    failed_assertion_types: list[str] = []

    explanation: str | None = None
```

Update `EvalResult`.

```python
class EvalResult(BaseModel):
    ...
    failed_case_count: int = 0
    violated_rule_counts: dict[str, int] = {}
    failed_assertion_path_counts: dict[str, int] = {}
    failures: list[CaseEvaluationFailure] = []
```

### 11.3 Matched Rules

The student prompt already asks for or can be made to ask for:

```json
{
  "matched_rules": ["rule_escalation_001"]
}
```

If the output schema does not include `matched_rules`, RuleKiln should either:

1. wrap evaluation prompts to request `matched_rules` in an evaluation envelope, or
2. infer rule matching from output paths/assertions where possible.

MVP recommendation:

```text
Require RuleKiln-compiled prompts to include matched_rules in the internal output contract when possible.
```

For user-facing tasks where `matched_rules` should not appear, hide it behind an internal evaluation envelope or strip it after evaluation.

### 11.4 Violated Rules

A rule is considered violated when:

- the case should trigger the rule based on expected output/evaluation criteria
- the distilled output fails the corresponding assertion/output path
- the matched rule is absent or contradicted
- the failure analyzer maps the failed assertion path to a rule output path

MVP mapping logic:

```text
failed_assertion.path == rule.output_path
  -> violated_rule_id includes rule.id
```

Fallback mapping:

```text
if no output_path match:
  use matched_rule_ids from model output
  use source_case_ids
  use topic similarity as low-confidence metadata only
```

### 11.5 Artifacts

Update failure artifacts:

```text
failures_fixed.jsonl
failures_broken.jsonl
failures_unchanged.jsonl
```

Each line should include:

```text
case_id
failure_class
expected
baseline_output
distilled_output
matched_rule_ids
violated_rule_ids
failed_assertion_paths
failed_assertion_types
explanation
```

Add aggregate artifact:

```text
violated_rule_summary.json
```

Example:

```json
{
  "rule_escalation_001": {
    "violated_count": 12,
    "broken_count": 3,
    "unchanged_wrong_count": 9,
    "failed_assertion_paths": ["$.escalation_needed"]
  }
}
```

### 11.6 UI Impact

Failures page should add columns:

```text
matched_rules
violated_rules
failed_assertion_paths
```

Rules page should show:

```text
violated_count
broken_count
unchanged_wrong_count
```

If artifacts are not available yet, the current stage-aware unavailable message remains acceptable.

---

## 12. Pipeline Changes

Updated pipeline:

```text
validate project
extract rules
embed rules
cluster rules
synthesize rules
review rule conflicts
prune rules
compile prompts
evaluate baseline
evaluate distilled
select strategy
analyze failures with rule mapping
check quality gates
log artifacts
export artifacts
complete job
```

New stages:

```text
reviewing_rule_conflicts
pruning_rules
```

### 12.1 Pseudocode

```python
async def run_distillation_job(job_id: str, worker_id: str) -> None:
    job = await load_job(job_id)

    await run_stage(job_id, "validating_project", validate_project)
    micro_rules = await run_stage(job_id, "extracting_rules", extract_rules)
    embeddings = await run_stage(job_id, "embedding_rules", embed_rules)

    clusters_by_strategy = await run_stage(
        job_id,
        "clustering_rules",
        cluster_dbscan_and_hdbscan,
    )

    synthesized_by_strategy = await run_stage(
        job_id,
        "synthesizing_rules",
        synthesize_by_strategy,
    )

    conflict_reviewed_by_strategy = await run_stage(
        job_id,
        "reviewing_rule_conflicts",
        review_conflicts_by_strategy,
    )

    selected_rules_by_strategy = await run_stage(
        job_id,
        "pruning_rules",
        prune_rules_by_strategy,
    )

    prompts_by_strategy = await run_stage(
        job_id,
        "compiling_prompts",
        compile_prompts_by_strategy,
    )

    baseline_eval = await run_stage(job_id, "evaluating_baseline", evaluate_baseline)
    distilled_evals = await run_stage(job_id, "evaluating_distilled", evaluate_distilled)

    selected_strategy = await run_stage(
        job_id,
        "selecting_strategy",
        select_strategy,
    )

    failure_report = await run_stage(
        job_id,
        "analyzing_failures",
        analyze_failures_with_rule_mapping,
    )

    gate_result = await run_stage(job_id, "checking_quality_gates", check_gates)
    await run_stage(job_id, "logging_artifacts", log_artifacts)
    await run_stage(job_id, "exporting_artifacts", export_artifacts)
```

---

## 13. API Compatibility

No public API shape change is required.

### 13.1 Job Creation

Existing:

```http
POST /distillation-jobs
```

Still returns:

```json
{
  "job_id": "job_123",
  "status": "queued"
}
```

Internally:

- old: scheduled `BackgroundTasks`
- new: writes queue claimable record

### 13.2 Job Status

Existing:

```http
GET /distillation-jobs/{job_id}
```

Should include:

```json
{
  "job_id": "job_123",
  "status": "running",
  "queue_status": "running",
  "stage": "reviewing_rule_conflicts",
  "progress": {
    "completed": 340,
    "total": 500
  }
}
```

If adding `queue_status` would break clients, include it as optional.

### 13.3 Outputs

No endpoint change required.

Existing output endpoints should read new artifacts when available.

---

## 14. Artifact Changes

Add new artifacts while preserving old ones.

Required existing artifacts:

```text
selected_distilled_prompt.md
rules.jsonl
eval_report.json
strategy_comparison.json
failures_fixed.jsonl
failures_broken.jsonl
failures_unchanged.jsonl
```

New artifacts:

```text
rule_conflicts_dbscan.jsonl
rule_conflicts_hdbscan.jsonl
rules_discarded_conflicts.jsonl
rules_selected_dbscan.jsonl
rules_selected_hdbscan.jsonl
rules_pruned_dbscan.jsonl
rules_pruned_hdbscan.jsonl
rule_pruning_report.json
violated_rule_summary.json
```

Artifact paths remain job-scoped:

```text
.rulekiln/runs/{job_id}/outputs/
.rulekiln/runs/{job_id}/exports/
.rulekiln/runs/{job_id}/metadata/
```

---

## 15. MLflow Changes

Log additional params:

```python
mlflow.log_param("execution_backend", execution_backend)
mlflow.log_param("worker_id", worker_id)
mlflow.log_param("max_rules", max_rules)
mlflow.log_param("min_rule_support_count", min_rule_support_count)
mlflow.log_param("provider_rate_limit_rpm", effective_rpm)
mlflow.log_param("provider_max_concurrency", effective_concurrency)
```

Log additional metrics:

```python
mlflow.log_metric("num_rules_discarded_conflicts", num_rules_discarded_conflicts)
mlflow.log_metric("num_rules_pruned", num_rules_pruned)
mlflow.log_metric("num_rules_selected", num_rules_selected)
mlflow.log_metric("num_violated_rules", num_violated_rules)
mlflow.log_metric("rate_limit_wait_seconds_total", wait_seconds_total)
```

Log additional artifacts:

```text
rule_conflicts_*.jsonl
rules_selected_*.jsonl
rules_pruned_*.jsonl
rule_pruning_report.json
violated_rule_summary.json
```

Prompt Registry remains optional.

---

## 16. UI Changes

Minimal UI changes:

### Job Status

Show new stages:

```text
reviewing_rule_conflicts
pruning_rules
```

### Results Page

Add:

```text
rules selected
rules pruned
rules discarded due to conflict
top violated rules
rate limit wait time optional
```

### Rules Page

Add fields:

```text
has_conflicts
conflict_summary
support_count
support_ratio
violated_count
pruned status if applicable
```

### Failures Page

Add columns:

```text
matched_rule_ids
violated_rule_ids
failed_assertion_paths
```

If artifacts are unavailable, keep the stage-aware unavailable message.

---

## 17. Settings Changes

Add:

```python
class AppSettings(BaseSettings):
    execution_backend: Literal["background_tasks", "postgres_queue"] = "postgres_queue"

    worker_poll_interval_seconds: float = 2.0
    worker_lease_seconds: int = 1800
    worker_id: str | None = None

    default_provider_max_concurrency: int = 3
    default_provider_rate_limit_rpm: int | None = None
    default_provider_rate_limit_tpm: int | None = None

    default_max_rules: int = 40
    default_min_rule_support_count: int = 2
    default_max_prompt_tokens: int = 8000
```

`.env.example` additions:

```text
EXECUTION_BACKEND=postgres_queue
WORKER_POLL_INTERVAL_SECONDS=2
WORKER_LEASE_SECONDS=1800
DEFAULT_PROVIDER_MAX_CONCURRENCY=3
DEFAULT_PROVIDER_RATE_LIMIT_RPM=
DEFAULT_PROVIDER_RATE_LIMIT_TPM=
DEFAULT_MAX_RULES=40
DEFAULT_MIN_RULE_SUPPORT_COUNT=2
DEFAULT_MAX_PROMPT_TOKENS=8000
```

Provider profile additions:

```text
PROVIDER_PROFILES__BEDROCK_PRIMARY__MAX_CONCURRENCY=2
PROVIDER_PROFILES__BEDROCK_PRIMARY__RATE_LIMIT_RPM=60
PROVIDER_PROFILES__BEDROCK_PRIMARY__RATE_LIMIT_TPM=
```

---

## 18. Testing Plan

### 18.1 Queue Tests

Add tests:

```text
test_claim_next_job_uses_skip_locked
test_two_workers_do_not_claim_same_job
test_expired_lease_returns_job_to_pending
test_max_attempts_marks_job_failed
test_completed_job_is_not_reclaimed
test_stage_resume_starts_at_first_incomplete_stage
```

### 18.2 Rate Limiter Tests

Add tests:

```text
test_provider_max_concurrency_is_enforced
test_provider_rpm_limit_waits
test_route_override_takes_precedence_over_profile
test_profile_limit_takes_precedence_over_app_default
```

Use fake providers only.

### 18.3 Conflict Review Tests

Add tests:

```text
test_conflicting_rule_is_marked_has_conflicts
test_discarded_conflict_rule_is_not_compiled
test_split_conflict_rule_produces_multiple_rules
test_conflict_artifacts_are_written
```

### 18.4 Pruning Tests

Add tests:

```text
test_rules_below_min_support_are_pruned
test_golden_backed_rule_is_preserved
test_max_rules_is_enforced
test_prompt_token_budget_is_enforced
test_pruning_report_contains_reasons
```

### 18.5 Eval Mapping Tests

Add tests:

```text
test_failed_assertion_path_maps_to_rule_output_path
test_failure_artifact_contains_violated_rule_ids
test_violated_rule_summary_counts_failures
test_failures_page_view_model_includes_rule_mapping
```

### 18.6 Regression Tests

Existing tests should continue to pass:

```text
job creation
job status
prompt compiler determinism
pipeline strategy selection
quality gates
artifact writing
MLflow logging
UI status and results pages
```

---

## 19. Implementation Tasks

### Phase 1: Durable Execution Backend

```text
A001 Add queue columns to distillation_jobs migration
A002 Add optional job_stage_markers table or equivalent artifact-stage uniqueness
A003 Implement JobQueueRepository.claim_next_job() using FOR UPDATE SKIP LOCKED
A004 Implement lease renewal and expired lease recovery
A005 Implement queue worker process rulekiln-worker
A006 Update POST /distillation-jobs to create pending jobs without BackgroundTasks when EXECUTION_BACKEND=postgres_queue
A007 Add worker service to docker-compose.yml
A008 Add queue lifecycle tests
```

### Phase 2: Provider Rate Limiting

```text
B001 Add rate_limit_rpm, rate_limit_tpm, and max_concurrency to ProviderProfile
B002 Add optional route-level rate limit overrides to ModelRoute
B003 Implement effective provider limit resolution
B004 Implement in-process provider rate limiter
B005 Wrap chat provider calls with rate limiter
B006 Wrap embedding provider calls with rate limiter
B007 Log rate-limit wait time and effective limits
B008 Add rate limiter tests
```

### Phase 3: Conflict Review

```text
C001 Add conflict fields to SynthesizedRule
C002 Add RuleConflictReview schema
C003 Implement rule_conflict_review Pydantic AI agent
C004 Add reviewing_rule_conflicts pipeline stage
C005 Exclude unresolved conflict rules from compilation
C006 Export rule_conflicts_*.jsonl and rules_discarded_conflicts.jsonl
C007 Add conflict review tests
```

### Phase 4: Rule Pruning

```text
D001 Add max_rules, min_rule_support_count, max_prompt_tokens, and preserve_golden_rules config
D002 Add support_count, support_ratio, golden_case_backed, and estimated_token_count to synthesized rule metadata
D003 Implement rule pruning service
D004 Add pruning_rules pipeline stage
D005 Update prompt compiler to compile only selected rules
D006 Export rules_selected_*.jsonl, rules_pruned_*.jsonl, and rule_pruning_report.json
D007 Add pruning tests
```

### Phase 5: Eval-to-Rule Mapping

```text
E001 Add CaseEvaluationFailure schema
E002 Add matched_rule_ids, violated_rule_ids, failed_assertion_paths, and failed_assertion_types to failure records
E003 Update evaluator/failure analyzer to map failed assertion paths to rule output paths
E004 Add violated_rule_counts and failed_assertion_path_counts to EvalResult
E005 Export violated_rule_summary.json
E006 Update UI view models for failures and rules
E007 Add eval mapping tests
```

### Phase 6: MLflow, UI, and Docs

```text
F001 Log execution backend and worker metadata to MLflow
F002 Log conflict/pruning/eval-rule mapping artifacts to MLflow
F003 Update UI job status stage labels
F004 Update results page with selected/pruned/conflict rule counts
F005 Update rules page with conflict/support/violation fields
F006 Update failures page with matched/violated rule IDs
F007 Update README and backend architecture docs
F008 Update Docker Compose documentation for api + worker services
```

---

## 20. Rollout Plan

Recommended order:

```text
1. Add schema migration and queue repository.
2. Add worker process but keep BackgroundTasks feature flag available.
3. Run queue worker locally with fake provider.
4. Switch local default to postgres_queue.
5. Add provider rate limiting.
6. Add conflict review.
7. Add rule pruning.
8. Add eval-to-rule mapping.
9. Update UI and MLflow artifacts.
10. Remove BackgroundTasks path after several stable runs.
```

Rollback plan:

```text
Set EXECUTION_BACKEND=background_tasks
```

This rollback should remain available until the queue worker is stable.

---

## 21. Acceptance Criteria

This design adjustment is complete when:

1. New jobs are queued in Postgres and picked up by a separate worker.
2. API process restart does not permanently orphan pending jobs.
3. Worker crash causes job lease expiry and retry.
4. Two workers cannot claim the same job simultaneously.
5. Provider calls respect configured `max_concurrency` and basic RPM limits.
6. Synthesized rules include conflict metadata.
7. Unresolved conflict rules are excluded from prompt compilation.
8. Rule pruning enforces `max_rules`, `min_rule_support_count`, and prompt token budget.
9. Failure artifacts include `matched_rule_ids`, `violated_rule_ids`, and `failed_assertion_paths`.
10. MLflow logs conflict, pruning, and violation artifacts.
11. UI still supports the same operator flow.
12. All tests run offline with fake providers.
13. Existing API request shape remains compatible.

---

## 22. Final Recommendation

Implement this adjustment before expanding UI complexity or adding hosted-product features.

The most important backend hardening item is the Postgres queue. Without it, long-running jobs remain operationally fragile.

The next most important product-quality item is eval-to-rule mapping. Without it, RuleKiln cannot fully deliver on the auditable rule-layer thesis.

Recommended priority:

```text
1. Postgres queue
2. Provider rate limiting
3. Rule pruning
4. Eval-to-rule mapping
5. One-pass conflict review
```

Conflict review is important, but if implementation time is tight, start by detecting conflicts and excluding unresolved conflict rules before adding sophisticated split/modify behavior.
