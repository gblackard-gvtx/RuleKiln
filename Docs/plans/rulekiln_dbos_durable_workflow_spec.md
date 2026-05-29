# RuleKiln DBOS Durable Workflow Adoption Spec

## 1. Purpose

RuleKiln needs durable, resumable execution for long-running prompt-hardening jobs.

This became a priority because local student-model evaluation can fail due to local infrastructure instability, while earlier stages may have already spent paid teacher-model credits. RuleKiln must not require rerunning expensive teacher stages just because the local student server, GPU, power supply, worker process, or queue worker fails later in the pipeline.

This spec defines how RuleKiln should use DBOS for durable workflow orchestration while retaining RuleKiln-owned artifact checkpoints, case-level evaluation persistence, provider error classification, and model usage/cost tracking.

---

## 2. Problem Statement

Current RuleKiln jobs can include expensive and long-running stages:

```text
teacher rule extraction
embedding
clustering
rule synthesis
prompt compilation
baseline student evaluation
DBSCAN student evaluation
HDBSCAN student evaluation
report aggregation
MLflow/artifact logging
```

The critical failure case is:

```text
1. RuleKiln spends OpenAI/Anthropic/Bedrock teacher credits.
2. Teacher outputs are generated.
3. Local student evaluation begins.
4. Local AI server fails because GPU/power draw is unstable.
5. The job fails.
6. Restarting the job risks repeating expensive teacher stages.
```

That is unacceptable for larger benchmarks and paid API usage.

RuleKiln needs a workflow model where completed stages are checkpointed and local evaluation can resume without re-spending teacher tokens.

---

## 3. Decision

Adopt DBOS as the durable workflow and queue layer, subject to a short implementation spike.

DBOS should own:

```text
workflow execution
durable queues
step checkpointing
workflow resume
retry orchestration
workflow visibility
```

RuleKiln should continue to own:

```text
artifact validity
idempotent stage outputs
case-level evaluation results
provider error classification
model usage/cost events
prompt/version artifacts
MLflow artifact logging
```

DBOS should simplify the orchestration layer, but it should not replace RuleKiln's explicit artifacts, per-case evaluation rows, or idempotency checks.

---

## 4. Core Safety Rule

Once RuleKiln has spent paid teacher tokens, it must not rerun teacher work unless:

```text
the teacher artifact is missing
the teacher artifact fails validation
the user explicitly requests a full rerun
the task/cases/provider config has materially changed
```

Default behavior should be:

```text
reuse existing valid artifacts
resume from the latest safe checkpoint
retry only the incomplete stage or incomplete cases
```

The most important guarantee:

```text
A local student evaluation failure must never force RuleKiln to rerun expensive teacher work.
```

---

## 5. Goals

This change should allow RuleKiln to:

```text
resume failed jobs from the last completed durable step
avoid repeating paid teacher calls
survive worker restarts
survive local model server failures
survive transient provider/network failures
retry restartable stages
skip already completed eval cases
resume baseline/dbscan/hdbscan local evaluation
surface retryable vs terminal failures in the UI
preserve auditability through artifacts and DB records
```

---

## 6. Non-Goals

Do not implement these in the first pass:

```text
full multi-tenant SaaS workflow management
distributed autoscaling
advanced workflow version migration
human approval workflows
cross-workflow dependency graphs
production traffic orchestration
provider-level billing enforcement
GPU power management
automatic hardware remediation
```

This spec is focused on durable RuleKiln job execution and resumable local evaluation.

---

## 7. Why DBOS Fits RuleKiln

RuleKiln jobs have the exact characteristics durable workflow tools are meant to handle:

```text
long-running jobs
paid external API calls
stage dependencies
local model instability
retryable errors
need for resume
Postgres already in the architecture
need for workflow visibility
```

Without DBOS, RuleKiln must hand-roll a significant amount of infrastructure:

```text
Postgres queue workers
stage records
retry state
resume state
worker leases
heartbeats
workflow recovery
manual polling
manual stage transitions
```

With DBOS, RuleKiln can model the pipeline as a durable workflow composed of checkpointed steps.

---

## 8. Important Caveat

DBOS does not remove the need for RuleKiln idempotency.

A DBOS step can be retried or resumed, but external side effects still need protection:

```text
OpenAI/Anthropic/Bedrock calls
artifact writes
MLflow writes
database inserts
local model calls
case evaluation rows
```

Each step must be written so that it can safely run again.

The pattern is:

```text
check durable output
if output exists and validates:
  skip
else:
  do work
  persist output atomically
```

---

## 9. Workflow Model

### 9.1 Top-Level Workflow

Create one durable DBOS workflow per RuleKiln job.

Conceptual workflow:

```python
@DBOS.workflow()
def run_rulekiln_job(job_id: str) -> None:
    validate_project(job_id)

    extract_micro_rules(job_id)
    embed_micro_rules(job_id)
    cluster_rules(job_id)
    synthesize_rules(job_id)
    resolve_conflicts(job_id)
    prune_rules(job_id)

    compile_prompts(job_id)

    evaluate_student_strategy(job_id, "baseline")
    evaluate_student_strategy(job_id, "dbscan")
    evaluate_student_strategy(job_id, "hdbscan")

    aggregate_evaluation_report(job_id)
    write_final_artifacts(job_id)
    log_mlflow_artifacts(job_id)
```

### 9.2 Durable Step Boundaries

Each major pipeline stage should be a DBOS step:

```text
validate_project
extract_micro_rules
embed_micro_rules
cluster_rules
synthesize_rules
resolve_conflicts
prune_rules
compile_prompts
evaluate_baseline
evaluate_dbscan
evaluate_hdbscan
aggregate_evaluation_report
log_mlflow_artifacts
```

### 9.3 Paid Provider Checkpoint Boundaries

Paid provider stages must end with durable artifacts.

Examples:

```text
extract_micro_rules
  writes outputs/micro_rules.jsonl

synthesize_rules
  writes outputs/synthesized_rules.json

resolve_conflicts
  writes outputs/conflict_resolution.json

compile_prompts
  writes outputs/baseline_prompt.md
  writes outputs/distilled_prompt_dbscan.md
  writes outputs/distilled_prompt_hdbscan.md
```

Once these artifacts exist and validate, local evaluation failures must not cause the teacher stages to rerun.

---

## 10. Recommended DBOS Pattern

Use DBOS for stage-level durable workflow orchestration and RuleKiln DB/artifacts for case-level idempotency.

### 10.1 Recommended MVP Pattern

```text
DBOS step per major stage
+
RuleKiln case-level eval result upserts
```

This is the recommended MVP.

Example:

```python
@DBOS.step()
def evaluate_student_strategy(job_id: str, strategy: str) -> None:
    cases = load_eval_cases(job_id)
    completed = load_completed_case_ids(job_id, strategy)

    for case in cases:
        if case.id in completed:
            continue

        result = call_student_model(case)
        eval_result = evaluate_result(case, result)

        upsert_eval_case_result(job_id, strategy, case.id, eval_result)
```

This gives:

```text
DBOS durability at stage boundary
RuleKiln durability at case boundary
```

### 10.2 Alternative Pattern

A more granular option is one DBOS step per evaluated case:

```python
@DBOS.workflow()
def evaluate_strategy(job_id: str, strategy: str):
    for case_id in case_ids:
        evaluate_one_case(job_id, strategy, case_id)

@DBOS.step()
def evaluate_one_case(job_id: str, strategy: str, case_id: str):
    ...
```

This gives finer-grained workflow checkpointing but may create many DBOS steps:

```text
300 validation cases x 3 strategies = 900 steps
```

This can be considered later if stage-level steps plus case-level upserts are insufficient.

---

## 11. Idempotent Stage Contract

Every DBOS step must follow this rule:

```text
If the expected durable output already exists and validates, return without repeating work.
```

Examples:

### 11.1 Teacher Extraction

```python
@DBOS.step()
def extract_micro_rules(job_id: str) -> None:
    if artifact_valid(job_id, "outputs/micro_rules.jsonl"):
        return

    micro_rules = call_teacher_for_rules(...)
    write_artifact_atomically(job_id, "outputs/micro_rules.jsonl", micro_rules)
```

### 11.2 Prompt Compilation

```python
@DBOS.step()
def compile_prompts(job_id: str) -> None:
    required = [
        "outputs/baseline_prompt.md",
        "outputs/distilled_prompt_dbscan.md",
        "outputs/distilled_prompt_hdbscan.md",
    ]

    if all(artifact_valid(job_id, path) for path in required):
        return

    compile_and_write_prompts(...)
```

### 11.3 Local Student Evaluation

```python
@DBOS.step()
def evaluate_student_strategy(job_id: str, strategy: str) -> None:
    cases = load_eval_cases(job_id)
    completed = load_completed_case_ids(job_id, strategy)

    for case in cases:
        if case.id in completed:
            continue

        evaluate_and_upsert_case_result(job_id, strategy, case)
```

---

## 12. Case-Level Evaluation Durability

DBOS step durability is not enough for local model evaluation.

If a step evaluates 300 cases and fails on case 240, RuleKiln must not rerun the first 239 local calls unless explicitly requested.

Therefore, RuleKiln must persist each completed case result immediately.

Required table or durable artifact model:

```sql
create table if not exists eval_case_results (
    id uuid primary key,
    job_id uuid not null references distillation_jobs(id),

    student_id text,
    strategy text not null,
    split text not null,
    case_id text not null,

    expected jsonb,
    actual jsonb,
    raw_output text,

    passed boolean not null,
    case_score double precision not null default 0,

    malformed boolean not null default false,
    invalid_label boolean not null default false,

    error_type text,
    error_message text,

    created_at timestamptz not null default now(),

    unique (job_id, student_id, strategy, split, case_id)
);
```

The unique constraint is required for idempotent upserts.

Resume logic:

```text
load all eval cases
load completed eval_case_results
skip completed case_ids
evaluate missing case_ids only
```

---

## 13. Provider Error Classification

RuleKiln must classify errors as retryable or terminal.

### 13.1 Retryable Errors

Retryable errors include:

```text
connection refused
connection reset
timeout
HTTP 502
HTTP 503
HTTP 504
local model server unavailable
worker killed
GPU server unavailable
rate limit
temporary provider outage
temporary network failure
```

Local AI server failures should normally be retryable.

### 13.2 Terminal Errors

Terminal errors include:

```text
invalid task.yaml
invalid cases.jsonl
missing provider profile
invalid model name
bad output schema
schema validation impossible
authentication failure
permission denied
unsupported provider feature
malformed project input
```

Quota exceeded may be retryable or terminal depending on configuration.

### 13.3 Error Behavior

Retryable local provider failure:

```text
mark stage as failed_retryable or waiting_for_retry
preserve completed case results
do not rerun teacher stages
allow resume/requeue
```

Terminal failure:

```text
mark job failed_terminal
surface error to user
do not retry automatically
```

---

## 14. Circuit Breaker for Flaky Local Providers

Add provider-profile settings for local providers:

```yaml
provider_profiles:
  local-qwen-lb:
    max_concurrency: 1
    max_retries: 3
    retry_backoff_seconds: 10
    circuit_breaker_failure_threshold: 5
    circuit_breaker_cooldown_seconds: 300
```

Behavior:

```text
if repeated retryable failures exceed threshold:
  pause provider calls
  mark stage waiting_for_retry
  requeue after cooldown
```

This prevents RuleKiln from hammering a failing local model server.

---

## 15. Queue and Worker Behavior

### 15.1 Job Creation

FastAPI should create or enqueue a DBOS workflow:

```text
POST /distillation-jobs
  -> create job row
  -> enqueue/start DBOS workflow with job_id
  -> return 202
```

### 15.2 Worker Execution

DBOS worker executes workflow steps.

RuleKiln should no longer need a fully custom `FOR UPDATE SKIP LOCKED` queue for the DBOS-managed path.

### 15.3 Resume

Resume should requeue or resume the same workflow/job ID.

UI action:

```text
Resume retryable job
```

Backend behavior:

```text
resume/restart DBOS workflow for same job_id
completed steps are skipped by DBOS or by RuleKiln artifact checks
completed cases are skipped by eval_case_results
```

---

## 16. Job Status Mapping

RuleKiln job status should expose DBOS/retry state in user-friendly terms.

Suggested statuses:

```text
pending
running
waiting_for_retry
failed_retryable
failed_terminal
completed
cancelled
```

Suggested stage details:

```json
{
  "status": "waiting_for_retry",
  "stage": "evaluating_student",
  "strategy": "dbscan",
  "student_id": "local_qwen_4b",
  "error_type": "ConnectionResetError",
  "retryable": true,
  "completed_cases": 239,
  "total_cases": 300,
  "next_retry_at": "2026-05-25T14:30:00Z"
}
```

---

## 17. UI Changes

Add job actions:

```text
Resume retryable job
Retry current stage
Retry missing local eval cases
Cancel job
```

For the current local-server failure scenario, the primary action should be:

```text
Resume local evaluation
```

not:

```text
Start new job
```

Add progress display for local evaluation:

```text
Evaluating DBSCAN prompt with local_qwen_4b
Completed cases: 239 / 300
Last error: local provider connection reset
Status: waiting for retry
```

Add a warning when paid teacher artifacts already exist:

```text
Teacher artifacts are complete and will be reused. Resuming this job will not rerun teacher extraction unless artifacts are invalid or you request a full rerun.
```

---

## 18. Artifacts

DBOS does not replace RuleKiln artifacts.

Required durable artifacts:

```text
outputs/micro_rules.jsonl
outputs/embeddings.jsonl or embeddings metadata
outputs/clusters.json
outputs/synthesized_rules.json
outputs/conflict_resolution.json
outputs/pruned_rules.json
outputs/baseline_prompt.md
outputs/distilled_prompt_dbscan.md
outputs/distilled_prompt_hdbscan.md
outputs/evals/{student_id}/baseline_case_results.jsonl
outputs/evals/{student_id}/dbscan_case_results.jsonl
outputs/evals/{student_id}/hdbscan_case_results.jsonl
outputs/eval_report.json
outputs/strategy_comparison.json
metadata/token_cost_summary.json
```

Each artifact should be written atomically where possible:

```text
write temporary file
fsync/flush if appropriate
rename to final path
validate final artifact
```

---

## 19. Model Usage and Cost Safety

Model usage/cost tracking must integrate with durable execution.

Rules:

```text
log model call events per actual provider call
do not duplicate model_call_events when a completed step is skipped
use idempotency keys where possible
aggregate cost from logged events
do not estimate duplicate teacher spend on resume
```

Recommended model call idempotency key:

```text
{job_id}:{stage}:{role}:{strategy}:{student_id}:{case_id}:{request_hash}
```

For teacher extraction by case:

```text
{job_id}:extracting_rules:teacher:{case_id}
```

For student evaluation:

```text
{job_id}:evaluating_student:student:{student_id}:{strategy}:{case_id}
```

---

## 20. OpenAI Batch Interaction

If OpenAI Batch support is enabled, DBOS should orchestrate batch submission and polling.

Batch flow:

```text
build batch input artifact
submit batch
persist provider_batch_id
mark step waiting_on_provider_batch
poll batch status
download results
parse results
log model_call events
write micro_rules.jsonl
complete step
```

Important:

```text
once provider_batch_id exists, resume must poll existing batch, not submit a duplicate batch
```

---

## 21. Migration Strategy

Use a spike first.

### 21.1 Spike Scope

Implement DBOS for a narrow workflow path:

```text
validate_project
compile_prompts
evaluate_baseline_local_student
```

The spike should prove:

```text
start job
evaluate local student over multiple cases
kill local llama server halfway through
job becomes retryable or pauses
restart local llama server
resume same job
completed cases are skipped
no prior stages rerun
final report aggregates all cases
```

### 21.2 Full Migration

After spike success, migrate:

```text
teacher extraction
embedding
clustering
synthesis
conflict resolution
pruning
all student strategies
report aggregation
MLflow logging
```

---

## 22. Implementation Tasks

```text
DBOS001 Add DBOS dependency and configuration.
DBOS002 Initialize DBOS with RuleKiln Postgres.
DBOS003 Add DBOS worker entrypoint.
DBOS004 Create run_rulekiln_job workflow.
DBOS005 Convert validate_project into DBOS step.
DBOS006 Convert prompt compilation into DBOS step.
DBOS007 Convert baseline local evaluation into DBOS step.
DBOS008 Add eval_case_results upsert/skip logic if not already present.
DBOS009 Add retryable provider error classification.
DBOS010 Add local provider circuit breaker settings.
DBOS011 Add resume retryable job endpoint.
DBOS012 Add UI resume action.
DBOS013 Add job status mapping for waiting_for_retry/failed_retryable.
DBOS014 Add artifact validity checks before each stage.
DBOS015 Ensure teacher artifacts are reused on resume.
DBOS016 Add model_call_event idempotency keys.
DBOS017 Add tests for worker restart/resume.
DBOS018 Add tests for local provider failure/resume.
DBOS019 Add tests proving teacher stages are not rerun.
DBOS020 Migrate remaining pipeline stages after spike.
DBOS021 Update docs.
```

---

## 23. Testing Requirements

### 23.1 Unit Tests

```text
test_stage_skips_when_artifact_exists
test_eval_strategy_skips_completed_case_results
test_retryable_provider_errors_are_classified
test_terminal_provider_errors_are_classified
test_circuit_breaker_opens_after_failure_threshold
test_model_call_idempotency_key_is_stable
```

### 23.2 Integration Tests

```text
test_dbos_workflow_completes_happy_path
test_dbos_workflow_resumes_after_worker_restart
test_local_eval_resumes_after_provider_failure
test_completed_eval_cases_are_not_rerun
test_teacher_artifacts_are_not_regenerated_on_eval_failure
test_resume_job_reuses_existing_job_id
test_failed_retryable_job_can_be_resumed
```

### 23.3 Manual Failure Test

Manual test scenario:

```text
1. Start RuleKiln job with local student.
2. Let baseline or dbscan evaluation start.
3. Stop llama.cpp / HAProxy / local AI server.
4. Confirm job enters failed_retryable or waiting_for_retry.
5. Restart local AI server.
6. Click Resume.
7. Confirm completed cases are skipped.
8. Confirm teacher artifacts are not regenerated.
9. Confirm final report completes.
```

---

## 24. Acceptance Criteria

This change is complete when:

1. RuleKiln can start a DBOS-managed workflow for a job.
2. Major pipeline stages are represented as durable steps.
3. Each stage checks for valid existing artifacts before doing work.
4. Teacher artifacts are persisted before local evaluation begins.
5. Local evaluation persists each case result immediately.
6. Local evaluation resumes missing cases after failure.
7. Retryable local provider failures do not mark the job terminal.
8. UI/API can resume a retryable job.
9. Resuming a job does not rerun completed teacher stages.
10. Resuming a job does not duplicate completed case results.
11. Model usage/cost events are not duplicated on resume.
12. Tests prove recovery from local provider failure.
13. Tests prove recovery from worker restart.
14. Documentation explains DBOS workflow behavior and resume semantics.

---

## 25. Immediate Mitigation Before DBOS

Until DBOS is implemented, reduce local server pressure:

```text
set local student max_concurrency = 1
reduce queue parallelism for local student eval
prefer validation subset sizes during early tests
avoid multiple local student strategies running concurrently
```

Example:

```env
PROVIDER_PROFILES__LOCAL_QWEN_LB__MAX_CONCURRENCY=1
```

If the server power supply is unstable, one GPU/model replica may be safer than two sustained replicas.

---

## 26. Final Recommendation

Use DBOS for durable workflow orchestration, but do not rely on DBOS alone.

The correct architecture is:

```text
DBOS:
  durable workflow
  durable step execution
  queues
  resume
  retries

RuleKiln:
  artifact checkpoints
  case-level eval persistence
  provider error classification
  cost/event idempotency
  audit artifacts
```

This should be treated as an MVP hardening requirement before running larger paid teacher benchmarks.
