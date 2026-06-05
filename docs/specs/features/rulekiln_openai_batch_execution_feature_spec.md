# RuleKiln OpenAI Batch Execution Feature Spec

## 1. Purpose

This spec adds optional OpenAI Batch API support to RuleKiln.

The goal is to reduce the cost of large, independent OpenAI model-call workloads such as teacher rule extraction, rubric judging, hosted student evaluation, and OpenAI embeddings.

RuleKiln's core economic thesis is:

```text
Use an expensive teacher model during prompt hardening.
Use a cheaper/local/smaller student model during production.
Show quality improvement and hardening cost.
```

OpenAI Batch support strengthens that thesis by allowing expensive teacher work to run asynchronously at lower cost when immediate responses are not required.

---

## 2. Background

OpenAI Batch API is designed for asynchronous bulk processing. OpenAI's docs describe it as useful for evaluations, dataset classification, and embedding repositories. It accepts a `.jsonl` input file where each line is one request, and each request must include a unique `custom_id`.

OpenAI currently describes Batch API as having:

```text
50% lower costs compared to synchronous APIs
separate higher rate limits
a 24-hour completion window
asynchronous execution
```

Important operational implication:

```text
Batch is cheaper but not interactive.
```

Therefore RuleKiln should use Batch only for stages that can tolerate waiting.

---

## 3. Current State

> **Implementation status (as of 2026-06-04):** OpenAI batch extraction has shipped as Phase 1/2 of the plan described in `docs/specs/SPEC_Batch_API_Support.md`. The `EXTRACTING_RULES` stage can now use the OpenAI Batch API end-to-end. See §3.1 for what is now implemented; §3.2 describes the remaining sequencing.

### 3.1 Implemented

- `BatchChatModelClient(ChatModelClient, ABC)` abstract interface with `submit_batch`, `poll_batch`, `collect_batch`.
- `OpenAIChatClient` now subclasses `BatchChatModelClient`. Per-call path unchanged (pydantic-ai). Batch path uses the OpenAI SDK directly against `/v1/responses`.
- `schemas/batch.py` — `BatchItem`, `BatchPollStatus`, `BatchItemResult`, `BatchResult`.
- `providers/batch_schema_registry.py` — schema registry for cross-boundary class lookup.
- `db/models.py` — `BatchJob` SQLAlchemy model; migration `0006_add_batch_jobs.py`.
- `config/settings.py` — `ProviderProfile.batch_enabled`, `AppSettings.batch_poll_interval_seconds`.
- `schemas/classroom.py` — `PhaseTeacherConfig.batch_enabled`, `batch_min_items`.
- `usage/pricing.py` — `PricingService.calculate_batch()` reads `batch_discount` from YAML.
- `workers/distillation_worker.py` — batch submit/collect path for `EXTRACTING_RULES` with `batch_min_items` threshold and sequential fallback; two new `PipelineStage` values.
- `workers/dbos_workflow.py` — three `@DBOS.step` functions (submit, poll, collect) with `DBOS.sleep` durable poll loop.

The actual implementation differs from §8–§9 of this spec in the following ways:
- `ProviderProfile.batch_enabled` replaces the proposed `supports_batch` / `default_execution_mode` fields.
- `PhaseTeacherConfig.batch_enabled` replaces `ModelRoute.execution_mode`; there is no `"auto"` mode — only explicit opt-in.
- The DB table is `batch_jobs` (not `provider_batch_jobs`); fields differ from §11 — see migration `0006` for the actual schema.
- The Batch API endpoint used is `/v1/responses` (not `/v1/chat/completions`).
- The `custom_id` is the case ID directly (not `{stage}:{job_id}:{case_id}`).

### 3.2 Planned next phases

```text
Phase 3 — EVALUATING_BASELINE + EVALUATING_DISTILLED (student eval batching)
Phase 4 — SYNTHESIZING_RULES + REVIEWING_RULE_CONFLICTS (low volume, batch_min_items gated)
Phase 5 — Anthropic batch provider (AnthropicChatClient subclasses BatchChatModelClient)
Phase 6 — Bedrock (deferred pending deployment requirement)
```

### 3.3 Pre-batch baseline (historical)

RuleKiln previously used synchronous provider calls only:

```text
for each case:
  call provider synchronously
  parse result
  continue pipeline
```

This was simple but expensive for large teacher workloads.

---

## 4. Decision

Add OpenAI Batch as an optional provider execution mode.

Execution modes:

```text
sync
batch
auto
```

Definitions:

```text
sync:
  use normal synchronous provider calls

batch:
  use provider batch execution when supported; fail if unsupported

auto:
  use batch when provider/stage/request count are suitable; otherwise use sync
```

MVP should implement OpenAI Batch first for:

```text
teacher rule extraction
```

Later phases can add:

```text
judge/rubric scoring
hosted student evaluation
OpenAI embeddings
conflict review
synthesis by cluster
```

---

## 5. Non-Goals

Do not implement these in the first pass:

```text
batch support for all providers
batch support for local llama.cpp servers
provider-agnostic batch abstraction beyond OpenAI
real-time streaming batch progress
multi-day batch orchestration
automatic batch splitting by token budget
production traffic batching
billing integration
human approval workflow for batch cost
```

The MVP should focus on OpenAI teacher extraction via Batch API.

---

## 6. Product Value

Without Batch:

```text
OpenAI teacher extraction:
  fastest
  highest token cost
  standard rate limits
```

With Batch:

```text
OpenAI teacher extraction:
  slower
  materially cheaper
  better suited for benchmark/offline hardening jobs
```

For RuleKiln benchmarks:

```text
Teacher:
  OpenAI GPT-5.5 via Batch

Student:
  local Qwen 4B via HAProxy

Embedding:
  local mxbai-embed-large-v1
```

This gives RuleKiln a stronger cost story:

```text
Use an expensive frontier teacher at batch pricing.
Use a local student with zero API runtime cost.
Measure whether prompt hardening makes the local student viable.
```

---

## 7. Where Batch Should Be Used

### 7.1 Strong Fit

Batch is a strong fit for independent, high-volume calls:

```text
extracting_rules:
  one teacher call per training case

evaluating_baseline:
  one hosted student call per validation case

evaluating_distilled:
  one hosted student call per validation case per prompt strategy

judge/rubric scoring:
  one judge call per output/case

embedding_rules:
  OpenAI embedding batches if not using local embeddings
```

### 7.2 Poor Fit

Batch is not a good fit for:

```text
single-case debugging
interactive UI preview
local llama.cpp student calls
tiny jobs
stages where the next result is needed immediately
stages where latency matters more than cost
```

### 7.3 MVP Target

MVP target:

```text
extracting_rules with OpenAI teacher
```

Threshold:

```text
Use batch only when request_count >= batch_min_requests.
Default batch_min_requests = 50.
```

---

## 8. Configuration

### 8.1 ProviderProfile fields

Add fields to `ProviderProfile`:

```python
class ProviderProfile(BaseModel):
    provider: ProviderKind
    base_url: str | None = None
    api_key_env_var: str | None = None

    supports_chat: bool = True
    supports_embeddings: bool = False

    supports_batch: bool = False
    default_execution_mode: Literal["sync", "batch", "auto"] = "sync"

    batch_min_requests: int = 50
    batch_completion_window: str = "24h"

    timeout_seconds: int = 60
    max_retries: int = 3

    max_concurrency: int = 3
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
```

### 8.2 ModelRoute overrides

Add optional fields to `ModelRoute`:

```python
class ModelRoute(BaseModel):
    id: str | None = None
    provider_profile: str
    model: str

    execution_mode: Literal["sync", "batch", "auto"] | None = None
    batch_min_requests: int | None = None
```

Resolution precedence:

```text
ModelRoute override
  > ProviderProfile default
  > AppSettings default
```

### 8.3 Example `.env`

```env
OPENAI_API_KEY=sk-...

PROVIDER_PROFILES__OPENAI_TEACHER__PROVIDER=openai
PROVIDER_PROFILES__OPENAI_TEACHER__BASE_URL=https://api.openai.com/v1
PROVIDER_PROFILES__OPENAI_TEACHER__API_KEY_ENV_VAR=OPENAI_API_KEY
PROVIDER_PROFILES__OPENAI_TEACHER__SUPPORTS_CHAT=true
PROVIDER_PROFILES__OPENAI_TEACHER__SUPPORTS_EMBEDDINGS=false
PROVIDER_PROFILES__OPENAI_TEACHER__SUPPORTS_BATCH=true
PROVIDER_PROFILES__OPENAI_TEACHER__DEFAULT_EXECUTION_MODE=batch
PROVIDER_PROFILES__OPENAI_TEACHER__BATCH_MIN_REQUESTS=50
PROVIDER_PROFILES__OPENAI_TEACHER__BATCH_COMPLETION_WINDOW=24h
```

### 8.4 Example job route

```yaml
teacher:
  provider_profile: openai-teacher
  model: gpt-5.5
  execution_mode: batch

student:
  provider_profile: local-qwen-lb
  model: Qwen_Qwen3.5-4B-Q5_K_L
  execution_mode: sync

embedding:
  provider_profile: local-mxbai-embed
  model: mxbai-embed-large-v1.Q6_K
  execution_mode: sync
```

---

## 9. Execution Mode Selection

Add `ExecutionModeResolver`.

```python
class ExecutionModeResolver:
    def resolve(
        self,
        *,
        provider_profile: ProviderProfile,
        model_route: ModelRoute,
        stage: str,
        request_count: int,
    ) -> Literal["sync", "batch"]:
        ...
```

Rules:

```text
if execution_mode == "sync":
  return sync

if execution_mode == "batch":
  require provider_profile.supports_batch
  require stage supports batch
  return batch

if execution_mode == "auto":
  if provider supports batch
     and stage supports batch
     and request_count >= batch_min_requests:
       return batch
  else:
       return sync
```

MVP batch-supported stages:

```text
extracting_rules
```

Future batch-supported stages:

```text
reviewing_rule_conflicts
evaluating_baseline
evaluating_distilled
judge_rubric_scoring
embedding_rules
```

---

## 10. Pipeline State Changes

Batch execution is asynchronous. The pipeline must persist enough state to resume later.

Add or support these stage states:

```text
submitting_provider_batch
waiting_on_provider_batch
downloading_provider_batch_results
processing_provider_batch_results
provider_batch_failed
provider_batch_expired
provider_batch_cancelled
```

For the `extracting_rules` stage, flow changes from:

```text
call teacher per case
parse responses
continue
```

to:

```text
build extraction batch input JSONL
upload input file
create OpenAI batch
persist batch metadata
mark stage waiting_on_provider_batch
poll batch status
download output/error files
parse responses by custom_id
write micro-rule artifacts
continue pipeline
```

---

## 11. Database Changes

### 11.1 provider_batch_jobs table

Add table:

```sql
create table if not exists provider_batch_jobs (
    id uuid primary key,
    job_id uuid not null references distillation_jobs(id),

    stage text not null,
    role text not null,
    provider_profile text not null,
    provider text not null,
    model text not null,

    execution_mode text not null default 'batch',

    provider_batch_id text,
    provider_input_file_id text,
    provider_output_file_id text,
    provider_error_file_id text,

    endpoint text not null,
    completion_window text not null default '24h',

    status text not null default 'created',
    request_count int not null default 0,
    completed_count int not null default 0,
    failed_count int not null default 0,

    input_artifact_path text,
    output_artifact_path text,
    error_artifact_path text,

    submitted_at timestamptz,
    completed_at timestamptz,
    failed_at timestamptz,
    expired_at timestamptz,
    cancelled_at timestamptz,

    error_message text,
    metadata jsonb not null default '{}',

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    unique (job_id, stage, role, provider_profile, model)
);
```

Indexes:

```sql
create index if not exists idx_provider_batch_jobs_job_id
on provider_batch_jobs(job_id);

create index if not exists idx_provider_batch_jobs_status
on provider_batch_jobs(status);

create index if not exists idx_provider_batch_jobs_provider_batch_id
on provider_batch_jobs(provider_batch_id);
```

### 11.2 Job status detail

Job status should expose batch state:

```json
{
  "status": "running",
  "stage": "waiting_on_provider_batch",
  "stage_detail": {
    "provider": "openai",
    "role": "teacher",
    "batch_id": "batch_abc123",
    "request_count": 500,
    "completed_count": 0,
    "failed_count": 0
  }
}
```

---

## 12. Batch Request Format

Each OpenAI Batch input line is JSONL.

For chat completions:

```json
{
  "custom_id": "extracting_rules:job_123:case_000042",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-5.5",
    "messages": [
      {
        "role": "system",
        "content": "..."
      },
      {
        "role": "user",
        "content": "..."
      }
    ],
    "temperature": 0,
    "response_format": {
      "type": "json_object"
    }
  }
}
```

If using the Responses API later:

```json
{
  "custom_id": "extracting_rules:job_123:case_000042",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-5.5",
    "input": [
      {
        "role": "system",
        "content": "..."
      },
      {
        "role": "user",
        "content": "..."
      }
    ]
  }
}
```

MVP recommendation:

```text
Use the same endpoint as the current OpenAI provider adapter.
If the current provider uses chat completions, start with /v1/chat/completions.
Add /v1/responses later if needed.
```

---

## 13. custom_id Contract

Every batch request must have a stable `custom_id`.

Recommended format:

```text
{stage}:{job_id}:{case_id}
```

Examples:

```text
extracting_rules:job_123:banking77_train_000042
evaluating_baseline:job_123:local_qwen_4b:banking77_validation_000042
evaluating_distilled:job_123:local_qwen_4b:hdbscan:banking77_validation_000042
```

For extraction MVP:

```text
extracting_rules:{job_id}:{case_id}
```

The parser must not rely on output order. It must map results by `custom_id`.

---

## 14. Artifacts

Write batch artifacts for auditability and reproducibility.

```text
.rulekiln/runs/{job_id}/provider_batches/
  extracting_rules_openai_teacher_input.jsonl
  extracting_rules_openai_teacher_output.jsonl
  extracting_rules_openai_teacher_errors.jsonl
  extracting_rules_openai_teacher_manifest.json
```

Manifest example:

```json
{
  "job_id": "job_123",
  "stage": "extracting_rules",
  "role": "teacher",
  "provider": "openai",
  "model": "gpt-5.5",
  "endpoint": "/v1/chat/completions",
  "request_count": 500,
  "provider_batch_id": "batch_abc123",
  "provider_input_file_id": "file_abc123",
  "provider_output_file_id": "file_xyz123",
  "provider_error_file_id": null,
  "completion_window": "24h",
  "status": "completed"
}
```

Privacy note:

```text
Batch input artifacts may contain prompt/case content.
For SaaS, artifact retention and raw payload capture must be configurable.
```

For local open source benchmarking, writing raw batch JSONL is acceptable.

---

## 15. OpenAI Batch Client

Add a client wrapper:

```text
src/rulekiln/providers/openai_batch.py
```

Responsibilities:

```text
write input JSONL
upload file with purpose=batch
create batch
retrieve batch status
download output file
download error file if present
cancel batch if requested
parse output JSONL
```

Suggested interface:

```python
class OpenAIBatchClient:
    async def submit_batch(
        self,
        *,
        endpoint: str,
        input_jsonl_path: Path,
        completion_window: str,
        metadata: dict[str, str] | None = None,
    ) -> OpenAIBatchSubmission:
        ...

    async def retrieve_batch(self, *, batch_id: str) -> OpenAIBatchStatus:
        ...

    async def download_results(self, *, output_file_id: str, output_path: Path) -> None:
        ...

    async def download_errors(self, *, error_file_id: str, error_path: Path) -> None:
        ...
```

---

## 16. Worker Behavior

### 16.1 Submit Batch

When the stage chooses batch execution:

```python
batch_job = await batch_repo.get_existing_or_create(...)

if not batch_job.provider_batch_id:
    input_path = await build_extraction_batch_input(...)
    submission = await openai_batch_client.submit_batch(...)
    await batch_repo.mark_submitted(...)
    await jobs_repo.update_stage(job_id, "waiting_on_provider_batch")
    return
```

The worker should not block for hours inside one process if the architecture supports re-queueing.

Recommended behavior:

```text
Submit batch.
Persist provider_batch_id.
Mark job/stage as waiting_on_provider_batch.
Requeue job for later polling.
```

### 16.2 Poll Batch

Polling worker behavior:

```python
batch = await openai_batch_client.retrieve_batch(batch_id)

if batch.status in ["validating", "in_progress", "finalizing"]:
    update counts
    schedule next poll
    return

if batch.status == "completed":
    download output
    process output
    continue pipeline

if batch.status in ["failed", "expired", "cancelled"]:
    mark stage failed
    apply retry/fallback policy
```

### 16.3 Poll Interval

Default:

```text
initial_poll_delay_seconds = 60
max_poll_interval_seconds = 600
```

Use stepped backoff:

```text
1 min
2 min
5 min
10 min
10 min
...
```

---

## 17. Retry and Fallback Policy

Batch failure handling must be explicit.

Recommended config:

```python
class BatchExecutionPolicy(BaseModel):
    retry_failed_batch_once: bool = True
    fallback_to_sync_on_batch_failure: bool = False
    allow_partial_results: bool = False
    max_failed_request_ratio: float = 0.0
```

MVP behavior:

```text
If whole batch fails:
  retry once if no prior retry
  otherwise fail stage

If batch expires:
  fail stage

If some individual requests fail:
  fail stage unless allow_partial_results=true
```

Do not silently continue with missing teacher extraction results.

Future behavior:

```text
retry only failed custom_ids
```

---

## 18. Result Parsing

OpenAI Batch output lines must be parsed by `custom_id`.

Expected successful output line shape is provider-specific, but generally includes:

```json
{
  "custom_id": "extracting_rules:job_123:case_000042",
  "response": {
    "status_code": 200,
    "body": {
      "choices": [
        {
          "message": {
            "content": "{...}"
          }
        }
      ],
      "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168
      }
    }
  }
}
```

Parser requirements:

```text
read every output line
extract custom_id
map to case_id
extract content
extract usage
validate structured output
write micro-rules with source_case_id
log model_call event for each completed request
```

Error file requirements:

```text
read every error line
extract custom_id if present
map to case_id if possible
record failed model_call event
surface errors in job artifact/report
```

---

## 19. Usage and Cost Tracking Integration

Batch support must integrate with model usage/cost tracking.

For each successful response line:

```text
log one model_call event
role = teacher
stage = extracting_rules
provider = openai
model = gpt-5.5
processing_mode = batch
usage = response.usage
cost = pricing_service.calculate(..., processing_mode=batch)
```

Add `processing_mode` to `ModelCallRecord`:

```python
processing_mode: Literal["sync", "batch"] = "sync"
```

Pricing service should support batch discount:

```yaml
pricing:
  openai:
    gpt-5.5:
      input_per_1m_tokens_usd: "5.00"
      output_per_1m_tokens_usd: "30.00"
      batch_discount: "0.50"
      source: "manual"
```

Cost formula:

```text
standard_cost = input_cost + output_cost
batch_cost = standard_cost * batch_discount
```

If `batch_discount = 0.50`, cost is 50% of standard.

Token/cost summary should break down by processing mode:

```json
{
  "by_processing_mode": {
    "batch": {
      "calls": 500,
      "estimated_cost_usd": "2.050000"
    },
    "sync": {
      "calls": 20,
      "estimated_cost_usd": "0.300000"
    }
  }
}
```

---

## 20. MLflow Logging

Log batch artifacts:

```text
provider_batches/*_manifest.json
provider_batches/*_input.jsonl
provider_batches/*_output.jsonl
provider_batches/*_errors.jsonl
```

For SaaS, raw input logging may be disabled.

Log params:

```text
teacher_execution_mode = batch
teacher_batch_id = batch_abc123
teacher_batch_endpoint = /v1/chat/completions
teacher_batch_completion_window = 24h
```

Log metrics:

```text
batch.teacher.request_count
batch.teacher.completed_count
batch.teacher.failed_count
batch.teacher.cost_usd
batch.teacher.input_tokens
batch.teacher.output_tokens
```

---

## 21. API and UI Changes

### 21.1 API

Job status should expose batch state:

```json
{
  "job_id": "job_123",
  "status": "running",
  "stage": "waiting_on_provider_batch",
  "batch": {
    "stage": "extracting_rules",
    "role": "teacher",
    "provider": "openai",
    "model": "gpt-5.5",
    "status": "in_progress",
    "request_count": 500,
    "completed_count": 200,
    "failed_count": 0
  }
}
```

### 21.2 UI

Job status page should show:

```text
Waiting on OpenAI Batch
Stage: extracting_rules
Requests: 500
Completed: 200
Failed: 0
Submitted: 2026-05-25 13:40
Completion window: 24h
```

Results page should show:

```text
Teacher execution mode: Batch
Teacher batch cost discount applied: yes
Teacher cost: $X
```

---

## 22. Cancellation

If RuleKiln supports job cancellation, then cancellation should attempt to cancel active provider batches.

Behavior:

```text
if job has active provider_batch_id:
  call OpenAI batch cancel
  mark provider_batch_job status = cancelling/cancelled
  mark RuleKiln job cancelled
```

If provider cancellation fails:

```text
log warning
mark RuleKiln job cancellation requested
continue polling until provider status is terminal
```

---

## 23. Security and Privacy

Batch input files contain prompts and case data.

For OSS:

```text
write batch JSONL artifacts locally
document that they may contain input data
```

For SaaS:

```text
make raw batch artifact retention configurable
optionally store only hashes/manifests
encrypt artifacts at rest
respect workspace data-retention settings
do not expose batch files across tenants
```

Do not log:

```text
OpenAI API keys
Authorization headers
raw payloads in application logs
raw prompts in OpenTelemetry spans by default
```

---

## 24. Testing Requirements

### 24.1 Unit Tests

```text
test_execution_mode_resolver_uses_batch_when_enabled_and_threshold_met
test_execution_mode_resolver_uses_sync_below_threshold
test_execution_mode_resolver_rejects_batch_when_provider_unsupported
test_build_extraction_batch_jsonl_writes_one_line_per_case
test_batch_custom_ids_are_stable_and_unique
test_batch_parser_maps_results_by_custom_id_not_order
test_batch_parser_extracts_usage
test_batch_parser_records_failed_lines
test_pricing_service_applies_batch_discount
```

### 24.2 Worker Tests

```text
test_worker_submits_batch_for_teacher_extraction
test_worker_persists_provider_batch_id
test_worker_marks_stage_waiting_on_provider_batch
test_worker_polls_in_progress_batch_without_advancing_pipeline
test_worker_downloads_completed_batch_output
test_worker_processes_completed_batch_results
test_worker_fails_stage_on_expired_batch
test_worker_retries_failed_batch_once_when_configured
```

### 24.3 Integration Tests

Use fake OpenAI batch client.

```text
test_fake_batch_client_complete_flow
test_fake_batch_client_partial_failure_flow
test_batch_results_create_micro_rules
test_batch_model_call_events_are_logged
test_token_cost_summary_includes_batch_mode
```

Do not call real OpenAI Batch API in CI.

### 24.4 Optional Manual Test

Add a manual test script:

```text
scripts/manual_openai_batch_smoke_test.py
```

It should:

```text
create tiny batch with 2 requests
submit to OpenAI
poll until completed
download result
print usage
```

This should require an explicit env var:

```text
RUN_OPENAI_BATCH_SMOKE_TEST=true
```

---

## 25. Acceptance Criteria

This feature is complete when:

1. Provider profiles can declare `supports_batch`.
2. Model routes can request `execution_mode=batch`.
3. RuleKiln can choose batch execution for OpenAI teacher extraction.
4. RuleKiln writes an OpenAI-compatible batch JSONL input file.
5. Batch requests include stable unique `custom_id` values.
6. RuleKiln uploads the batch input file and creates an OpenAI batch.
7. RuleKiln persists provider batch metadata.
8. Job status shows waiting-on-batch state.
9. Worker can poll batch status and resume after completion.
10. Worker downloads and parses batch output files.
11. Batch results are mapped by `custom_id`, not output order.
12. Micro-rules are created from completed batch responses.
13. Failed batch lines are recorded and surfaced.
14. Model call events are logged per completed batch response.
15. Usage and cost tracking supports `processing_mode=batch`.
16. Pricing service applies configured batch discount.
17. MLflow logs batch artifacts/metrics.
18. UI/API exposes batch status and batch cost.
19. CI tests use a fake batch client and do not call real OpenAI.
20. Documentation explains when to use sync vs batch.

---

## 26. Implementation Tasks

```text
BATCH001 Add supports_batch/default_execution_mode fields to ProviderProfile.
BATCH002 Add execution_mode/batch_min_requests overrides to ModelRoute.
BATCH003 Add ExecutionModeResolver.
BATCH004 Add provider_batch_jobs migration.
BATCH005 Add OpenAIBatchClient.
BATCH006 Add batch input JSONL builder for teacher extraction.
BATCH007 Add stable custom_id builder/parser.
BATCH008 Add batch submission path in extracting_rules stage.
BATCH009 Add waiting_on_provider_batch stage status.
BATCH010 Add batch polling/resume logic in worker.
BATCH011 Add batch output downloader.
BATCH012 Add batch output parser for chat completions.
BATCH013 Add batch error parser.
BATCH014 Convert completed batch responses into micro-rules.
BATCH015 Add processing_mode to ModelCallRecord.
BATCH016 Update PricingService to support batch_discount.
BATCH017 Log model_call events for batch responses.
BATCH018 Write provider batch artifacts and manifest.
BATCH019 Log batch artifacts/metrics to MLflow.
BATCH020 Add API batch status fields.
BATCH021 Add UI batch status section.
BATCH022 Add cancellation support for active batch jobs if job cancellation exists.
BATCH023 Add fake batch client for tests.
BATCH024 Add unit/worker/integration tests.
BATCH025 Add docs for OpenAI Batch execution mode.
```

---

## 27. Recommended Implementation Order

Implement in this order:

1. Schema/config fields.
2. ExecutionModeResolver.
3. provider_batch_jobs table.
4. OpenAIBatchClient.
5. Batch JSONL builder for extracting_rules.
6. Submit batch and persist metadata.
7. Poll status and resume worker.
8. Download and parse results.
9. Create micro-rules from batch results.
10. Integrate usage/cost tracking.
11. Add artifacts and MLflow logging.
12. Add UI/API batch status.
13. Add tests and documentation.

---

## 28. README Language

Suggested README language:

```md
### Batch Teacher Runs

For large datasets, RuleKiln can run OpenAI teacher extraction through OpenAI Batch API. Batch mode is asynchronous, but it can significantly reduce teacher-call cost for non-interactive prompt-hardening jobs.

Use batch mode when:
- the job has many independent teacher calls
- immediate results are not required
- cost matters more than latency

Use sync mode when:
- debugging
- running small jobs
- using local providers
- the UI needs immediate feedback
```

---

## 29. Final Recommendation

Add OpenAI Batch support, but do it stage-by-stage.

Start with:

```text
OpenAI teacher extraction via Batch API
```

Do not batch local student evaluation. Your local Qwen servers behind HAProxy should remain synchronous.

The best initial RuleKiln benchmark setup is:

```text
Teacher:
  OpenAI GPT-5.5 via Batch

Student:
  local Qwen 4B via HAProxy sync endpoint

Embedding:
  local mxbai embedding server
```

This will give RuleKiln a strong benchmark and SaaS cost story:

```text
The hardening run uses a powerful teacher at batch pricing.
The hardened prompt is evaluated on a local/cheap student.
The output includes quality delta and hardening cost.
```
