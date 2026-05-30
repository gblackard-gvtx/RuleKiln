# RuleKiln Model Usage and Cost Tracking Spec

## 1. Purpose

RuleKiln must track model usage and estimated cost for every distillation run.

This is required because RuleKiln’s core value proposition depends on proving the economic tradeoff:

```text
Use an expensive teacher model during prompt hardening.
Use a cheaper/local/smaller student model during production.
Show quality improvement and hardening cost.
```

Without usage and cost tracking, RuleKiln can show that a student improved, but it cannot show what the hardening run cost or whether the optimization is economically attractive.

---

## 2. Current State

The current implementation does **not** track model usage or cost end-to-end.

Known current state:

- `log_model_call()` exists in `events.py`.
- `log_model_call()` has parameters for `prompt_tokens` and `completion_tokens`.
- `log_model_call()` is not wired into the pipeline.
- Provider clients do not consistently extract token usage from provider responses.
- Provider clients do not return a structured usage object.
- `DistillationJob` does not store token/cost totals.
- MLflow does not log cost metrics.
- No token/cost summary artifact is written.
- UI does not display teacher/student/embedding/judge usage or cost.

This spec makes model usage and cost tracking a first-class runtime concern.

---

## 3. Decision

Implement model usage and cost tracking as a **provider-layer concern**.

Do not sprinkle cost calculations throughout pipeline stages.

The desired flow is:

```text
Provider client call
  -> extract token usage / latency / model metadata
  -> return ModelCallResult with usage
  -> log model_call event
  -> aggregate per-job totals
  -> write token_cost_summary.json
  -> update database summary fields
  -> log MLflow metrics/artifacts
  -> display in UI/API
```

---

## 4. Goals

This change should allow RuleKiln to answer:

- How much did this hardening run cost?
- How much of the cost came from the teacher?
- How much came from the student evaluations?
- How much came from embeddings?
- How much came from judge/rubric calls?
- How many provider calls were made?
- How many input/output tokens were used?
- Which provider/model consumed the most tokens?
- Which stage was most expensive?
- Which student model had the best quality/cost profile?
- What is the estimated cost of using this hardened prompt with a local or cheaper production student?

---

## 5. Non-Goals

Do not implement these in the first pass:

- real-time billing
- Stripe integration
- customer invoicing
- GPU amortization for local models
- provider price auto-sync
- exact tokenizer parity for every local model
- organization-level spend limits
- budget enforcement
- cost forecasting dashboards
- production traffic monitoring

The MVP should track **per-run model usage and estimated hardening cost**.

---

## 6. Core Concepts

### 6.1 Model Roles

RuleKiln model calls should be tagged by role:

```text
teacher
student
embedding
judge
```

Role meaning:

```text
teacher:
  Extracts rules, synthesizes rules, may resolve conflicts.

student:
  Runs baseline and hardened prompts for evaluation.

embedding:
  Embeds micro-rules or rule text for clustering.

judge:
  Scores rubric-based outputs or performs conflict review when not using the teacher.
```

### 6.2 Stages

Every model call should be associated with a pipeline stage:

```text
extracting_rules
embedding_rules
synthesizing_rules
reviewing_rule_conflicts
evaluating_baseline
evaluating_distilled
analyzing_failures
checking_quality_gates
```

### 6.3 Strategy

Student evaluation calls should include a strategy when applicable:

```text
baseline
dbscan
hdbscan
```

### 6.4 Student ID

For multi-student classroom runs, student model calls must include:

```text
student_id
```

Example:

```text
local_qwen_4b
nova_lite
local_llama_8b
```

---

## 7. Data Models

### 7.1 ModelUsage

Add a shared usage model.

```python
class ModelUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None

    estimated: bool = False
```

Notes:

- `prompt_tokens` and `completion_tokens` support OpenAI-style naming.
- `input_tokens` and `output_tokens` should be RuleKiln’s normalized internal naming.
- For OpenAI-compatible APIs:
  - `input_tokens = prompt_tokens`
  - `output_tokens = completion_tokens`
- If provider usage is missing, estimate usage and set `estimated=True`.

### 7.2 ModelCallCost

```python
class ModelCallCost(BaseModel):
    input_cost_usd: Decimal = Decimal("0")
    output_cost_usd: Decimal = Decimal("0")
    total_cost_usd: Decimal = Decimal("0")
    pricing_source: str | None = None
    estimated: bool = True
```

Notes:

- Cost is estimated by default.
- Cost should be represented with `Decimal`, not float.
- Local/open-source models can default to zero token cost.

### 7.3 ModelCallRecord

```python
class ModelCallRecord(BaseModel):
    job_id: UUID

    stage: str
    role: Literal["teacher", "student", "embedding", "judge"]

    provider_profile: str
    provider: str
    model: str

    student_id: str | None = None
    strategy: str | None = None
    case_id: str | None = None

    usage: ModelUsage
    cost: ModelCallCost

    latency_ms: int | None = None
    status: Literal["success", "failed"]
    error_type: str | None = None
```

### 7.4 Provider Result Types

Provider clients should return result objects with usage.

For chat:

```python
class ChatCompletionResult(BaseModel):
    content: str
    parsed: Any | None = None
    usage: ModelUsage | None = None
    raw_model: str | None = None
    provider_response_id: str | None = None
```

For embeddings:

```python
class EmbeddingResult(BaseModel):
    embeddings: list[list[float]]
    usage: ModelUsage | None = None
    raw_model: str | None = None
    provider_response_id: str | None = None
```

---

## 8. Provider Usage Extraction

### 8.1 OpenAI Chat

Extract from response usage:

```python
usage = response.usage

ModelUsage(
    prompt_tokens=usage.prompt_tokens,
    completion_tokens=usage.completion_tokens,
    total_tokens=usage.total_tokens,
    input_tokens=usage.prompt_tokens,
    output_tokens=usage.completion_tokens,
    estimated=False,
)
```

### 8.2 OpenAI-Compatible Chat

Attempt to extract the same usage fields.

If missing:

```text
estimate tokens
usage.estimated = true
```

This is expected for some local servers.

### 8.3 OpenAI Embeddings

Extract usage when available:

```python
usage = response.usage

ModelUsage(
    prompt_tokens=usage.prompt_tokens,
    total_tokens=usage.total_tokens,
    input_tokens=usage.prompt_tokens,
    output_tokens=0,
    estimated=False,
)
```

### 8.4 llama.cpp / Local OpenAI-Compatible Servers

Local llama.cpp servers may return usage inconsistently depending on version and endpoint.

Behavior:

```text
if response contains usage:
  use provider usage
else:
  estimate usage locally
```

Local token cost should default to zero unless configured otherwise.

### 8.5 Failed Calls

For failed calls:

- Log the call event with `status="failed"`.
- Include provider/model/stage/role metadata.
- Include latency if available.
- Include `error_type`.
- Usage may be null or estimated if request text is available.

---

## 9. Token Estimation

When provider usage is missing, estimate token counts.

MVP fallback:

```text
estimated_tokens = ceil(character_count / 4)
```

This is approximate but good enough for MVP cost visibility.

Future improvement:

```text
provider/model-specific tokenizers
tiktoken for OpenAI-compatible tokenization
llama.cpp tokenizer endpoint if available
Bedrock/Anthropic usage extraction where available
```

Rules:

- Mark estimated usage with `estimated=True`.
- Do not mix estimated and actual totals without labeling.
- Aggregated summaries should include whether any usage was estimated.

---

## 10. Pricing Configuration

Do not hard-code model prices in code.

Add configurable pricing.

Recommended file:

```text
config/model_pricing.yaml
```

Example:

```yaml
pricing:
  openai:
    gpt-5.5:
      input_per_1m_tokens_usd: "0.00"
      output_per_1m_tokens_usd: "0.00"
      effective_date: "2026-05-01"
      source: "manual"

  openai_compatible:
    Qwen_Qwen3.5-4B-Q5_K_L:
      input_per_1m_tokens_usd: "0.00"
      output_per_1m_tokens_usd: "0.00"
      source: "local"

    mxbai-embed-large-v1.Q6_K:
      input_per_1m_tokens_usd: "0.00"
      output_per_1m_tokens_usd: "0.00"
      source: "local"
```

Notes:

- Prices change over time. Treat this file as configuration.
- Use strings for decimal values to avoid float precision issues.
- If price is missing, cost should be zero or unknown depending on settings.

Recommended behavior for missing pricing:

```text
default MVP:
  cost = 0
  cost.estimated = true
  pricing_source = "missing_pricing_config"

strict future mode:
  fail validation if paid provider has no pricing config
```

---

## 11. Cost Calculation

Add `PricingService`.

```python
class PricingService:
    def calculate(
        self,
        *,
        provider: str,
        model: str,
        usage: ModelUsage,
    ) -> ModelCallCost:
        ...
```

Cost formula:

```text
input_cost = input_tokens / 1_000_000 * input_per_1m_tokens_usd
output_cost = output_tokens / 1_000_000 * output_per_1m_tokens_usd
total_cost = input_cost + output_cost
```

Use `Decimal`.

Example:

```python
input_cost = (
    Decimal(input_tokens)
    / Decimal("1000000")
    * Decimal(input_per_1m_tokens_usd)
)
```

---

## 12. Provider Call Tracking Wrapper

Add a wrapper around all provider calls.

Recommended location:

```text
src/rulekiln/providers/tracking.py
```

Example:

```python
async def tracked_chat_call(
    *,
    context: ModelCallContext,
    call: Callable[[], Awaitable[ChatCompletionResult]],
) -> ChatCompletionResult:
    started = time.monotonic()

    try:
        result = await call()
        latency_ms = int((time.monotonic() - started) * 1000)

        usage = result.usage or estimate_usage(context=context)
        cost = pricing_service.calculate(
            provider=context.provider,
            model=context.model,
            usage=usage,
        )

        await model_call_logger.log_model_call(
            ModelCallRecord(
                job_id=context.job_id,
                stage=context.stage,
                role=context.role,
                provider_profile=context.provider_profile,
                provider=context.provider,
                model=context.model,
                student_id=context.student_id,
                strategy=context.strategy,
                case_id=context.case_id,
                usage=usage,
                cost=cost,
                latency_ms=latency_ms,
                status="success",
            )
        )

        return result

    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)

        await model_call_logger.log_failed_model_call(
            context=context,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )

        raise
```

All provider adapters must use this wrapper.

Do not rely on pipeline stages remembering to call `log_model_call()`.

---

## 13. ModelCallContext

Add a context object passed into provider calls.

```python
class ModelCallContext(BaseModel):
    job_id: UUID
    stage: str
    role: Literal["teacher", "student", "embedding", "judge"]

    provider_profile: str
    provider: str
    model: str

    student_id: str | None = None
    strategy: str | None = None
    case_id: str | None = None

    prompt_hash: str | None = None
    input_hash: str | None = None
```

This context supports tracking, observability, and later cost dashboards.

---

## 14. Database Changes

### 14.1 model_call_events Table

Add table:

```sql
create table if not exists model_call_events (
    id uuid primary key,
    job_id uuid not null references distillation_jobs(id),

    stage text not null,
    role text not null,
    provider_profile text not null,
    provider text not null,
    model text not null,

    student_id text,
    strategy text,
    case_id text,

    input_tokens int,
    output_tokens int,
    total_tokens int,
    usage_estimated boolean not null default false,

    input_cost_usd numeric(12, 6) not null default 0,
    output_cost_usd numeric(12, 6) not null default 0,
    total_cost_usd numeric(12, 6) not null default 0,
    cost_estimated boolean not null default true,
    pricing_source text,

    latency_ms int,
    status text not null,
    error_type text,

    created_at timestamptz not null default now()
);
```

Indexes:

```sql
create index if not exists idx_model_call_events_job_id
on model_call_events(job_id);

create index if not exists idx_model_call_events_job_role
on model_call_events(job_id, role);

create index if not exists idx_model_call_events_job_stage
on model_call_events(job_id, stage);

create index if not exists idx_model_call_events_job_student
on model_call_events(job_id, student_id);
```

### 14.2 distillation_jobs Summary Fields

Add job-level aggregate fields:

```sql
alter table distillation_jobs
add column if not exists total_input_tokens bigint default 0,
add column if not exists total_output_tokens bigint default 0,
add column if not exists total_tokens bigint default 0,
add column if not exists estimated_total_cost_usd numeric(12, 6) default 0,
add column if not exists teacher_cost_usd numeric(12, 6) default 0,
add column if not exists student_cost_usd numeric(12, 6) default 0,
add column if not exists embedding_cost_usd numeric(12, 6) default 0,
add column if not exists judge_cost_usd numeric(12, 6) default 0;
```

Aggregates should be computed from `model_call_events` at the end of the job.

---

## 15. Aggregation

Add `ModelUsageAggregator`.

Recommended location:

```text
src/rulekiln/usage/aggregator.py
```

Responsibilities:

- aggregate calls by job
- aggregate calls by role
- aggregate calls by model
- aggregate calls by stage
- aggregate calls by student
- update `distillation_jobs` summary fields
- write `token_cost_summary.json`

Aggregation dimensions:

```text
total
by_role
by_model
by_stage
by_student
```

---

## 16. Artifact Output

Always write:

```text
.rulekiln/runs/{job_id}/metadata/token_cost_summary.json
```

Example:

```json
{
  "job_id": "job_123",
  "total": {
    "calls": 1420,
    "input_tokens": 125000,
    "output_tokens": 42000,
    "total_tokens": 167000,
    "estimated_cost_usd": "4.820000",
    "has_estimated_usage": true,
    "has_missing_pricing": false
  },
  "by_role": {
    "teacher": {
      "calls": 500,
      "input_tokens": 90000,
      "output_tokens": 30000,
      "total_tokens": 120000,
      "estimated_cost_usd": "4.100000"
    },
    "student": {
      "calls": 900,
      "input_tokens": 30000,
      "output_tokens": 10000,
      "total_tokens": 40000,
      "estimated_cost_usd": "0.000000"
    },
    "embedding": {
      "calls": 20,
      "input_tokens": 5000,
      "output_tokens": 0,
      "total_tokens": 5000,
      "estimated_cost_usd": "0.000000"
    }
  },
  "by_model": {
    "openai:gpt-5.5": {
      "roles": ["teacher"],
      "calls": 500,
      "input_tokens": 90000,
      "output_tokens": 30000,
      "estimated_cost_usd": "4.100000"
    },
    "openai_compatible:Qwen_Qwen3.5-4B-Q5_K_L": {
      "roles": ["student"],
      "calls": 900,
      "input_tokens": 30000,
      "output_tokens": 10000,
      "estimated_cost_usd": "0.000000"
    }
  },
  "by_student": {
    "local_qwen_4b": {
      "calls": 900,
      "input_tokens": 30000,
      "output_tokens": 10000,
      "estimated_cost_usd": "0.000000"
    }
  }
}
```

---

## 17. MLflow Logging

Log metrics:

```python
mlflow.log_metric("cost.total_usd", total_cost)
mlflow.log_metric("cost.teacher_usd", teacher_cost)
mlflow.log_metric("cost.student_usd", student_cost)
mlflow.log_metric("cost.embedding_usd", embedding_cost)
mlflow.log_metric("cost.judge_usd", judge_cost)

mlflow.log_metric("tokens.total", total_tokens)
mlflow.log_metric("tokens.input", input_tokens)
mlflow.log_metric("tokens.output", output_tokens)

mlflow.log_metric("tokens.teacher.total", teacher_tokens)
mlflow.log_metric("tokens.student.total", student_tokens)
mlflow.log_metric("tokens.embedding.total", embedding_tokens)
mlflow.log_metric("tokens.judge.total", judge_tokens)
```

Log artifact:

```text
metadata/token_cost_summary.json
```

Log params:

```python
mlflow.log_param("teacher_provider", teacher_provider)
mlflow.log_param("teacher_model", teacher_model)
mlflow.log_param("student_provider", student_provider)
mlflow.log_param("student_model", student_model)
mlflow.log_param("embedding_provider", embedding_provider)
mlflow.log_param("embedding_model", embedding_model)
```

For multiple students, use stable metric names:

```text
students.local_qwen_4b.tokens.total
students.local_qwen_4b.cost_usd
```

---

## 18. UI Changes

The job results page should show:

```text
Total estimated cost
Teacher cost
Student cost
Embedding cost
Judge cost
Total model calls
Total tokens
Input tokens
Output tokens
Has estimated usage?
Has missing pricing?
```

For local students, display:

```text
Student token cost: $0.00
```

But still show:

```text
student calls
student tokens
student latency
```

This makes local runtime savings visible.

Suggested UI section:

```text
Run Cost and Usage

Total estimated cost: $4.82
Teacher: $4.10
Student: $0.00
Embedding: $0.00
Judge: $0.72

Total calls: 1,420
Total tokens: 167,000
```

---

## 19. API Changes

Job detail response should include usage summary:

```json
{
  "job_id": "job_123",
  "status": "completed",
  "usage": {
    "total_input_tokens": 125000,
    "total_output_tokens": 42000,
    "total_tokens": 167000,
    "estimated_total_cost_usd": "4.820000",
    "by_role": {
      "teacher": {
        "estimated_cost_usd": "4.100000"
      },
      "student": {
        "estimated_cost_usd": "0.000000"
      }
    }
  }
}
```

If this would bloat current responses, provide a separate endpoint:

```http
GET /distillation-jobs/{job_id}/usage
```

MVP recommendation:

```text
include summary fields on job detail
add detailed artifact link for full breakdown
```

---

## 20. Observability / OpenTelemetry

If OpenTelemetry is enabled, provider-call spans should include:

```text
rulekiln.role
rulekiln.stage
rulekiln.provider
rulekiln.provider_profile
rulekiln.model
rulekiln.student_id
rulekiln.strategy
rulekiln.case_id
llm.input_tokens
llm.output_tokens
llm.total_tokens
llm.cost_usd
llm.usage_estimated
llm.cost_estimated
```

Do not store raw prompts or raw case inputs in spans by default.

---

## 21. Privacy and Security

Do not log:

```text
raw prompts
raw case inputs
raw model outputs
API keys
secrets
provider request headers
customer PII
```

Model usage tracking should log metadata, not raw payloads.

Allowed:

```text
job_id
stage
role
provider
model
student_id
strategy
case_id
token counts
cost
latency
status
error type
hashes
artifact refs
```

---

## 22. Testing Requirements

### 22.1 Provider Usage Tests

```text
test_openai_provider_extracts_token_usage
test_openai_compatible_provider_extracts_usage_when_present
test_openai_compatible_provider_estimates_usage_when_missing
test_embedding_provider_extracts_usage_when_present
test_failed_provider_call_logs_failed_event
```

### 22.2 Cost Calculator Tests

```text
test_cost_calculator_uses_pricing_config
test_cost_calculator_uses_decimal_math
test_local_model_cost_defaults_to_zero
test_missing_pricing_marks_cost_estimated
test_embedding_cost_uses_input_tokens_only
```

### 22.3 Event Logging Tests

```text
test_model_call_event_is_logged_for_teacher_call
test_model_call_event_is_logged_for_student_call
test_model_call_event_is_logged_for_embedding_call
test_model_call_event_is_logged_for_judge_call
test_model_call_event_includes_stage_role_model_and_provider
```

### 22.4 Aggregation Tests

```text
test_job_usage_summary_aggregates_by_role
test_job_usage_summary_aggregates_by_model
test_job_usage_summary_aggregates_by_student
test_job_usage_summary_updates_distillation_job
test_token_cost_summary_artifact_is_written
```

### 22.5 MLflow Tests

```text
test_mlflow_logs_cost_metrics
test_mlflow_logs_token_metrics
test_mlflow_logs_token_cost_summary_artifact
```

### 22.6 UI/API Tests

```text
test_job_detail_includes_usage_summary
test_usage_endpoint_returns_role_breakdown
test_results_page_displays_total_cost
test_results_page_displays_teacher_student_costs
```

---

## 23. Acceptance Criteria

This change is complete when:

1. Provider clients return usage metadata when available.
2. Missing provider usage is estimated and marked as estimated.
3. Every provider call logs a model call event.
4. Failed provider calls log failed model call events.
5. Model call events are persisted to the database.
6. Model costs are calculated from configurable pricing.
7. Local model cost defaults to zero.
8. Job-level usage totals are aggregated.
9. `distillation_jobs` stores total token/cost summaries.
10. `token_cost_summary.json` is written for every completed job.
11. MLflow logs token and cost metrics.
12. UI/API exposes run-level cost and usage.
13. Tests cover provider usage extraction, cost calculation, event logging, and aggregation.
14. No raw prompts, inputs, outputs, or secrets are logged in usage events by default.

---

## 24. Implementation Tasks

```text
UTC001 Add ModelUsage schema.
UTC002 Add ModelCallCost schema.
UTC003 Add ModelCallRecord schema.
UTC004 Add ChatCompletionResult / EmbeddingResult usage fields.
UTC005 Update OpenAI chat provider to extract usage.
UTC006 Update OpenAI-compatible chat provider to extract or estimate usage.
UTC007 Update embedding provider to extract or estimate usage.
UTC008 Add token estimation fallback.
UTC009 Add model_pricing.yaml.
UTC010 Implement PricingService.
UTC011 Add ModelCallContext.
UTC012 Implement tracked_chat_call wrapper.
UTC013 Implement tracked_embedding_call wrapper.
UTC014 Wire provider adapters through tracking wrappers.
UTC015 Revive or replace log_model_call().
UTC016 Add model_call_events table migration.
UTC017 Add distillation_jobs token/cost summary columns.
UTC018 Implement ModelUsageAggregator.
UTC019 Write token_cost_summary.json artifact.
UTC020 Log token/cost metrics to MLflow.
UTC021 Add usage summary to job detail API.
UTC022 Add usage section to UI results page.
UTC023 Add OpenTelemetry provider-call attributes.
UTC024 Add provider usage tests.
UTC025 Add cost calculator tests.
UTC026 Add aggregation tests.
UTC027 Add MLflow tests.
UTC028 Update README/docs with cost tracking behavior.
```

---

## 25. Recommended Implementation Order

Implement in this order:

1. Provider result usage model.
2. OpenAI/OpenAI-compatible usage extraction.
3. Token estimation fallback.
4. Pricing config and cost calculator.
5. Provider tracking wrapper.
6. `model_call_events` table.
7. Job-level aggregation.
8. `token_cost_summary.json` artifact.
9. MLflow metrics.
10. UI/API display.

This order produces value early and keeps the pipeline changes controlled.

---

## 26. Product Value

Without cost tracking, RuleKiln can say:

```text
The student improved.
```

With cost tracking, RuleKiln can say:

```text
The student improved by +0.12 macro F1.
The hardening run cost $X in teacher calls.
The local student cost $0 in API runtime.
The resulting prompt is a candidate for cheaper production deployment.
```

That second story is the one that supports both open-source credibility and paid SaaS value.
