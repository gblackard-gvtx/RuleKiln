# RuleKiln Batch API Implementation

> **Implementation status (2026-06-04):** Phases 1 and 2 are complete. OpenAI batch extraction
> (`EXTRACTING_RULES` stage) is fully implemented and passing. See `docs/specs/features/rulekiln_openai_batch_execution_feature_spec.md` §3.1 for the full list of what shipped.
>
> Key implementation decisions that resolved the open questions in §15:
> - **A** — `BatchChatModelClient(ChatModelClient, ABC)` is the interface. ✅
> - **B** — Config-driven polling via `AppSettings.batch_poll_interval_seconds` (default 60 s). ✅
> - **C** — OpenAI per-call remains pydantic-ai; batch uses direct SDK. ✅
> - **D** — Bedrock deferred. ✅
> - **E** — Explicit opt-in (`batch_enabled=False` default everywhere). ✅
> - **F** — One batch per strategy variant (deferred to Phase 4). ✅
>
> **Class hierarchy resolved:** `OpenAIChatClient` directly subclasses `BatchChatModelClient`
> (merged, not a separate subclass). `_RateLimitedBatchChatClient` in `providers/chat/__init__.py`
> preserves `isinstance` checks through the rate-limiting wrapper.
>
> **`collect_batch` schema lookup:** `output_schema_class_name` is stored in `BatchJob.metadata_json`
> at submit time and passed into `collect_batch()` as a keyword argument by the worker.
>
> **Endpoint:** `/v1/responses` (not `/v1/chat/completions`). Response text extracted from
> `output[0].content[0].type="output_text"` via defensive helper `_extract_response_text()`.

--- Spec

## OpenAI First, Anthropic Second

This spec adapts the RuleKiln batch implementation plan while changing the rollout order: **OpenAI batch support ships first**, followed by **Anthropic batch support**. It preserves the core architecture from the plan: a `BatchChatModelClient` sub-interface, durable DBOS polling, `BatchJob` persistence, per-item idempotency via `StageMarker`, explicit opt-in gating, and fallback to sequential `complete_structured`.

OpenAI’s current Batch API is file-based: upload a JSONL file, create a batch, poll the batch, then retrieve output and error files. Structured Outputs should be used for schema adherence where possible. The Batch API is designed for asynchronous jobs with a 24-hour completion window and lower cost than synchronous calls.

---

## 1. Goals

Implement provider-agnostic batching in RuleKiln with **OpenAI as the first production provider**, then add Anthropic using the same internal contract.

The first shipped path should support:

1. OpenAI batch submission, polling, and collection.
2. Batch execution for `EXTRACTING_RULES`.
3. Durable DBOS polling using workflow-level `DBOS.sleep`.
4. Per-item result mapping back to existing `StageMarker` idempotency.
5. Sequential fallback for unsupported providers, small batches, partial failures, and full batch failures.
6. Batch cost accounting through existing pricing infrastructure.

Anthropic should be implemented second by conforming to the same `BatchChatModelClient` contract, with no worker-level redesign.

---

## 2. Non-goals

Do not implement Bedrock batch inference in this workstream.

Do not replace the existing synchronous `complete_structured` path.

Do not switch all OpenAI per-call traffic away from `pydantic-ai` during the OpenAI batch phase. OpenAI batch should use the direct OpenAI SDK while the existing per-call path remains unchanged.

Do not batch `REFINING_RULES`; it is sequential by design because each refinement iteration depends on the previous one.

---

## 3. Architecture Decision

### 3.1 Add `BatchChatModelClient`

Add a new abstract class in `providers/contracts.py`:

```python
class BatchChatModelClient(ChatModelClient, ABC):
    @abstractmethod
    async def submit_batch(
        self,
        items: list[BatchItem],
        config: ProviderConfig,
    ) -> str:
        ...

    @abstractmethod
    async def poll_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
    ) -> BatchPollStatus:
        ...

    @abstractmethod
    async def collect_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
    ) -> BatchResult:
        ...
```

`OpenAIChatClient` should subclass `BatchChatModelClient` first.

`AnthropicChatClient` should subclass `BatchChatModelClient` in the second provider phase.

`BedrockChatClient`, `FakeChatClient`, and any other non-batch clients remain `ChatModelClient` only.

The worker activates batching only when:

```python
use_batch = (
    provider_profile.batch_enabled
    and phase_config.batch_enabled
    and isinstance(chat_client, BatchChatModelClient)
    and len(pending_items) >= phase_config.batch_min_items
)
```

---

## 4. Batch Data Schemas

Create `schemas/batch.py`.

```python
class BatchItem(BaseModel):
    custom_id: str
    system_prompt: str
    user_prompt: str
    output_schema_json: dict[str, object]
    output_schema_class_name: str
```

```python
class BatchPollStatus(BaseModel):
    batch_id: str
    provider: str
    processing: bool
    succeeded_count: int
    errored_count: int
    total_count: int
    estimated_completion_at: datetime | None = None
```

```python
class BatchItemResult(BaseModel):
    custom_id: str
    status: Literal["succeeded", "errored", "expired"]
    result: ChatCompletionResult | None = None
    error_message: str | None = None
    usage: ModelUsage | None = None
```

```python
class BatchResult(BaseModel):
    batch_id: str
    provider: str
    status: Literal["completed", "partial", "failed", "expired"]
    items: list[BatchItemResult]
    succeeded_count: int
    errored_count: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal
```

---

## 5. Schema Registry

Create `providers/batch_schema_registry.py`.

Purpose: map `output_schema_class_name` back to a concrete `type[BaseModel]` during collection.

```python
_SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {}

def register_schema(cls: type[BaseModel]) -> type[BaseModel]:
    _SCHEMA_REGISTRY[cls.__name__] = cls
    return cls

def get_schema_class(name: str) -> type[BaseModel]:
    try:
        return _SCHEMA_REGISTRY[name]
    except KeyError as exc:
        raise BatchSchemaRegistryError(f"Unknown batch schema: {name}") from exc
```

All structured output models used in batch-eligible stages must be registered at import time.

---

## 6. Database Model

Add `BatchJob` SQLAlchemy model and Alembic migration.

Fields:

```text
id
job_id
stage
strategy
provider
provider_batch_id
status
submitted_at
completed_at
item_count
succeeded_count
errored_count
input_file_id
output_file_id
error_file_id
metadata_json
created_at
updated_at
```

`input_file_id`, `output_file_id`, and `error_file_id` are especially important for OpenAI because the provider’s batch lifecycle is file-based.

Recommended unique constraint:

```text
(job_id, stage, strategy, provider, provider_batch_id)
```

Recommended lookup index:

```text
(job_id, stage, strategy, status)
```

---

## 7. OpenAI Batch Provider

### 7.1 File

Create:

```text
providers/batch/openai_batch.py
```

or, if provider implementations live beside each other:

```text
providers/openai_batch.py
```

Preferred class:

```python
class OpenAIBatchChatClient(OpenAIChatClient, BatchChatModelClient):
    ...
```

or, if `OpenAIChatClient` should directly support both per-call and batch:

```python
class OpenAIChatClient(BatchChatModelClient):
    ...
```

The key requirement is that `isinstance(openai_client, BatchChatModelClient)` returns `True`.

### 7.2 SDK

Use direct OpenAI SDK calls for batch.

The existing OpenAI per-call implementation may remain on `pydantic-ai`. Batch should bypass it because the Batch API is provider-native and file-oriented.

### 7.3 Endpoint Choice

Use `/v1/responses` for new batch requests unless there is a compatibility blocker in the current `OpenAIChatClient` response parsing.

Reasoning:

The official Batch API supports `/v1/responses`, and OpenAI positions Responses as a primary path for text, structured output, tools, and multimodal workflows. Structured Outputs are supported by the Batch API and should be used for schema adherence.

Keep `/v1/chat/completions` as an implementation fallback if the existing prompt formatting or model configuration requires Chat Completions parity.

### 7.4 JSONL Request Shape

Each JSONL line should be one request:

```json
{
  "custom_id": "extracting_rules:case:123",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-5.4-mini",
    "input": [
      {
        "role": "system",
        "content": "..."
      },
      {
        "role": "user",
        "content": "..."
      }
    ],
    "text": {
      "format": {
        "type": "json_schema",
        "name": "ExtractedRules",
        "schema": {},
        "strict": true
      }
    }
  }
}
```

The actual `schema` value comes from:

```python
item.output_schema_json
```

The `custom_id` must be unique within the batch. Results may not be returned in the same order as inputs, so `custom_id` is the stable join key.

### 7.5 `submit_batch`

Responsibilities:

1. Serialize `BatchItem` objects to JSONL.
2. Upload the JSONL file with purpose `batch`.
3. Create the OpenAI batch with:
   - `input_file_id`
   - `endpoint="/v1/responses"`
   - `completion_window="24h"`
   - metadata containing `job_id`, `stage`, `strategy`.
4. Persist `BatchJob` before returning.
5. Return provider batch ID.

Pseudo-flow:

```python
async def submit_batch(
    self,
    items: list[BatchItem],
    config: ProviderConfig,
) -> str:
    jsonl_path = await write_openai_batch_jsonl(items, config)

    uploaded_file = await self.client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch",
    )

    batch = await self.client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "provider": "openai",
            "stage": config.stage,
            "strategy": config.strategy or "",
        },
    )

    return batch.id
```

### 7.6 `poll_batch`

Map OpenAI batch state into `BatchPollStatus`.

Processing states should include:

```text
validating
in_progress
finalizing
cancelling
```

Terminal states should include:

```text
completed
failed
expired
cancelled
```

Use OpenAI `request_counts` when present:

```python
BatchPollStatus(
    batch_id=batch.id,
    provider="openai",
    processing=batch.status in PROCESSING_STATUSES,
    succeeded_count=batch.request_counts.completed if batch.request_counts else 0,
    errored_count=batch.request_counts.failed if batch.request_counts else 0,
    total_count=batch.request_counts.total if batch.request_counts else 0,
    estimated_completion_at=None,
)
```

### 7.7 `collect_batch`

Responsibilities:

1. Retrieve the batch.
2. If `output_file_id` exists, download and parse output JSONL.
3. If `error_file_id` exists, download and parse error JSONL.
4. Convert each row to `BatchItemResult`.
5. Re-parse structured content through the schema registry.
6. Emit `ChatCompletionResult`.
7. Return aggregate `BatchResult`.

For successful output rows:

```python
schema_cls = get_schema_class(item.output_schema_class_name)
parsed = schema_cls.model_validate_json(response_text)
```

Then wrap in existing `ChatCompletionResult`.

For error rows:

```python
BatchItemResult(
    custom_id=custom_id,
    status="errored",
    error_message=provider_error_message,
)
```

If the batch status is `expired`, map unreturned items to `expired`.

---

## 8. DBOS Workflow Integration

Add two new `PipelineStage` values for the first target stage:

```python
EXTRACTING_RULES_BATCH_SUBMITTED = "extracting_rules_batch_submitted"
EXTRACTING_RULES_BATCH_COLLECTED = "extracting_rules_batch_collected"
```

Add DBOS steps:

```python
@DBOS.step(retries_allowed=False)
async def submit_extraction_batch_step(...):
    ...
```

```python
@DBOS.step(retries_allowed=False)
async def poll_extraction_batch_step(...):
    ...
```

```python
@DBOS.step(retries_allowed=False)
async def collect_extraction_batch_step(...):
    ...
```

Workflow shape:

```text
compile_prompts_step
  -> submit_extraction_batch_step
  -> DBOS.sleep(batch_poll_interval_seconds)
  -> poll_extraction_batch_step
       loop until not processing
  -> collect_extraction_batch_step
  -> continue pipeline
```

The submit step must write `BatchJob` before marking `EXTRACTING_RULES_BATCH_SUBMITTED`.

On resume:

1. Check whether the batch-submitted marker exists.
2. If yes, load the existing `BatchJob`.
3. Reuse `provider_batch_id`.
4. Do not re-submit.

---

## 9. Worker Behavior

### 9.1 Eligible First Stage

OpenAI-first rollout should initially batch only:

```text
EXTRACTING_RULES
```

This is the highest-value and cleanest stage because it is one model call per extraction case and already has per-case idempotency.

### 9.2 Successful Items

For each succeeded item:

1. Parse structured output.
2. Write the normal extraction result.
3. Write `ModelCallRecord` with `is_batch=True`.
4. Mark the case-level `StageMarker` complete.

### 9.3 Errored or Expired Items

For each errored or expired item:

1. Log provider error.
2. Do not write the normal artifact.
3. Do not mark the per-case stage complete.
4. Increment model failure counters.
5. Allow retry via sequential fallback on pipeline resume.

### 9.4 Full Batch Failure

If the whole batch fails or expires:

1. Mark `BatchJob.status` as `failed` or `expired`.
2. Clear `EXTRACTING_RULES_BATCH_SUBMITTED`.
3. Fall back to sequential `complete_structured` for all uncompleted items.

Partial failure must not fail the pipeline.

---

## 10. Configuration

### 10.1 Provider Profile

Add:

```python
class ProviderProfile(BaseModel):
    batch_enabled: bool = False
```

This is the provider capability gate.

### 10.2 Phase Teacher Config

Add:

```python
class PhaseTeacherConfig(BaseModel):
    batch_enabled: bool = False
    batch_min_items: int = 10
```

This is the stage intent gate.

### 10.3 App Settings

Add:

```python
class AppSettings(BaseModel):
    batch_poll_interval_seconds: int = 60
```

Default remains conservative.

### 10.4 Future Evaluation Toggle

For later eval batching:

```python
class DistillationRequest(BaseModel):
    batch_eval: bool = False
```

Do not add eval batching in the OpenAI P0 unless extraction batching is already stable.

### 10.5 UI Wiring Requirement

The UI must expose batching controls and wire them through to the request/configuration layer. At minimum, the UI should allow authorized users to:

1. Enable or disable batching for eligible teacher phases.
2. Configure or surface the `batch_min_items` threshold where appropriate.
3. Enable or disable batch evaluation once eval batching is implemented.
4. Clearly show when a run will use batch mode versus sequential mode.
5. Prevent users from enabling batch mode when the selected provider profile does not have `batch_enabled=True`.

The UI should not bypass backend gating. Backend capability checks remain authoritative through `ProviderProfile.batch_enabled`, `PhaseTeacherConfig.batch_enabled`, `batch_min_items`, and `isinstance(chat_client, BatchChatModelClient)`.

---

## 11. Usage and Pricing

Update `schemas/usage.py`:

```python
class ModelCallRecord(BaseModel):
    is_batch: bool = False
    batch_id: str | None = None
```

Update `usage/pricing.py`:

```python
def calculate_batch(...):
    base = self.calculate(...)
    discount = pricing_config.batch_discount or Decimal("0")
    return base * (Decimal("1") - discount)
```

The implementation should read the discount from `config/model_pricing.yaml`, not hard-code it.

---

## 12. Implementation Sequence

### Phase 1 — Shared Infrastructure

1. Add `schemas/batch.py`.
2. Add `BatchChatModelClient` to `providers/contracts.py`.
3. Add `providers/batch_schema_registry.py`.
4. Add `BatchJob` model.
5. Add Alembic migration.
6. Add `batch_enabled` to `ProviderProfile`.
7. Add `batch_enabled` and `batch_min_items` to `PhaseTeacherConfig`.
8. Add `batch_poll_interval_seconds` to `AppSettings`.
9. Add `is_batch` and `batch_id` to `ModelCallRecord`.
10. Add `calculate_batch()` to `PricingService`.
11. Wire UI controls for batch enablement into the request/configuration layer.
12. Unit test schema serialization, registry lookup, pricing, config defaults, and UI-to-request mapping.

### Phase 2 — OpenAI Batch Provider

1. Add `providers/batch/openai_batch.py`.
2. Implement JSONL serialization for `/v1/responses`.
3. Implement file upload with purpose `batch`.
4. Implement batch creation with `completion_window="24h"`.
5. Implement polling.
6. Implement output and error file collection.
7. Implement structured output parsing through schema registry.
8. Add mock OpenAI batch tests.
9. Add contract tests against `BatchChatModelClient`.

### Phase 3 — OpenAI Extraction Integration

1. Add extraction batch stage markers.
2. Add DBOS submit, poll, and collect steps.
3. Add non-DBOS local fallback polling with `asyncio.sleep`.
4. Add worker gate using provider config, phase config, client type, and item threshold.
5. Add full-batch-failure fallback to sequential.
6. Add partial-failure handling.
7. Add integration tests with a mock batch client.
8. Add crash/resume test proving no duplicate batch submission.

### Phase 4 — OpenAI Eval Batching

After extraction batching is stable:

1. Add `DistillationRequest.batch_eval`.
2. Add batch support for `EVALUATING_BASELINE`.
3. Add batch support for `EVALUATING_DISTILLED`.
4. Prefer one batch per strategy variant.
5. Add stage pairs for each eval stage.
6. Add integration tests for multi-strategy result mapping.

### Phase 5 — Anthropic Batch Provider

1. Add `providers/batch/anthropic_batch.py`.
2. Implement `AnthropicBatchChatClient`.
3. Reuse `BatchItem`, `BatchPollStatus`, `BatchResult`, and `BatchItemResult`.
4. Reuse worker batch path.
5. Add provider-specific status mapping.
6. Add mock Anthropic batch tests.
7. Enable Anthropic for `EXTRACTING_RULES`.
8. Extend Anthropic to synthesis and conflict review if volume justifies it.

### Phase 6 — Anthropic Synthesis and Conflict Review

1. Add batch support for `SYNTHESIZING_RULES`.
2. Add batch support for `REVIEWING_RULE_CONFLICTS`.
3. Use `batch_min_items` to avoid batching small cluster sets.
4. Add stage pairs for both stages.

### Phase 7 — Bedrock Decision

Defer Bedrock until explicitly required.

---

## 13. Stage Candidacy

### P0

```text
EXTRACTING_RULES
```

Reason: highest volume, clean per-case idempotency, best first integration point.

### P1

```text
EVALUATING_BASELINE
EVALUATING_DISTILLED
```

Reason: high volume and strong cost-saving potential.

### P2

```text
SYNTHESIZING_RULES
REVIEWING_RULE_CONFLICTS
```

Reason: valid but smaller volume. Batch only when item count clears threshold.

### Not Eligible

```text
REFINING_RULES
EMBEDDING_RULES
CLUSTERING_RULES
PRUNING_RULES
COMPILING_PROMPTS
VALIDATING_PROJECT
ANALYZING_FAILURES
ABLASTING_RULES
OPTIMIZING_PRUNING
```

Note: verify the actual enum name for `ABLASTING_RULES` / `ABLATING_RULES` in the codebase and use the existing spelling.

---

## 14. Acceptance Criteria

OpenAI P0 is complete when:

1. `OpenAIChatClient` or `OpenAIBatchChatClient` satisfies `BatchChatModelClient`.
2. Batch mode can be enabled through provider and phase config.
3. `EXTRACTING_RULES` submits one OpenAI batch for pending extraction items.
4. Batch submission is idempotent across DBOS crash/resume.
5. Polling uses durable `DBOS.sleep`.
6. Collection maps results by `custom_id`.
7. Successful items write the same artifacts as sequential execution.
8. Failed items do not mark case-level stage completion.
9. Full batch failure falls back to sequential execution.
10. `ModelCallRecord` records `is_batch=True` and `batch_id`.
11. Pricing applies `batch_discount` from YAML.
12. UI controls exist for eligible batching options and correctly flow into backend request/configuration fields.
13. Tests cover success, partial failure, full failure, expired batch, crash/resume, below-threshold fallback, and UI-to-request mapping.

---

## 15. Open Decisions

| ID | Decision | Recommendation | Resolve Before |
|---|---|---|---|
| A | Should `BatchChatModelClient(ChatModelClient)` be the shared abstraction? | Yes | Phase 1 |
| B | OpenAI batch endpoint: `/v1/responses` or `/v1/chat/completions`? | Prefer `/v1/responses`; allow fallback if compatibility requires chat completions | Phase 2 |
| C | Keep OpenAI per-call on `pydantic-ai` while batch uses direct SDK? | Yes | Phase 2 |
| D | Polling interval strategy | Config-driven, default `60s` | Phase 1 |
| E | Batch default | Explicit opt-in everywhere | Phase 1 |
| F | Eval batching granularity | One batch per strategy variant | Phase 4 |
| G | Bedrock | Defer unless a deployment requires it | Phase 7 |

---

## 16. Summary

The main change from the original plan is sequencing: **OpenAI becomes the first provider implementation**, not Anthropic. The internal abstraction should remain provider-neutral so Anthropic can be added with minimal worker churn. OpenAI’s file-based Batch API requires a little more infrastructure up front — file IDs, output file parsing, error file parsing — but that makes the shared `BatchJob` model and collection contract stronger before adding Anthropic.