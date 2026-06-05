Batch API Implementation Plan for RuleKiln

  Codebase Baseline

  Provider contract (providers/contracts.py): One abstract method — ChatModelClient.complete_structured(system_prompt, user_prompt, output_schema, config) -> ChatCompletionResult. Entirely synchronous/per-request. No batching surface
  exists today.

  Stage chain (distillation_worker.py): 19 PipelineStage enum values. The stages that call complete_structured are: EXTRACTING_RULES, SYNTHESIZING_RULES, REVIEWING_RULE_CONFLICTS, EVALUATING_BASELINE, EVALUATING_DISTILLED, and
  REFINING_RULES. The rest are CPU-only or use the embedding client.

  DBOS (workers/dbos_workflow.py): Six @DBOS.step-decorated functions. Steps are idempotent and replayed from stored outcomes on crash. At the sub-step level the worker uses StageMarker(job_id, stage, strategy, artifact_type) in Postgres
  for per-case/per-cluster idempotency. DBOS.sleep() is durable at the workflow level.

  Pricing (config/model_pricing.yaml, usage/pricing.py): batch_discount is already present in the YAML for OpenAI models but the PricingService.calculate() method does not yet read it.

  ---
  1. Where Batch Submission Fits in the ChatModelClient Contract
  
  Decision Point A — Interface topology

  Three options exist:

  ┌──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Option                    │                                       Description                                       │                                           Verdict                                           │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Add methods to ChatModelClient               │ submit_batch / collect_batch as abstract methods with a default raise                   │ Contaminates all non-batch providers (Fake, Bedrock fallback) with stubs                    │
  │                                              │ NotImplementedError                                                                     │                                                                                             │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ New ABC                                      │ Batch providers subclass both; callers isinstance-check                                 │ Clean; batching is opt-in; existing code is untouched                                       │
  │ BatchChatModelClient(ChatModelClient)        │                                                                                         │                                                                                             │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Separate BatchExecutor wrapper               │ Orchestrates calls through an inner ChatModelClient                                     │ Loses the provider abstraction; batch Anthropic needs direct SDK access, not the per-call   │
  │                                              │                                                                                         │ path                                                                                        │
  └──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────┘

  Recommendation: Add a new abstract class BatchChatModelClient(ChatModelClient) in providers/contracts.py. It extends the existing ABC and declares three methods:

  submit_batch(items: list[BatchItem], config: ProviderConfig) -> str          # → provider batch_id
  poll_batch(batch_id: str, config: ProviderConfig) -> BatchPollStatus
  collect_batch(batch_id: str, config: ProviderConfig) -> BatchResult

  Providers that support batch (AnthropicChatClient, OpenAIChatClient) subclass this. BedrockChatClient, FakeChatClient, and all others remain as ChatModelClient subclasses only. The call site in the worker checks isinstance(client, 
  BatchChatModelClient) before entering the batch path.

  complete_structured is not deprecated and remains the single interface used in batch-ineligible stages (REFINING_RULES, synthesis when batch is off).

  What BatchItem must carry

  Because the batch is submitted all at once and results are collected later (potentially after a process restart), BatchItem must be fully self-contained — the schema type cannot be passed by Python reference across a process boundary.
  The item carries:
  - custom_id: str — opaque key the worker controls (maps to case id, cluster id, rule id, etc.)
  - system_prompt: str
  - user_prompt: str
  - output_schema_json: dict[str, object] — the JSON Schema emitted by output_schema.model_json_schema()
  - output_schema_class_name: str — used at collection time to look up the class in a local registry so the response text can be re-parsed into the correct Pydantic model

  The local registry (providers/batch_schema_registry.py) maps class names to their type[BaseModel] at import time. This is the only way to reconstruct structured output after a process boundary.

  ---
  2. Polling/Waiting and DBOS Integration

  The core problem

  Anthropic and OpenAI batch jobs complete in minutes to hours. A DBOS step that does while not done: await asyncio.sleep(60) will eventually either time out the step or hold a worker thread indefinitely. Neither is acceptable.

  Recommended architecture — two new DBOS stages, durable sleep loop

  Add two new PipelineStage values per batch-eligible stage: a *_BATCH_SUBMITTED stage and a *_BATCH_COLLECTED stage. Example for extraction:

  EXTRACTING_RULES_BATCH_SUBMITTED = "extracting_rules_batch_submitted"
  EXTRACTING_RULES_BATCH_COLLECTED = "extracting_rules_batch_collected"

  The workflow becomes:

  compile_prompts_step
    └─ submit_extraction_batch_step       # submits to provider, writes BatchJobRecord to DB
       └─ [DBOS.sleep(poll_interval_s)]  # durable sleep, survives crash
          └─ poll_extraction_batch_step   # loops with sleep until BatchPollStatus.done
             └─ collect_extraction_batch_step  # downloads, parses, writes micro-rules

  DBOS.sleep() at the workflow level is the right primitive: it is checkpointed, crash-safe, and does not hold a worker thread. The poll loop itself is a tight while poll_status.processing: await DBOS.sleep(interval) inside the
  _run_rulekiln_stage_workflow. retries_allowed=False is kept on steps (same as today) since each step is already idempotent through StageMarker.

  Fallback when DBOS is absent

  The non-DBOS path (used in tests and local envs) uses plain asyncio.sleep. This is already how the worker operates for the non-batch path. Batch polling in the test harness uses a much shorter polling interval and a mock provider that
  completes immediately.

  BatchJobRecord persistence

  A new BatchJob SQLAlchemy model must be written before the step decorator returns. This means:
  - job_id, stage, strategy
  - provider_batch_id (the string returned by the provider)
  - status: Literal["submitted", "polling", "completed", "failed", "expired"]
  - submitted_at, completed_at
  - item_count, succeeded_count, errored_count
  
  The StageMarker system then records EXTRACTING_RULES_BATCH_SUBMITTED after the record is written, so a crashed-and-resumed workflow can recover the provider_batch_id rather than re-submitting the batch.

  Decision Point B — Polling interval strategy

  Provider SLAs suggest:
  - Anthropic: typically 15–60 minutes
  - OpenAI: typically 15 minutes–6 hours

  A fixed 60-second poll is wasteful for large batches. Options:
  - Fixed 60s (simple, reasonable)
  - Exponential back-off starting at 30s, capping at 300s (better for long jobs)
  - Config-driven via AppSettings.batch_poll_interval_seconds (most flexible)

  Recommendation: Config-driven with default 60s. Add batch_poll_interval_seconds: int = 60 to AppSettings.

  ---
  3. Result Mapping and Partial-Failure Contract

  Per-item status

  The collect step receives results from the provider. Each item has one of three terminal states across both APIs:

  ┌─────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
  │             Provider status             │                           RuleKiln mapping                            │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Anthropic succeeded                     │ BatchItemResult(status="succeeded", result=ChatCompletionResult(...)) │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Anthropic errored / OpenAI error status │ BatchItemResult(status="errored", error_message=...)                  │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Anthropic expired / OpenAI expired      │ BatchItemResult(status="expired", ...) — treat as errored             │
  └─────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘

  Partial-failure contract

  The worker already handles per-case failures in the sequential path: except Exception: return None, True sets malformed=True for that case, increments model_failure_count, and the pipeline continues. The batch collect loop must honour
  the same contract:

  - Succeeded items: parse the response text against output_schema (same Pydantic validation as today), write the result to DB, call mark_stage_complete(case_marker) — identical to the sequential path.
  - Errored/expired items: log the error, do not write a stage marker, increment an error counter. The item will be retried on next pipeline resume via the sequential path (since its stage marker is absent). This is the correct behavior.
  - Complete batch failure (status failed or expired at the job level): the worker clears the EXTRACTING_RULES_BATCH_SUBMITTED stage marker, falls back to sequential complete_structured for all uncompleted items.

  Do not treat a partial batch failure as a pipeline failure. The existing idempotency system absorbs it.

  Usage tracking

  ModelCallRecord and tracked_chat_call are built for per-call telemetry. Batch calls emit one ModelCallRecord per item in the collect step, with latency_ms set to the item's creation-to-collection delta. A new field is_batch: bool = 
  False on ModelCallRecord enables cost reporting to apply the discount. The PricingService gains a calculate_batch() variant that multiplies by (1 - batch_discount).

  ---
  4. Provider Sequencing Recommendation

  Deliver Anthropic first, OpenAI second, Bedrock deferred

  Anthropic (P0):
  - Teacher extraction is the dominant cost in any distillation job. Anthropic is the default teacher provider.
  - The Anthropic SDK is already used directly (not through pydantic-ai). client.messages.batches is a first-class SDK feature.
  - Structured output is handled the same way as complete_structured today: the batch request embeds the JSON schema instruction in the system prompt, and the collect step parses the text response against the schema.
  - Scope: ~300 lines in a new providers/batch/anthropic_batch.py, plus schema, DB model, worker integration.

  OpenAI (P1):
  - Second most common teacher/student. The Batch API is file-based (upload JSONL → get batch ID → poll → download output JSONL), which is more complex than Anthropic but well-documented.
  - Currently uses pydantic-ai (OpenAIModel/Agent). Batch must bypass pydantic-ai and use openai.AsyncOpenAI directly — same pattern as AnthropicChatClient. Decision Point C: confirm it is acceptable to dual-implement (pydantic-ai for
  per-call, direct SDK for batch) rather than switching the per-call path to direct SDK.
  - Scope: ~400 lines in providers/batch/openai_batch.py.

  Bedrock (defer or explicit scope decision):
  - Bedrock's equivalent is "Model Invocation Jobs" — not the Converse API. It is S3-based (input/output JSONL in S3 buckets), requires IAM permissions for SageMaker/Bedrock batch roles, and only supports a subset of foundation models.
  - This is architecturally disjoint from the Anthropic/OpenAI batch implementation. It deserves its own scoping conversation. 
  - Decision Point D: Is Bedrock the primary student provider in any deployed environment? If yes, scope it explicitly. If no, defer until demand is demonstrated.

  ---
  5. New Pydantic Schemas Needed
  
  schemas/batch.py (new)

  BatchItem
    custom_id: str
    system_prompt: str
    user_prompt: str
    output_schema_json: dict[str, object]
    output_schema_class_name: str

  BatchPollStatus
    batch_id: str
    provider: str
    processing: bool
    succeeded_count: int
    errored_count: int
    total_count: int
    estimated_completion_at: datetime | None

  BatchItemResult
    custom_id: str
    status: Literal["succeeded", "errored", "expired"]
    result: ChatCompletionResult | None  # present iff succeeded
    error_message: str | None
    usage: ModelUsage | None             # present iff succeeded

  BatchResult
    batch_id: str
    provider: str
    status: Literal["completed", "partial", "failed", "expired"]
    items: list[BatchItemResult]
    succeeded_count: int
    errored_count: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal

  Additions to existing schemas

  ModelCallRecord in schemas/usage.py:
  - Add is_batch: bool = False
  - Add batch_id: str | None = None

  PhaseTeacherConfig in schemas/classroom.py:
  - Add batch_enabled: bool = False
  - Add batch_min_items: int = 10 — below this threshold fall back to sequential (see §6)

  ProviderProfile in config/settings.py:
  - Add batch_enabled: bool = False

  AppSettings in config/settings.py:
  - Add batch_poll_interval_seconds: int = 60

  New DB model

  BatchJob SQLAlchemy model (see §2 for field list). This also needs the Alembic migration.

  ---
  6. Toggling Batch Mode
  
  Two-level authority: provider capability + stage intent

  Level 1 — provider capability gate:
  ProviderProfile.batch_enabled: bool = False. This must be True for batch to activate regardless of any other setting. It signals "this profile's provider + credentials have batch access." A provider profile with batch_enabled=False
  always uses the sequential path. This prevents surprise charges on accounts without batch API access.

  Level 2 — stage-level opt-in:
  PhaseTeacherConfig.batch_enabled: bool = False. The three teacher phases (instruction_extraction, cluster_consolidation, conflict_resolution) each get independent control. For student evaluation, DistillationRequest gains a batch_eval: 
  bool = False flag at the top level (simpler than per-strategy control).

  Resolution logic (in the worker, before each stage):
  use_batch = (
      provider_profile.batch_enabled
      and phase_config.batch_enabled
      and isinstance(chat_client, BatchChatModelClient)
      and len(pending_items) >= phase_config.batch_min_items
  )
  
  Decision Point E: Should batch be enabled by default for any stage once the provider supports it, or require explicit opt-in? Recommendation: explicit opt-in (batch_enabled=False default everywhere) for the first two releases.
  Auto-enable can be revisited once cost/latency behavior is validated in production.

  Minimum-items threshold: batch_min_items defaults to 10. Below this the async-wait overhead of the batch API is not justified. For synthesis (typically 5–50 clusters), a small job may never hit the threshold and silently uses sequential
  — which is correct.

  ---
  7. Pipeline Stage Batch Candidacy

  Eligible

  ┌──────────────────────────┬─────────┬──────────────────────────────────────────────────┬──────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Stage           │  Role   │                   Call pattern                   │ Priority │                                                               Notes                                                                │
  ├──────────────────────────┼─────────┼──────────────────────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EXTRACTING_RULES         │ teacher │ One call per extraction case. Can be 50–5000     │ P0       │ Highest value. Per-case StageMarker already exists; collect loop maps naturally.                                                   │
  │                          │         │ calls.                                           │          │                                                                                                                                    │
  ├──────────────────────────┼─────────┼──────────────────────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EVALUATING_BASELINE      │ student │ One call per eval case × baseline strategy       │ P1       │ High volume; student model; 50% discount applies. Multiple baseline variants could be separate batches or one merged batch.        │
  │                          │         │ variants.                                        │          │ Decision Point F.                                                                                                                  │
  ├──────────────────────────┼─────────┼──────────────────────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EVALUATING_DISTILLED     │ student │ One call per eval case × (dbscan + hdbscan).     │ P1       │ Same as baseline eval; already a separate DBOS step per strategy — maps well to one batch per strategy.                            │
  ├──────────────────────────┼─────────┼──────────────────────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ SYNTHESIZING_RULES       │ teacher │ One call per cluster. Typically 5–50.            │ P2       │ Low volume; the batch_min_items gate will keep small jobs sequential. Value is on larger jobs with many clusters.                  │
  ├──────────────────────────┼─────────┼──────────────────────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ REVIEWING_RULE_CONFLICTS │ teacher │ One call per synthesized rule. Typically 5–50.   │ P2       │ Same volume characteristics as synthesis. Rules are independent; per-rule StageMarker already exists.                              │
  └──────────────────────────┴─────────┴──────────────────────────────────────────────────┴──────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Not eligible

  ┌─────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                      Stage                      │                                                                                       Reason                                                                                        │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ REFINING_RULES                                  │ Explicitly excluded. This is the paper's Phase 3 closed-loop: each refinement iteration re-evaluates the student on the revised rules. Iteration N's output is iteration N+1's      │
  │                                                 │ input. Strictly sequential. No batching possible.                                                                                                                                   │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ EMBEDDING_RULES                                 │ Uses EmbeddingClient.embed_texts, a different interface that already accepts multiple texts in one call. Not a ChatModelClient call.                                                │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CLUSTERING_RULES                                │ Pure CPU (DBSCAN/HDBSCAN scikit-learn). No model calls.                                                                                                                             │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ PRUNING_RULES                                   │ Pure CPU, scoring and selection logic. No model calls.                                                                                                                              │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ COMPILING_PROMPTS                               │ Template assembly. No model calls.                                                                                                                                                  │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ VALIDATING_PROJECT                              │ Schema validation. No model calls.                                                                                                                                                  │
  ├─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ANALYZING_FAILURES / ABLATING_RULES /           │ If these reach LLM calls in future, evaluate then. Today they are metrics and DB aggregation.                                                                                       │
  │ OPTIMIZING_PRUNING                              │                                                                                                                                                                                     │
  └─────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Recommended Implementation Sequence
  
  Phase 1 — Infrastructure (no provider yet)

  1. schemas/batch.py — all schemas above
  2. db/models.py + Alembic migration — BatchJob model
  3. providers/contracts.py — BatchChatModelClient ABC
  4. providers/batch_schema_registry.py — class-name → type[BaseModel] registry
  5. config/settings.py — batch_enabled on ProviderProfile, batch_poll_interval_seconds on AppSettings
  6. schemas/classroom.py — batch_enabled + batch_min_items on PhaseTeacherConfig
  7. usage/pricing.py — calculate_batch() using batch_discount from YAML
  8. Tests for all the above (no actual API calls).

  Phase 2 — Anthropic batch (EXTRACTING_RULES)

  1. providers/batch/anthropic_batch.py — AnthropicBatchChatClient(BatchChatModelClient), three methods
  2. workers/distillation_worker.py — batch path in EXTRACTING_RULES stage, using batch_min_items gate and fallback
  3. Two new PipelineStage values: EXTRACTING_RULES_BATCH_SUBMITTED, EXTRACTING_RULES_BATCH_COLLECTED
  4. workers/dbos_workflow.py — new submit/poll/collect DBOS steps for extraction
  5. Integration tests with a mock Anthropic batch client.

  Phase 3 — Extend to eval stages (EVALUATING_BASELINE + EVALUATING_DISTILLED)

  1. Worker integration for student eval batch (both strategies)
  2. DistillationRequest.batch_eval: bool = False toggle
  3. New PipelineStage pairs for each eval stage.

  Phase 4 — Anthropic batch for synthesis + conflict review

  1. Worker integration for SYNTHESIZING_RULES and REVIEWING_RULE_CONFLICTS
  2. These reuse the same AnthropicBatchChatClient; only the worker stage logic changes.

  Phase 5 — OpenAI batch

  1. providers/batch/openai_batch.py — direct openai.AsyncOpenAI, file upload path
  2. Per-call path remains pydantic-ai (no change to existing OpenAIChatClient)
  3. Decision Point C resolved before this phase begins.

  Phase 6 — Bedrock (conditional on Decision Point D)

  ---
  Open Decision Points Summary

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────┐
  │  #  │                                        Question                                        │  Must resolve before   │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ A   │ Interface topology: confirm BatchChatModelClient(ChatModelClient) as a sub-ABC         │ Phase 1                │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ B   │ Polling interval strategy: fixed 60s vs. exponential back-off vs. config-driven        │ Phase 1                │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ C   │ OpenAI: acceptable to bypass pydantic-ai for batch while keeping it for per-call?      │ Phase 5                │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ D   │ Bedrock: is batch inference required for any current deployment?                       │ Phase 6 scope decision │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ E   │ Batch default: explicit opt-in everywhere, or auto-enable per provider?                │ Phase 2                │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
  │ F   │ Baseline eval variants: one merged batch per stage, or one batch per strategy variant? │ Phase 3                │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────┘
