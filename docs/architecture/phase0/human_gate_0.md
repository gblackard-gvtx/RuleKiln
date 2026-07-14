# Human Gate 0 — Product Decisions

**Status:** Pending

No package implementation should begin until these decisions are approved.

## Decision 1 — First reference task

**Recommendation:** ContractNLI classification for the first engineering reference.

Reason:

- the dataset and pipeline already exist;
- it exercises long-input classification, minority labels, malformed handling, and rule refinement;
- it provides continuity with current work.

Limitation:

- it is a research benchmark, not the eventual flagship product demonstration.

**Decision:**  
- [x] Approve ContractNLI  
- [ ] Choose another task: ______________________

## Decision 2 — Initial student model

**Recommendation:** Use the existing local Gemma student profile for the first golden-path run.

Record exact values before implementation:

- model ID/checkpoint:
- quantization:
- context size:
- chat template:
- llama.cpp version:
- decoding defaults:

**Decision:**  
- [x] Approved  
- [ ] Replacement model: ______________________

## Decision 3 — Teacher/synthesis arrangement

**Decision:** Approved with separate logical model roles.

RuleKiln will preserve distinct roles for extraction, synthesis, and conflict resolution, even when they use the same provider profile.

Initial configuration:

| Role | Provider profile | Model ID | Purpose |
|---|---|---|---|
| Extraction teacher | `openai_teacher` | `gpt-5.4-mini` | High-volume micro-rule extraction from training cases |
| Synthesis teacher | `openai_teacher` | `gpt-5.2` | Consolidate clustered micro-rules into generalized semantic rules |
| Refinement teacher | `openai_teacher` | `gpt-5.2` | Conflict resolution and validation-driven rule refinement |
| Judge | Not required for Phase 0 | — | Add only when an evaluation or promotion policy explicitly requires a separate judge |

Fallback behavior:

- Each role uses the shared `openai_teacher` profile.
- The per-phase model ID overrides the profile's default model.
- If a phase-specific model override is absent, the phase falls back to the model configured on `openai_teacher`.
- Model IDs and effective runtime settings must be persisted in job and package provenance.

Rationale:

- Extraction is O(number of training cases), so it uses the lower-cost `gpt-5.4-mini`.
- Synthesis is O(number of clusters), so a stronger model can be used at substantially lower call volume.
- Conflict resolution is O(number of refinement iterations) and requires the strongest reasoning of the three roles.
- Logical role separation is preserved so models can be changed independently later.

**Decision:**  
- [x] Approved  
- [ ] Modified arrangement

Approved arrangement:

- Extraction teacher: `openai_teacher` / `gpt-5.4-mini`
- Synthesis teacher: `openai_teacher` / `gpt-5.2`
- Refinement teacher: `openai_teacher` / `gpt-5.2`
- Judge: disabled for the initial classification golden path

## Decision 4 — Initial runtime

**Decision:** Approved.

RuleKiln's initial student runtime will be a local `llama.cpp` server exposed through an OpenAI-compatible API.

Initial runtime configuration:

| Field | Value |
|---|---|
| Runtime | `llama.cpp` |
| API protocol | OpenAI-compatible Chat Completions API |
| Provider type | `openai_compatible` |
| Deployment | Local model server |
| Student model | `gemma-4-E4B-it-Q4_K_M.gguf` |
| Quantization | `Q4_K_M` |
| Chat support | Enabled |
| Embedding support | Disabled for the student profile |
| Concurrency | Provider-profile controlled |
| Network endpoint | Configured by the RuleKiln provider profile; not embedded into portable packages |

The `.rkprompt` package will target a generic OpenAI-compatible runtime, with `llama.cpp` as the first validated implementation.

Runtime defaults and compatibility information must persist:

- runtime adapter;
- requested model ID;
- effective model ID;
- model filename or checkpoint;
- quantization;
- context-window size;
- chat template;
- decoding parameters;
- stop sequences;
- maximum output tokens;
- `llama.cpp` version;
- output-schema support;
- runtime configuration hash.

The package must not assume that every OpenAI-compatible endpoint behaves identically. A model/runtime combination is considered supported only after the package regression suite passes against that exact combination.

**Decision:**

- [x] Approved
- [ ] Replacement runtime

Approved runtime:

- Runtime: `llama.cpp`
- Interface: OpenAI-compatible API
- Initial student: `gemma-4-E4B-it-Q4_K_M.gguf`
- Deployment target: local/offline-capable inference

The student model and the runtime are separate decisions:

- **Student model:** the model weights/checkpoint being evaluated.
- **Runtime:** the inference engine and API contract used to execute that model.

For Phase 0, the selected combination is:

`gemma-4-E4B-it-Q4_K_M.gguf` running under `llama.cpp` through an OpenAI-compatible endpoint.

## Decision 5 — Output contract

**Recommendation:**

```json
{"label": "<allowed-label>"}
```

Constraints:

- JSON object only;
- exactly one required property: `label`;
- no additional properties;
- label must be in the declared vocabulary.

**Decision:**  
- [x] Approved  
- [ ] Modified output contract: ______________________

## Decision 6 — First milestone scope

**Decision:** Approved — the first RuleKiln product milestone is limited to multiclass classification with strict structured output.

The supported task contract is:

- one input case;
- one prediction;
- one label selected from a package-declared vocabulary;
- strict JSON output;
- no additional output properties.

Canonical output:

```json
{"label": "<allowed-label>"}


## Supporting Decision — Initial embedding runtime

- Provider profile: `local_mxbai_embed`
- Model: `mxbai-embed-large-v1.Q6_K.gguf`
- Runtime: `llama.cpp` or current embedding server
- Dimension: `1024`
- Normalization: `<effective behavior>`
- Distance metric: `cosine`
- Quantization: `Q6_K`