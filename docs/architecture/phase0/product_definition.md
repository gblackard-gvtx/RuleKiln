# RuleKiln Product Definition

**Status:** Proposed  
**Phase:** 0 — Product contract and golden-path design  
**Decision owner:** Maintainer  
**Last updated:** 2026-07-14

## Product statement

RuleKiln is a prompt distillation compiler.

It converts labeled task examples into tested, versioned, portable prompt packages that can be executed by smaller, local, or specialized language models without requiring model fine-tuning.

Prompt-Level Distillation is an architectural foundation for RuleKiln, not the product identity. RuleKiln may use extraction, consolidation, synthesis, refinement, evaluation, and packaging techniques that differ from a specific paper when those changes improve reliability, portability, or user value.

## Primary user

The initial user is a developer who has:

- a bounded classification task;
- labeled examples;
- a target language model;
- a need to improve that model without changing its weights;
- a requirement for held-out evaluation, reproducibility, and portable deployment.

## Initial product promise

Given a split-safe labeled dataset and configured model roles, RuleKiln will:

1. establish a student-model baseline;
2. extract candidate task rules from training cases;
3. consolidate and synthesize those rules;
4. compile deterministic prompt candidates;
5. evaluate candidates on held-out validation cases;
6. report fixed, broken, malformed, and per-label outcomes;
7. select a candidate without implying that it is production-ready;
8. export a self-contained `.rkprompt` package;
9. preserve the provenance needed to reproduce and audit the package.

## Initial supported task mode

The first product milestone supports:

> Multiclass classification with strict structured output.

The canonical response shape is:

```json
{"label": "<allowed-label>"}
```

The label must be exactly one value from the package-declared label vocabulary.

The initial milestone does not support arbitrary free-form generation, multi-turn agent plans, or unrestricted tool use.

## Product boundary

RuleKiln is not initially:

- a general model fine-tuning platform;
- a universal automatic-prompt-optimization system;
- a clustering product;
- a benchmark dashboard;
- a general autonomous-agent framework;
- an exact reproduction of any single research implementation;
- a guarantee that every dataset or student model will improve.

## Primary product artifact

The primary output is a versioned RuleKiln Prompt Package.

A package contains the compiled prompt, semantic rules, output contract, runtime defaults, evaluation evidence, regression cases, compatibility declarations, and provenance needed to use the result independently of the RuleKiln orchestration stack.

Canonical package layout:

```text
task-name.rkprompt
└── ZIP archive
    ├── manifest.json
    ├── system_prompt.txt
    ├── semantic_rules.json
    ├── output_schema.json
    ├── runtime_defaults.json
    ├── evaluation_summary.json
    ├── compatibility.json
    ├── provenance.json
    ├── regression_cases.jsonl
    └── README.md
```

During development, the same structure may exist as an unpacked directory. The distributable `.rkprompt` form is a deterministic ZIP archive.

## Canonical terms

### Micro-rule

A natural-language instruction extracted from one labeled training example.

A micro-rule:

- is produced during supervised rule extraction;
- is tied to one or more source training-case IDs;
- may be one of several rules extracted from the same example;
- is not yet assumed to be globally correct or deployment-ready.

### Synthesized rule

A consolidated instruction produced from a group of related micro-rules.

A synthesized rule:

- references its source micro-rule IDs;
- records its consolidation strategy and provenance;
- is evaluated for downstream fixes, regressions, conflicts, and token cost;
- may be accepted, revised, pruned, or rejected.

### Semantic rule set

The structured, model-independent collection of accepted task rules before target-specific rendering.

A semantic rule set contains:

- stable rule IDs;
- rule text and scope;
- priorities and exceptions;
- conflicts and dependencies;
- provenance;
- support and evaluation statistics;
- active or pruned status.

### Compiled prompt

A deterministic, target-aware rendering of a semantic rule set plus task instructions and an output contract.

A compiled prompt records:

- included and excluded rule IDs;
- rule ordering;
- target adapter;
- token count;
- output-contract version;
- compilation configuration;
- prompt hash.

A compiled prompt is an artifact inside a package, not the complete package.

### Prompt package

A versioned, portable archive containing a compiled prompt and all information required to inspect, execute, test, and audit it.

A prompt package is the primary product output of RuleKiln.

### Selected strategy

The highest-ranked strategy that passed the configured evaluation eligibility rules for a particular run.

Selection means only:

- the strategy won that run under the configured primary metric and gates.

Selection does not mean:

- the result is reproducible;
- the package is validated;
- the package is released;
- the package is portable to untested models.

### Validated package

A prompt package that has passed the configured validation policy for named target models and runtimes.

At minimum, validation requires:

- complete required provenance;
- successful package integrity checks;
- repeatable evaluation under pinned settings;
- strict metric and malformed-output thresholds;
- regression-suite success;
- explicit target-model compatibility records.

A validated package is still not necessarily released.

### Released package

A validated package explicitly approved by a human maintainer for distribution or operational use.

Release is never automatic.

A released package:

- has an immutable package version;
- identifies supported models and known limitations;
- includes a release decision record;
- cannot be silently modified in place.

## Package lifecycle

The initial lifecycle is intentionally small:

- `experimental` — created from a run; evidence may be incomplete.
- `candidate` — selected from a valid evaluation and ready for further testing.
- `validated` — repeatability and target-runtime validation have passed.
- `released` — explicitly approved by a human maintainer.

## Product invariants

1. **Per-case predictions are authoritative.** Aggregate metrics are derived artifacts.
2. **Malformed output never becomes a fourth class.**
3. **All assigned cases count in strict metrics.**
4. **Train, validation, and test lineage is explicit and hashed.**
5. **A selected strategy is not automatically validated or released.**
6. **Model compatibility is explicit, not implied.**
7. **Historical artifacts are not rewritten to fabricate missing provenance.**
8. **A package can be inspected and regression-tested without the RuleKiln worker stack.**
9. **Released packages are immutable.**
10. **Human approval is required for release.**

## Proposed Human Gate 0 decisions

These are recommendations, not yet approved decisions.

| Decision | Proposed value | Rationale |
|---|---|---|
| First reference task | ContractNLI classification | Existing pipeline and artifacts make it the fastest path to validate the architecture. It remains a research benchmark, not the final product showcase. |
| Initial student model | Existing local Gemma student profile | Minimizes new infrastructure and exercises the local-model value proposition. |
| Teacher arrangement | Separate extraction and synthesis/refinement roles; the same provider may fill multiple roles | Preserves architectural clarity without requiring multiple vendors. |
| Initial runtime | llama.cpp through an OpenAI-compatible endpoint | Already aligned with the project and suitable for local deployment. |
| Output contract | Strict JSON object with one `label` field | Easy to validate, portable, and appropriate for classification. |
| First milestone | Classification-only | Prevents agent/tool scope from destabilizing the package and evaluation contracts. |

## Deferred capabilities

The following are explicitly outside the first milestone:

- unrestricted free-form generation;
- multi-step autonomous agents;
- automatic tool execution;
- mobile SDKs;
- automatic production promotion;
- fine-tuning and LoRA;
- TextGrad-style APO;
- broad clustering-algorithm expansion;
- automatic test-set iteration;
- universal cross-model portability claims;
- package mutation after validation or release.

## First milestone success criteria

The first product milestone is successful when one command can:

1. consume a split-safe classification task;
2. run a baseline;
3. construct a candidate semantic rule set and prompt;
4. evaluate it on validation with strict metrics;
5. export a `.rkprompt` package;
6. run the package through a standalone local-runtime regression command;
7. reproduce the package's declared validation evidence from authoritative records.
