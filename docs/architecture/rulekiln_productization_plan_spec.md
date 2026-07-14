# RuleKiln Productization Plan Spec

**Status:** Proposed  
**North star:** Turn RuleKiln into a compiler that converts labeled examples into tested, versioned, portable prompt packages for smaller models, local runtimes, and bounded AI subagents.

---

## 1. Product definition

RuleKiln should be presented as:

> A prompt distillation compiler that converts task examples into validated prompt packages that can be deployed to smaller, local, or specialized language models.

Prompt-Level Distillation provides the structural foundation: supervised rule extraction, rule consolidation, closed-loop refinement, and prompt-based inference. Exact reproduction of one paper is a research track, not the product definition.

### Core user journey

1. Supply labeled examples with explicit train, validation, and test boundaries.
2. Run a repeatable distillation pipeline.
3. Compare distilled candidates against strong baselines.
4. Inspect which rules fixed and broke cases.
5. Export a self-contained prompt package.
6. Run the package without the RuleKiln orchestration stack.
7. Validate it on a local, quantized, or specialized target model.

### Initial scope

Prioritize:

- classification;
- single-response structured output;
- local and OpenAI-compatible runtimes;
- deterministic package export and regression testing.

Defer unrestricted multi-step agents and automatic production promotion.

---

## 2. Architectural principles

1. **The deployable package is the product.** The final prompt must be usable outside the pipeline.
2. **Per-case records are authoritative.** Metrics, gates, rankings, and promotion decisions derive from hashed per-case prediction records.
3. **Split safety is mandatory.** One physical file is acceptable only when each stage verifies its permitted logical split.
4. **Clustering is a strategy.** DBSCAN and HDBSCAN must not define the architecture.
5. **Rule quality outranks cluster quality.** Downstream fixes, regressions, token cost, conflicts, and portability matter more than cluster count alone.
6. **Compilation is deterministic and target-aware.** Semantic rules, output contracts, token budgets, and model formatting are separate concerns.
7. **Selection is not promotion.** A selected strategy is not automatically reusable or releasable.
8. **Human review is a required stage.** Costly runs, qualitative inspection, benchmark claims, and release decisions remain human-controlled.

---

## 3. Conceptual domains

```text
tasks/
  schemas, datasets, split policy

discovery/
  student execution, rule extraction, rule quality

consolidation/
  strategy interface, embeddings, clustering, synthesis

refinement/
  conflict analysis, teacher revision, convergence

compilation/
  semantic rules, target adapters, token budgets, output contracts

evaluation/
  per-case records, strict metrics, paired comparison, regression analysis

packaging/
  prompt packages, manifests, compatibility, lifecycle status

runtimes/
  OpenAI-compatible, llama.cpp, reference runner
```

This is initially a conceptual architecture. Do not begin with a large physical repository reorganization.

---

# Phase 0 — Product contract and golden-path design

## Objective

Define what RuleKiln produces and freeze a narrow first product path before adding more algorithms or UI.

## Engineering work

- Write the formal product definition.
- Define canonical terms:
  - micro-rule;
  - synthesized rule;
  - semantic rule set;
  - compiled prompt;
  - prompt package;
  - selected strategy;
  - validated package;
  - released package.
- Define the authoritative pipeline from examples to package.
- Draft the first versioned `.rkprompt` package schema.
- Define schema-versioning and legacy compatibility rules.
- Select one initial task mode: classification with strict structured output.
- Record deferred capabilities.

## Deliverables

- `docs/architecture/product_definition.md`
- `docs/architecture/golden_path.md`
- ADR for prompt packages
- ADR for evaluation authority
- Initial `.rkprompt` schema

## Human Gate 0 — Product decisions

The maintainer selects:

- the first reference task;
- the initial student model;
- the initial teacher/synthesis model arrangement;
- the initial runtime, preferably llama.cpp through an OpenAI-compatible endpoint;
- the output contract, preferably strict JSON;
- whether the first milestone is classification-only.

## Exit criteria

- Product definition is approved.
- One task, model, and runtime are selected.
- Deferred work is explicit.
- No implementation proceeds based on an undefined package format.

---

# Phase 1 — Dataset integrity and authoritative evaluation

## Objective

Ensure every result is trustworthy before further optimization.

## Engineering work

### Split and lineage enforcement

- Support separate and combined files with explicit stage routing.
- Fail closed on missing or unknown splits.
- Persist source split, case count, ordered case-ID hash, and case-set hash on every stage artifact.
- Enforce train-only lineage for discovery, extraction, embeddings, clustering, synthesis, and initial compilation.
- Enforce validation-only lineage for candidate evaluation, paired comparison, gates, and selection.
- Keep test inaccessible until explicit final evaluation.
- Persist `metadata/split_leakage_validation.json`.

### Metric authority

Implement and expose:

- strict macro-F1;
- strict accuracy;
- valid-output macro-F1;
- valid-output accuracy;
- malformed count and rate;
- per-label precision, recall, F1, and support;
- fixed, broken, unchanged-correct, and unchanged-wrong.

Rules:

- malformed predictions count as incorrect in strict metrics;
- malformed is not a fourth label;
- paired comparisons join by stable case ID;
- all strategies use the same assigned-case denominator;
- stale summaries cannot influence promotion;
- persisted summaries are validated against source hashes.

### Provenance

Persist hashes for:

- dataset revision and split;
- ordered case IDs;
- task configuration;
- provider/model configuration;
- parser configuration;
- quality gates;
- compiled prompt;
- predictions;
- evaluation result.

## Deliverables

- authoritative evaluation pipeline;
- strict/valid metric documentation;
- split-leakage report;
- stale-artifact validator;
- generation manifest;
- updated evaluation report.

## Human Gate 1 — Real split-safety smoke run

The maintainer runs one small combined-file job and reviews:

- leakage findings;
- stage case counts;
- case-set hashes;
- malformed outputs;
- paired totals;
- stale-artifact warnings;
- promotion blockers.

The human confirms that validation never influenced initial prompt construction and that test was not accessed.

## Exit criteria

- Automated suite passes.
- A real smoke job has no blocking split contamination.
- UI metrics trace to authoritative per-case records.
- Denominators and malformed handling are unambiguous.

---

# Phase 2 — Golden-path distillation workflow

## Objective

Provide one dependable workflow from examples to a candidate package.

## Pipeline

```text
Load split-safe task
  -> run baseline
  -> extract train micro-rules
  -> consolidate and synthesize
  -> compile candidate prompt
  -> evaluate on validation
  -> analyze fixed/broken cases
  -> optionally refine on validation
  -> create package candidate
```

## Engineering work

- Provide one CLI/API operation for the golden path.
- Make stage inputs, outputs, hashes, and resume behavior explicit.
- Do not evaluate test by default.
- Prevent reuse of incompatible cached artifacts.
- Persist pre-refinement and post-refinement prompt hashes.
- Mark refined prompts as validation-tuned.
- Keep refinement optional generally, with an explicit paper-style profile.
- Produce a concise run summary:
  - baseline;
  - candidate;
  - strict metric delta;
  - malformed delta;
  - fixed/broken counts;
  - prompt token cost;
  - provenance status;
  - package status.

## Deliverables

- golden-path command;
- resumable stage execution;
- run summary;
- pre/post-refinement comparison;
- candidate package record.

## Human Gate 2 — First full development run

The maintainer runs the pipeline and manually reviews:

- at least 20 random micro-rules;
- every synthesized rule;
- every malformed response;
- top fixed and broken cases;
- prompt text for leakage or label shortcuts;
- prompt length and rule redundancy.

The human records acceptable patterns, bad-rule patterns, unacceptable regressions, and refinement-overfit concerns.

## Exit criteria

- Full run completes without manual artifact repair.
- Improvement can be explained from concrete rules and cases.
- Test remains untouched.
- The candidate is reproducible from persisted inputs.

---

# Phase 3 — Rule quality and consolidation abstraction

## Objective

Make the system optimize useful rules rather than clustering mechanics.

## Engineering work

### Strategy interface

Introduce a common consolidation contract.

Initial strategies:

1. DBSCAN
2. HDBSCAN
3. no clustering
4. deterministic task-key grouping
5. optional hypothesis-grouped synthesis for tasks such as ContractNLI

Do not add more strategies until these are compared.

### Rule-quality artifacts

For each rule, persist:

- source case IDs and labels;
- support count and ratio;
- examples fixed and broken;
- net value;
- token cost;
- conflicts;
- redundancy group;
- target-model sensitivity;
- active/pruned status.

### Structural health

For clustered strategies validate:

- cluster-count range;
- noise rate;
- largest-cluster rate;
- single-cluster collapse;
- missing assignments;
- deterministic assignment hash.

Structural failure blocks promotion but does not erase raw evaluation results.

### Ablation

Support leave-one-rule-out or grouped ablation as an explicit offline diagnostic, not an automatic cost on every run.

## Deliverables

- consolidation strategy interface;
- strategy adapters;
- deterministic control strategy;
- rule-quality report;
- cluster-health report;
- rule-level regression analysis.

## Human Gate 3 — Controlled strategy study

The maintainer freezes dataset, models, decoding, and validation set, then runs only the approved strategies.

Manual review covers:

- cluster summaries;
- giant or collapsed clusters;
- minority-label behavior;
- representative rules;
- token cost;
- model-specific regressions.

## Exit criteria

- Strategy decisions are based on downstream value and reliability.
- One-cluster results cannot be promoted.
- Major gains and regressions are attributable to specific rules.
- A non-clustering control exists.

---

# Phase 4 — Deterministic prompt compiler

## Objective

Compile validated semantic rules into target-aware prompt variants.

## Engineering work

### Semantic rule representation

Each rule carries:

- stable ID;
- instruction;
- priority;
- exceptions;
- task/label scope;
- source support;
- conflict relationships;
- token cost;
- status.

### Compilation stages

```text
validated semantic rules
  -> conflict ordering
  -> redundancy removal
  -> prioritization
  -> token-budget allocation
  -> output-contract insertion
  -> target-model rendering
  -> compiled prompt
```

### Variants

Produce:

- full;
- balanced;
- compact;
- minimal.

Each records:

- prompt hash;
- token count;
- included and excluded rule IDs;
- exclusion reasons;
- target adapter;
- output-contract version.

### Initial adapters

- generic OpenAI-compatible chat;
- llama.cpp-compatible chat;
- one selected model-family adapter.

Output-format instructions remain separate from semantic rules.

## Deliverables

- semantic rule schema;
- deterministic compiler;
- token-budget optimizer;
- prompt variants;
- target adapters;
- compilation manifest.

## Human Gate 4 — Prompt and budget review

The maintainer reviews every variant for correctness, contradictions, excess wording, model assumptions, and output-contract clarity, then runs each approved variant on the same validation set.

## Exit criteria

- Unchanged inputs compile to identical hashes.
- Token/performance tradeoffs are visible.
- Output-contract changes are independently testable.
- At least one compact variant remains useful.

---

# Phase 5 — First-class RuleKiln Prompt Package

## Objective

Create the portable product artifact.

## Package layout

```text
task-name.rkprompt/
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

## Required package information

- schema and package version;
- task mode and output schema;
- prompt and rule-set hashes;
- token count;
- runtime defaults;
- derived and refinement splits;
- validated models;
- strict metrics;
- malformed rate;
- lifecycle status;
- provenance references.

## CLI

```bash
rulekiln package build <job-id> --strategy <strategy>
rulekiln package inspect <package>
rulekiln package test <package> --provider <profile>
rulekiln package export <package> --target llama-cpp
```

## Lifecycle

- `experimental`
- `candidate`
- `validated`
- `released`

Only `candidate` may be automatic. `validated` requires repeatability and target-runtime testing. `released` requires explicit human approval.

## Deliverables

- package schema;
- builder;
- inspector;
- independent regression runner;
- export adapters;
- compatibility policy.

## Human Gate 5 — Independent package test

The maintainer copies the package outside the worker environment and verifies:

- hashes match the source run;
- runtime defaults are sufficient;
- regression cases execute independently;
- no database or orchestration dependency remains;
- limitations are accurately documented.

## Exit criteria

- A package works on another machine/runtime.
- Inspection explains provenance and supported models.
- Release status cannot change without a human.

---

# Phase 6 — Reference local runtime

## Objective

Demonstrate practical offline deployment.

## Engineering work

- Build a minimal package runner.
- Support OpenAI-compatible local endpoints and llama.cpp.
- Apply deterministic decoding settings.
- Validate JSON output.
- Report malformed responses and retries.
- Check context size, chat template, and output-contract compatibility.
- Record model, quantization, hardware, latency, memory, tokens/second, and task metrics.
- Keep the runner independent from the training pipeline.

## Deliverables

- reference runner;
- llama.cpp adapter;
- compatibility report;
- offline regression command;
- deployment example.

## Human Gate 6 — Local hardware evaluation

The maintainer runs the package on real local hardware and decides whether accuracy, prompt length, latency, and memory are useful. Phone deployment remains a future claim until actually demonstrated.

## Exit criteria

- One package runs fully offline.
- Runtime measurements are recorded.
- Regression cases pass on the declared target model.
- Mobile feasibility is assessed honestly.

---

# Phase 7 — Demonstrate value across tasks

## Objective

Prove that RuleKiln provides repeatable value beyond one benchmark.

## Reference tasks

1. ContractNLI as a research benchmark.
2. A practical structured-classification task.
3. A bounded local task designed for offline deployment.

## Required report per task

- dataset/revision and split hashes;
- baseline and package prompt;
- teacher/student/embedding models;
- consolidation strategy;
- strict and per-label metrics;
- malformed count/rate;
- fixed/broken cases;
- prompt tokens;
- teacher and inference cost;
- local runtime measurements;
- repeat-run variation;
- known limitations.

## Public-claim standard

For important claims:

- run at least three unchanged evaluations when feasible;
- persist prompt, case-set, cluster, and prediction hashes;
- report variation;
- use paired confidence analysis;
- do not call a non-significant point estimate a definitive win.

## Human Gate 7 — Evidence and claims review

The maintainer reviews benchmark fairness, qualitative failures, uncertainty, licensing, and public wording. Results are not published automatically.

## Exit criteria

- At least two tasks show repeatable practical value over a strong scaffold.
- At least one task runs offline.
- Reports include costs, malformed outputs, and regressions.

---

# Phase 8 — Portability and classroom evaluation

## Objective

Measure transfer across small models without making multi-model orchestration part of the golden path.

## Engineering work

- Evaluate one semantic rule set on multiple student models.
- Separate universal rules, target rendering, and model-specific rules.
- Produce a compatibility matrix with model, quantization, metric, malformed rate, variant, and adapter.
- Detect cross-model regressions.

## Human Gate 8 — Supported-model decision

The maintainer decides which models are validated, experimental, or unsupported.

## Exit criteria

- Packages explicitly name validated target models.
- Unsupported portability is not implied.
- Model-specific adaptations remain separate from semantic rules.

---

# Phase 9 — Bounded distilled subagents

## Objective

Extend the package architecture only after classification and structured output are stable.

## Initial scope

Support one-step bounded decisions:

- choose a tool or action;
- emit validated arguments;
- escalate uncertainty;
- refuse unsupported actions.

Do not begin with unrestricted autonomous multi-step agents.

## Package additions

- role and capabilities;
- allowed tools;
- decision rules;
- input/output schemas;
- escalation and refusal rules;
- regression traces;
- validated models.

## Evaluation

Measure tool-selection accuracy, argument correctness, escalation quality, refusal correctness, malformed calls, and task completion.

## Human Gate 9 — Use-case and safety approval

The maintainer approves one bounded use case, permissions, escalation policy, test corpus, and release risk.

## Exit criteria

- One bounded subagent package passes regression testing.
- Permissions and escalation are explicit.
- The package remains portable and independently testable.

---

## 4. Parallel paper-validation lane

Keep paper-oriented validation separate from the product roadmap.

### Sequence

1. Create a split-safe, full-document ContractNLI dataset.
2. Establish a scaffold baseline.
3. Run extraction/consolidation without refinement.
4. Run refinement.
5. Freeze the prompt.
6. Evaluate locked test once.
7. Substitute one component at a time:
   - embedding;
   - teacher;
   - student;
   - quantization;
   - consolidation;
   - compiler variant.

This supports controlled comparison, not an exact reproduction claim when models, prompts, data conversion, or runtime differ.

---

## 5. Deferred work

Do not let these block the first product milestones:

- more clustering algorithms;
- automatic production promotion;
- major dashboard expansion;
- StereoSet and LogiQA;
- fine-tuning/TextGrad;
- large repository reorganization;
- mobile SDKs before a stable local runner;
- unrestricted tool agents;
- exact paper-score chasing with non-equivalent components.

---

## 6. Cross-phase standards

### Testing

Every phase includes:

- unit tests;
- artifact round-trip tests;
- deterministic hashing tests;
- legacy compatibility tests;
- one real smoke run;
- documented human review.

### Branch boundaries

Keep phases independently reviewable. Do not combine evaluator changes, clustering retuning, package schemas, and runtime deployment in one branch.

Suggested commit order:

1. schemas/docs;
2. core implementation;
3. artifact persistence;
4. API/UI;
5. tests;
6. human-run findings and fixes.

### Cost control

Teacher, synthesis, refinement, full validation, test, and repeated runs require explicit human initiation. Coding agents should prepare commands and expected artifacts, not assume permission to spend inference budget.

---

## 7. Program-level success criteria

RuleKiln can credibly be described as a strong and useful project when it demonstrates:

1. repeatable improvement over strong scaffold baselines;
2. packages that run without the orchestration stack;
3. measurable value on a local or quantized model;
4. transparent token, malformed, fixed/broken, and compatibility tradeoffs;
5. complete data/config/rule/prompt/prediction provenance;
6. split-safe and stale-artifact-safe evaluation;
7. a usable examples-to-package golden path;
8. independent target-runtime regression testing;
9. honest public reports including uncertainty and negative findings;
10. one compelling offline or specialized reference application.

---

## 8. Immediate execution order

Start with **Phase 0**, not another clustering sweep.

The first branch should produce:

1. product definition;
2. golden-path document;
3. initial `.rkprompt` schema;
4. authoritative evaluation-flow diagram;
5. reference task/model/runtime decision record.

Stop at Human Gate 0. After approval, implement Phase 1 and run one real combined-file smoke job before beginning new optimization work.
