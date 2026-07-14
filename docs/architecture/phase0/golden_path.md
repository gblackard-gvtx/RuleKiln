# RuleKiln Golden Path

**Status:** Proposed  
**Supported mode:** Multiclass classification with strict JSON output  
**Phase:** 0 — Product contract and golden-path design

## Purpose

The golden path is the smallest complete workflow that produces a portable, auditable RuleKiln Prompt Package.

The path deliberately excludes optional research branches until the core product flow is stable.

## Authoritative flow

```text
Task definition
    ↓
Dataset loading and split validation
    ↓
Baseline evaluation
    ↓
Train-case student execution
    ↓
Teacher micro-rule extraction
    ↓
Rule consolidation
    ↓
Rule synthesis
    ↓
Deterministic prompt compilation
    ↓
Validation evaluation
    ↓
Paired fixed/broken analysis
    ↓
Optional validation refinement
    ↓
Candidate selection
    ↓
Prompt package build
    ↓
Independent package regression test
```

## Source-of-truth hierarchy

The authoritative hierarchy is:

```text
Per-case inputs and predictions
    ↓
Derived strict metrics
    ↓
Paired comparisons
    ↓
Eligibility gates
    ↓
Strategy selection
    ↓
Package lifecycle assessment
```

Persisted aggregate summaries are caches. They must not override per-case records when hashes or case-set identities disagree.

## Stage contracts

### 1. Task definition

Required inputs:

- task mode;
- label vocabulary;
- input-field mapping;
- output schema;
- dataset path or paths;
- split routing;
- model-role profiles;
- decoding settings;
- parser configuration;
- quality-gate configuration.

Required outputs:

- immutable task snapshot;
- task configuration hash;
- output-schema hash.

### 2. Dataset loading and split validation

Required behavior:

- every case has a stable ID;
- every case has a recognized split;
- train, validation, and test case-set hashes are persisted;
- missing and unknown splits fail closed;
- combined files are allowed only with explicit stage routing;
- source overlap warnings are distinguished from pipeline leakage errors.

Required outputs:

- ordered split-specific case lists;
- generation or source manifest;
- split-leakage validation artifact.

### 3. Baseline evaluation

The validation baseline and distilled candidates must use the same:

- student model;
- decoding configuration;
- parser;
- output schema;
- validation case set.

Required outputs:

- per-case prediction records;
- strict metrics;
- valid-output diagnostic metrics;
- malformed artifacts;
- per-label metrics;
- prediction and evaluation hashes.

### 4. Train-case student execution

Only train cases are permitted.

Required outputs:

- train predictions;
- train successes and failures;
- ordered train case-set hash.

### 5. Teacher micro-rule extraction

Each train case may yield one or more micro-rules.

Required outputs:

- micro-rule artifact;
- source case IDs;
- extraction model and configuration;
- extraction prompt hash;
- micro-rule artifact hash.

### 6. Rule consolidation

Consolidation is a strategy interface.

The first implementation may use the current selected default, but downstream stages must not depend directly on DBSCAN or HDBSCAN.

Required outputs:

- group assignments;
- requested and effective strategy configuration;
- structural-health findings;
- assignment hash.

### 7. Rule synthesis

The synthesis role produces semantic rule candidates from accepted groups.

Required outputs:

- synthesized rules;
- source micro-rule IDs;
- synthesis model/configuration;
- synthesized-rule hash.

### 8. Deterministic prompt compilation

The compiler:

- resolves accepted rule ordering;
- removes deterministic duplicates;
- applies rule priorities and exceptions;
- inserts label definitions;
- inserts the strict output contract;
- applies the target-model adapter;
- records included and excluded rule IDs;
- records token count and compilation configuration.

Required outputs:

- compiled prompt;
- compilation manifest;
- prompt hash.

### 9. Validation evaluation

Only validation cases are permitted.

Required outputs:

- candidate per-case predictions;
- strict macro-F1 and accuracy;
- valid-output diagnostics;
- malformed count, rate, and categories;
- per-label metrics;
- prediction and evaluation hashes.

### 10. Paired analysis

Baseline and candidate are joined by stable case ID.

Required outputs:

- fixed;
- broken;
- unchanged correct;
- unchanged wrong;
- net case change;
- per-label transition analysis;
- paired-comparison hash.

### 11. Optional validation refinement

Refinement is optional in the general product path.

When enabled:

- passing and failing validation examples are sampled;
- pre-refinement and post-refinement prompts are preserved;
- the result is marked `validation_tuned=true`;
- validation case-set and refinement hashes are persisted;
- test remains inaccessible.

### 12. Candidate selection

Selection requires:

- valid evaluation provenance;
- matching case-set hashes;
- strict primary metric;
- configured eligibility gates;
- no blocking stale-artifact finding.

Selection does not imply package validation or release.

### 13. Prompt package build

The package builder copies immutable package-relevant artifacts into the canonical package layout.

The package records:

- prompt;
- semantic rules;
- output schema;
- runtime defaults;
- evaluation summary;
- compatibility declarations;
- provenance;
- regression cases;
- lifecycle status.

### 14. Independent package regression test

The package must be usable without database access or pipeline workers.

The standalone runner validates:

- archive and manifest integrity;
- file hashes;
- prompt hash;
- output schema;
- runtime compatibility;
- regression-case behavior.

## Initial defaults

Proposed first-path defaults:

- task mode: `classification`;
- output media type: `application/json`;
- output object: exactly one `label` field;
- primary metric: `strict_macro_f1`;
- malformed response: incorrect prediction;
- target runtime: OpenAI-compatible local endpoint;
- reference runtime: llama.cpp;
- release policy: manual only.

## Human intervention points

Human approval is required before:

1. a costly full extraction or synthesis run;
2. accepting synthesized rules after qualitative review;
3. enabling validation refinement for a benchmark claim;
4. accessing the test split;
5. marking a package `validated`;
6. marking a package `released`.

## Golden-path exclusions

The initial path excludes:

- multi-student classroom orchestration;
- automatic cluster sweeps;
- automatic package release;
- mobile SDK export;
- tool-using subagents;
- fine-tuning;
- unrestricted prompt search;
- test-driven prompt iteration.

## Acceptance test

The golden path is implemented successfully when:

```text
examples + task configuration
    → candidate.rkprompt
    → standalone regression result
```

can be completed without manual database repair, hidden service dependencies, or ambiguous metric provenance.
