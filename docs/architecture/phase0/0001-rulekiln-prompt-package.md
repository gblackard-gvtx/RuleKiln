# ADR-0001: RuleKiln Prompt Package as the Primary Product Artifact

- **Status:** Proposed
- **Date:** 2026-07-14
- **Decision owner:** Maintainer

## Context

RuleKiln currently produces prompts, rule artifacts, metrics, provenance, and strategy-comparison records across a pipeline run. These artifacts are useful for research and debugging, but they do not yet form a portable product that can be copied, inspected, tested, and executed independently.

Treating the compiled prompt alone as the output omits:

- the output contract;
- target runtime defaults;
- validated model compatibility;
- regression cases;
- evaluation evidence;
- provenance;
- lifecycle status.

## Decision

RuleKiln will treat a versioned Prompt Package as its primary product artifact.

The distributable format is a deterministic ZIP archive with the `.rkprompt` extension.

The initial canonical layout is:

```text
manifest.json
system_prompt.txt
semantic_rules.json
output_schema.json
runtime_defaults.json
evaluation_summary.json
compatibility.json
provenance.json
regression_cases.jsonl
README.md
```

An unpacked directory with the same layout may be used during development.

## Package identity

A package is identified by:

- `package_id`;
- semantic `package_version`;
- manifest schema version;
- prompt hash;
- semantic-rule-set hash.

A validated or released package is immutable. Any content change requires a new package version and new hashes.

## Lifecycle

The initial lifecycle states are:

- `experimental`;
- `candidate`;
- `validated`;
- `released`.

Only a human may transition a package to `released`.

## Consequences

### Positive

- The product output is portable.
- Runtime use no longer depends on RuleKiln workers or a database.
- Evaluation and provenance travel with the prompt.
- Compatibility claims can be explicit.
- Regression tests can be run on another machine.

### Negative

- Package schemas require compatibility discipline.
- Build and verification tooling must be implemented.
- Large regression artifacts may increase package size.
- Historical runs may not contain enough provenance to build validated packages.

## Rejected alternatives

### Compiled prompt text only

Rejected because it cannot communicate output schema, runtime settings, evidence, or compatibility.

### Database record as the product

Rejected because it is not portable and couples runtime use to RuleKiln infrastructure.

### Unversioned directory

Rejected because it provides no stable compatibility or distribution contract.

## Follow-up

- Implement the package manifest schema.
- Implement deterministic archive construction.
- Implement `package build`, `package inspect`, and `package test`.
- Define validation and release policy in a later phase.
