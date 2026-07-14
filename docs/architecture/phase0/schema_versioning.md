# RuleKiln Package Schema Versioning and Legacy Compatibility

**Status:** Proposed

## Version fields

RuleKiln packages use two distinct versions:

1. `schema_version` — compatibility of the package manifest and package layout.
2. `package_version` — semantic version of a particular distilled package.

Example:

```json
{
  "schema_version": "1.0.0",
  "package_version": "0.1.0"
}
```

## Schema compatibility rules

RuleKiln follows semantic-versioning rules for package schemas.

### Major version

A major version change may:

- remove or rename required fields;
- change field meaning;
- change hash semantics;
- change archive layout;
- change required verification behavior.

A reader must reject unsupported major versions.

### Minor version

A minor version may:

- add optional fields;
- add optional files;
- add new enum values when safely ignorable;
- add new compatibility metadata.

Readers supporting the same major version should ignore unknown optional fields.

A minor version must not change the meaning of an existing field.

### Patch version

A patch version may:

- clarify validation;
- correct non-semantic documentation;
- tighten constraints without changing valid existing payload meaning.

## Required reader behavior

A package reader must:

- verify the manifest before loading referenced files;
- reject unsupported major versions;
- preserve unknown fields when performing a read/write round trip where practical;
- reject missing required files;
- report unknown lifecycle or task-mode values clearly;
- never infer missing historical provenance from current repository configuration.

## Package immutability

- `experimental` packages may be rebuilt in development, but each build has a new content hash.
- `candidate`, `validated`, and `released` packages are immutable artifacts.
- Any content change requires a new `package_version`.
- A `released` package must never be rewritten in place.

## Legacy runs

Historical RuleKiln runs may be imported into an experimental package only when the available evidence can be represented truthfully.

Legacy imports must record:

```json
{
  "legacy_source": true,
  "legacy_limitations": [
    "missing_prediction_hash",
    "runtime_decoding_unverified"
  ]
}
```

A legacy import with incomplete required provenance cannot be marked `validated` or `released`.

## Migration

Schema migration must be explicit.

Recommended future command:

```bash
rulekiln package migrate old.rkprompt --target-schema 2.0.0
```

Migration must:

- preserve the original package;
- write a new package;
- record source package hashes;
- document lossy or inferred fields;
- refuse migrations that would fabricate provenance.

## Hashing

- `manifest.json` uses canonical JSON hashing.
- Text and JSONL files are hashed as normalized UTF-8 bytes with LF line endings.
- Binary files are hashed as raw bytes.
- The manifest lists each file's SHA-256.
- Archive byte identity is not the primary content identity; manifest file hashes are.
