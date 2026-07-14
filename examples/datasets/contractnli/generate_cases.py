from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterable, Mapping, Sequence

DATASET_ID = "kiddothe2b/contract-nli"
DATASET_CONFIG = "contractnli_a"
DEFAULT_DATASET_REVISION = "refs/convert/parquet"
GENERATOR_SCHEMA_VERSION = "rulekiln.contractnli_generator.v2.1"
CASE_SCHEMA_VERSION = "rulekiln.case.v1"
MANIFEST_SCHEMA_VERSION = "rulekiln.dataset_generation.v1"

# Fallback for kiddothe2b/contract-nli, contractnli_a.
# 0=contradiction, 1=entailment, 2=neutral
FALLBACK_LABEL_NAMES = ["contradiction", "entailment", "neutral"]

# Full splits are the safe default. Pass explicit limits for smoke/development runs.
TRAIN_LIMIT: int | None = None
VAL_LIMIT: int | None = None
TEST_LIMIT: int | None = None

CANONICAL_LABELS = ("contradiction", "entailment", "neutral")
LABEL_ALIASES = {
    "contradiction": "contradiction",
    "entailment": "entailment",
    "neutral": "neutral",
    "notmentioned": "neutral",
    "not_mentioned": "neutral",
    "not mentioned": "neutral",
}

SPLIT_SEED_OFFSETS = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


@dataclass(frozen=True)
class PreparedPremise:
    text: str
    original_chars: int
    written_chars: int
    truncated: bool
    truncation_policy: str


@dataclass(frozen=True)
class GeneratedSplit:
    split: str
    source_case_count: int
    source_document_count: int
    selected_case_count: int
    selected_document_count: int
    label_counts: dict[str, int]
    truncated_count: int
    premise_lengths: list[int]
    case_ids: list[str]
    document_hashes: set[str]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(payload)


def _normalize_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    compact = normalized.replace("_", "").replace(" ", "")

    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    if compact in LABEL_ALIASES:
        return LABEL_ALIASES[compact]

    raise ValueError(
        f"Unsupported ContractNLI label {value!r}; "
        f"expected one of {sorted(LABEL_ALIASES)}"
    )


def _extract_label_names(dataset: Any) -> list[str]:
    label_feature = dataset["train"].features.get("label")
    names = getattr(label_feature, "names", None)
    if names:
        return [str(name) for name in names]
    return list(FALLBACK_LABEL_NAMES)


def _row_label(
    row: Mapping[str, object],
    source_label_names: Sequence[str],
) -> tuple[int, str, str]:
    try:
        label_id = int(row["label"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label value in row: {row.get('label')!r}") from exc

    try:
        source_label = str(source_label_names[label_id])
    except IndexError as exc:
        raise ValueError(
            f"Label id {label_id} is outside discovered label names "
            f"{list(source_label_names)!r}"
        ) from exc

    return label_id, source_label, _normalize_label(source_label)


def _document_hash(row: Mapping[str, object]) -> str:
    return _sha256_text(str(row["premise"]))


def _hypothesis_hash(row: Mapping[str, object]) -> str:
    return _sha256_text(str(row["hypothesis"]))


def _source_row_index(row: Mapping[str, object]) -> int:
    try:
        return int(row["_source_row_index"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Generated rows must contain a stable _source_row_index"
        ) from exc


def _make_case_id(split: str, row: Mapping[str, object]) -> str:
    """Create a unique, auditable ID while preserving duplicate source rows.

    ContractNLI contains some repeated premise/hypothesis content. Content hashes
    alone therefore cannot be used as globally unique row identifiers. The
    source-row ordinal is included so paper-fidelity generation preserves every
    source example without silently deduplicating or colliding.
    """
    document_id = _document_hash(row)[:16]
    hypothesis_id = _hypothesis_hash(row)[:12]
    source_index = _source_row_index(row)
    return (
        f"contractnli_{split}_{source_index:06d}_"
        f"{document_id}_{hypothesis_id}"
    )


def _prepare_premise(
    premise: str,
    *,
    max_chars: int | None,
    allow_truncation: bool,
    truncation_strategy: str,
) -> PreparedPremise:
    original_chars = len(premise)

    if max_chars is None or original_chars <= max_chars:
        return PreparedPremise(
            text=premise,
            original_chars=original_chars,
            written_chars=original_chars,
            truncated=False,
            truncation_policy="none",
        )

    if max_chars <= 0:
        raise ValueError("--max-premise-chars must be greater than zero")

    if not allow_truncation:
        raise ValueError(
            f"Premise has {original_chars} characters, exceeding "
            f"--max-premise-chars={max_chars}. "
            "Truncation is disabled by default because it can remove the "
            "evidence needed for ContractNLI. Pass --allow-truncation explicitly."
        )

    if truncation_strategy == "head":
        text = premise[:max_chars]
        policy = f"head:{max_chars}"
    elif truncation_strategy == "head_tail":
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        separator = "\n\n[...TRUNCATED...]\n\n"
        budget = max_chars - len(separator)
        if budget <= 1:
            raise ValueError(
                "--max-premise-chars is too small for head_tail truncation"
            )
        head_chars = budget // 2
        tail_chars = budget - head_chars
        text = premise[:head_chars] + separator + premise[-tail_chars:]
        policy = f"head_tail:{max_chars}"
    else:
        raise ValueError(f"Unsupported truncation strategy: {truncation_strategy}")

    return PreparedPremise(
        text=text,
        original_chars=original_chars,
        written_chars=len(text),
        truncated=True,
        truncation_policy=policy,
    )


def _group_rows_by_document(
    rows: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_document_hash(row)].append(row)

    # Stable ordering inside each contract.
    for document_rows in grouped.values():
        document_rows.sort(
            key=lambda row: (
                _hypothesis_hash(row),
                str(row.get("label", "")),
            )
        )

    return dict(grouped)


def _select_rows(
    rows: Sequence[dict[str, object]],
    *,
    split: str,
    limit: int | None,
    seed: int,
    sampling_mode: str,
) -> list[dict[str, object]]:
    if limit is not None and limit < 0:
        raise ValueError(f"{split} limit must be >= 0 or omitted")

    if limit is None or limit >= len(rows):
        selected = list(rows)
        selected.sort(key=lambda row: (_document_hash(row), _hypothesis_hash(row)))
        return selected

    rng = random.Random(seed + SPLIT_SEED_OFFSETS[split])

    if sampling_mode == "rows":
        selected = list(rows)
        rng.shuffle(selected)
        selected = selected[:limit]
        selected.sort(key=lambda row: (_document_hash(row), _hypothesis_hash(row)))
        return selected

    if sampling_mode != "documents":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    grouped = _group_rows_by_document(rows)
    document_ids = sorted(grouped)
    rng.shuffle(document_ids)

    selected: list[dict[str, object]] = []
    for document_id in document_ids:
        document_rows = grouped[document_id]

        # Preserve complete contracts. Stop before exceeding the requested case cap.
        if selected and len(selected) + len(document_rows) > limit:
            continue
        if not selected and len(document_rows) > limit:
            # A tiny limit can be smaller than one complete contract. Preserve the
            # contract and make the deviation visible in the manifest.
            selected.extend(document_rows)
            break

        selected.extend(document_rows)
        if len(selected) >= limit:
            break

    selected.sort(key=lambda row: (_document_hash(row), _hypothesis_hash(row)))
    return selected


def _write_cases_to_file(
    rows: Sequence[dict[str, object]],
    *,
    target_split: str,
    f: IO[str],
    source_label_names: Sequence[str],
    source_revision: str,
    max_premise_chars: int | None,
    allow_truncation: bool,
    truncation_strategy: str,
) -> GeneratedSplit:
    source_documents = {_document_hash(row) for row in rows}
    selected_documents: set[str] = set()
    label_counts: Counter[str] = Counter()
    premise_lengths: list[int] = []
    case_ids: list[str] = []
    truncated_count = 0

    for row in rows:
        label_id, source_label, label = _row_label(row, source_label_names)
        premise = str(row["premise"])
        hypothesis = str(row["hypothesis"])
        prepared = _prepare_premise(
            premise,
            max_chars=max_premise_chars,
            allow_truncation=allow_truncation,
            truncation_strategy=truncation_strategy,
        )

        document_hash = _document_hash(row)
        hypothesis_hash = _hypothesis_hash(row)
        case_id = _make_case_id(target_split, row)

        case: dict[str, object] = {
            "schema_version": CASE_SCHEMA_VERSION,
            "id": case_id,
            "split": target_split,
            "task_mode": "classification",
            "input": {
                "premise": prepared.text,
                "hypothesis": hypothesis,
            },
            "expected": {"label": label},
            "evaluation": {
                "assertions": [
                    {
                        "type": "must_equal",
                        "path": "$.label",
                        "value": label,
                        "weight": 1.0,
                    }
                ]
            },
            "metadata": {
                "source": "contractnli",
                "source_dataset": DATASET_ID,
                "source_config": DATASET_CONFIG,
                "source_revision": source_revision,
                "source_label": source_label,
                "canonical_label": label,
                "label_id": label_id,
                "document_hash": document_hash,
                "hypothesis_hash": hypothesis_hash,
                "original_premise_chars": prepared.original_chars,
                "written_premise_chars": prepared.written_chars,
                "truncated": prepared.truncated,
                "truncation_policy": prepared.truncation_policy,
                "generator_schema_version": GENERATOR_SCHEMA_VERSION,
                "source_row_index": _source_row_index(row),
            },
            "weight": 1.0,
        }

        f.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

        case_ids.append(case_id)
        selected_documents.add(document_hash)
        label_counts[label] += 1
        premise_lengths.append(prepared.original_chars)
        truncated_count += int(prepared.truncated)

    return GeneratedSplit(
        split=target_split,
        source_case_count=len(rows),
        source_document_count=len(source_documents),
        selected_case_count=len(rows),
        selected_document_count=len(selected_documents),
        label_counts=dict(sorted(label_counts.items())),
        truncated_count=truncated_count,
        premise_lengths=premise_lengths,
        case_ids=case_ids,
        document_hashes=selected_documents,
    )


def write_rulekiln_cases(
    rows: Sequence[dict[str, object]],
    *,
    target_split: str,
    out_path: Path,
    source_label_names: Sequence[str],
    source_revision: str,
    max_premise_chars: int | None,
    allow_truncation: bool,
    truncation_strategy: str,
    mode: str = "w",
) -> GeneratedSplit:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode, encoding="utf-8", newline="\n") as f:
        return _write_cases_to_file(
            rows,
            target_split=target_split,
            f=f,
            source_label_names=source_label_names,
            source_revision=source_revision,
            max_premise_chars=max_premise_chars,
            allow_truncation=allow_truncation,
            truncation_strategy=truncation_strategy,
        )


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _length_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def _split_manifest(
    *,
    result: GeneratedSplit,
    source_case_count: int,
    source_document_count: int,
    file_path: Path,
) -> dict[str, object]:
    return {
        "source_case_count": source_case_count,
        "source_document_count": source_document_count,
        "written_case_count": result.selected_case_count,
        "written_document_count": result.selected_document_count,
        "label_counts": result.label_counts,
        "truncated_count": result.truncated_count,
        "premise_chars": _length_summary(result.premise_lengths),
        "ordered_case_ids_hash": _canonical_json_hash(result.case_ids),
        "document_set_hash": _canonical_json_hash(sorted(result.document_hashes)),
        "file": {
            "path": file_path.name,
            "sha256": _sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        },
    }


def _validate_outputs(
    split_results: Mapping[str, GeneratedSplit],
) -> dict[str, object]:
    all_case_ids = [
        case_id
        for result in split_results.values()
        for case_id in result.case_ids
    ]
    duplicate_case_ids = [
        case_id
        for case_id, count in Counter(all_case_ids).items()
        if count > 1
    ]

    overlaps: dict[str, list[str]] = {}
    split_names = sorted(split_results)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            shared = sorted(
                split_results[left].document_hashes
                & split_results[right].document_hashes
            )
            if shared:
                overlaps[f"{left}__{right}"] = shared

    return {
        "unique_case_ids": not duplicate_case_ids,
        "duplicate_case_ids": duplicate_case_ids,
        "document_overlap_across_splits": bool(overlaps),
        "document_overlap_details": overlaps,
        "canonical_labels": list(CANONICAL_LABELS),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RuleKiln ContractNLI cases without silently truncating "
            "document evidence."
        )
    )
    parser.add_argument(
        "--discover-labels",
        action="store_true",
        help="Print discovered source labels and exit without writing files.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=TRAIN_LIMIT,
        help="Maximum train cases. Omit for the full split.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=VAL_LIMIT,
        help="Maximum validation cases. Omit for the full split.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=TEST_LIMIT,
        help="Maximum test cases. Omit for the full split.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("documents", "rows"),
        default="documents",
        help=(
            "Subset by complete contracts (recommended) or individual rows. "
            "Ignored when no split limit is set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Deterministic subset-selection seed. Default: 1729.",
    )
    parser.add_argument(
        "--max-premise-chars",
        type=int,
        default=None,
        help=(
            "Optional premise character cap. No cap by default. "
            "Requires --allow-truncation when any premise exceeds the cap."
        ),
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Explicitly permit premise truncation.",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=("head", "head_tail"),
        default="head_tail",
        help="Used only with --max-premise-chars and --allow-truncation.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
        help=(
            "Hugging Face revision used in hf:// parquet URIs. "
            "Use an immutable commit SHA for reproducible paper-fidelity runs."
        ),
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help=(
            "Also write cases.jsonl containing all splits. Disabled by default "
            "to reduce accidental split leakage."
        ),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not materialize the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_premise_chars is None and args.allow_truncation:
        raise SystemExit(
            "--allow-truncation requires --max-premise-chars"
        )

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(
            "datasets is not installed; run: uv pip install datasets",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    splits_to_load = ["train", "validation"]
    if not args.skip_test:
        splits_to_load.append("test")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = load_dataset(
            "parquet",
            data_files={
                split: (
                    f"hf://datasets/{DATASET_ID}@{args.dataset_revision}/"
                    f"{DATASET_CONFIG}/{split}/*.parquet"
                )
                for split in splits_to_load
            },
        )

    source_label_names = _extract_label_names(dataset)

    if args.discover_labels:
        for index, source_label in enumerate(source_label_names):
            print(
                f"{index}: {source_label} -> "
                f"{_normalize_label(source_label)}"
            )
        return

    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: dict[str, list[dict[str, object]]] = {
        split: [
            {
                **dict(row),
                "_source_row_index": source_index,
            }
            for source_index, row in enumerate(dataset[split])
        ]
        for split in splits_to_load
    }

    limits: dict[str, int | None] = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }

    selected_rows: dict[str, list[dict[str, object]]] = {
        split: _select_rows(
            rows,
            split=split,
            limit=limits[split],
            seed=args.seed,
            sampling_mode=args.sampling_mode,
        )
        for split, rows in raw_rows.items()
    }

    split_results: dict[str, GeneratedSplit] = {}
    split_paths: dict[str, Path] = {}

    for split in splits_to_load:
        output_path = generated_dir / f"cases.{split}.jsonl"
        split_paths[split] = output_path
        split_results[split] = write_rulekiln_cases(
            selected_rows[split],
            target_split=split,
            out_path=output_path,
            source_label_names=source_label_names,
            source_revision=args.dataset_revision,
            max_premise_chars=args.max_premise_chars,
            allow_truncation=args.allow_truncation,
            truncation_strategy=args.truncation_strategy,
        )

    combined_metadata: dict[str, object] | None = None
    if args.write_combined:
        combined_path = generated_dir / "cases.jsonl"
        with combined_path.open("w", encoding="utf-8", newline="\n") as combined:
            for split in splits_to_load:
                with split_paths[split].open("r", encoding="utf-8") as split_file:
                    for line in split_file:
                        combined.write(line)
        combined_metadata = {
            "path": combined_path.name,
            "sha256": _sha256_file(combined_path),
            "size_bytes": combined_path.stat().st_size,
        }

    labels_payload = {
        "source_labels": source_label_names,
        "canonical_labels": list(CANONICAL_LABELS),
        "mapping": {
            source_label: _normalize_label(source_label)
            for source_label in source_label_names
        },
    }
    labels_path = generated_dir / "labels.json"
    labels_path.write_text(
        json.dumps(labels_payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    validation = _validate_outputs(split_results)
    if validation["document_overlap_across_splits"]:
        print(
            "WARNING: document overlap detected across source splits. "
            "Review generation_manifest.json.",
            file=sys.stderr,
        )
    if not validation["unique_case_ids"]:
        raise RuntimeError("Duplicate generated case IDs detected")

    manifest_splits: dict[str, object] = {}
    for split in splits_to_load:
        source_documents = {
            _document_hash(row)
            for row in raw_rows[split]
        }
        manifest_splits[split] = _split_manifest(
            result=split_results[split],
            source_case_count=len(raw_rows[split]),
            source_document_count=len(source_documents),
            file_path=split_paths[split],
        )

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": args.dataset_revision,
            "revision_is_probably_immutable_sha": (
                len(args.dataset_revision) >= 7
                and all(
                    char in "0123456789abcdefABCDEF"
                    for char in args.dataset_revision
                )
            ),
        },
        "generation": {
            "seed": args.seed,
            "sampling_mode": args.sampling_mode,
            "limits": limits,
            "max_premise_chars": args.max_premise_chars,
            "allow_truncation": args.allow_truncation,
            "truncation_strategy": (
                args.truncation_strategy
                if args.max_premise_chars is not None
                else "none"
            ),
            "write_combined": args.write_combined,
            "skip_test": args.skip_test,
        },
        "labels": labels_payload,
        "splits": manifest_splits,
        "validation": validation,
        "files": {
            "labels": {
                "path": labels_path.name,
                "sha256": _sha256_file(labels_path),
                "size_bytes": labels_path.stat().st_size,
            },
            "combined": combined_metadata,
        },
    }

    manifest_path = generated_dir / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    for split in splits_to_load:
        result = split_results[split]
        print(
            f"{split.title()}: "
            f"{result.selected_case_count} cases, "
            f"{result.selected_document_count} contracts, "
            f"{result.truncated_count} truncated"
        )
    print(f"Canonical labels: {', '.join(CANONICAL_LABELS)}")
    print(f"Wrote manifest: {manifest_path}")
    if not manifest["dataset"]["revision_is_probably_immutable_sha"]:  # type: ignore[index]
        print(
            "WARNING: --dataset-revision does not look like an immutable "
            "commit SHA. Pin a revision before publishing benchmark results.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()