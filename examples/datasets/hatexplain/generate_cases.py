from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import IO

LABEL_NAMES = ["hate", "offensive", "normal"]
LABEL_MAP = {0: "hate", 1: "offensive", 2: "normal"}

TRAIN_LIMIT = 500
VAL_LIMIT = 300


def majority_label(annotators: dict[str, object]) -> str | None:
    votes: list[int] = list(annotators["label"])  # type: ignore[index]
    majority = max(set(votes), key=votes.count)
    count = votes.count(majority)
    # Require strict majority (>= 2 of 3 annotators agreeing)
    if count < 2:
        return None
    return LABEL_MAP[majority]


def _write_cases_to_file(
    rows: list[dict[str, object]],
    target_split: str,
    f: IO[str],
    limit: int,
) -> None:
    written = 0
    skipped = 0
    for row in rows:
        if written >= limit:
            break
        annotators = row["annotators"]  # type: ignore[index]
        label = majority_label(annotators)  # type: ignore[arg-type]
        if label is None:
            skipped += 1
            print(
                f"WARNING: skipping post_id={row['post_id']} — no majority label",
                file=sys.stderr,
            )
            continue
        votes: list[int] = list(annotators["label"])  # type: ignore[index]
        vote_dist = {LABEL_MAP[k]: votes.count(k) for k in set(votes)}
        tokens: list[str] = list(row["post_tokens"])  # type: ignore[index]
        post_text = " ".join(tokens)
        case: dict[str, object] = {
            "schema_version": "rulekiln.case.v1",
            "id": f"hatexplain_{target_split}_{written:06d}",
            "split": target_split,
            "task_mode": "classification",
            "input": {"post": post_text},
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
                "source": "hatexplain",
                "post_id": str(row["post_id"]),
                "vote_distribution": vote_dist,
            },
            "weight": 1.0,
        }
        f.write(json.dumps(case) + "\n")
        written += 1
    if skipped:
        print(f"Skipped {skipped} cases with no majority label", file=sys.stderr)


def write_rulekiln_cases(
    rows: list[dict[str, object]],
    *,
    target_split: str,
    out_path: Path,
    limit: int,
    mode: str = "w",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode) as f:
        _write_cases_to_file(rows, target_split, f, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RuleKiln cases for HateXplain")
    parser.add_argument(
        "--discover-labels",
        action="store_true",
        help="Print unique labels from first 100 rows and exit (no file writes)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print("datasets not installed; run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset("hatexplain", trust_remote_code=False)

    if args.discover_labels:
        rows = [dict(r) for r in ds["train"]][:100]  # type: ignore[index]
        seen: set[str] = set()
        for row in rows:
            ann = row["annotators"]
            for v in ann["label"]:
                seen.add(LABEL_MAP[int(v)])
        for label in sorted(seen):
            print(label)
        return

    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [dict(r) for r in ds["train"]]  # type: ignore[index]
    val_rows = [dict(r) for r in ds["validation"]]  # type: ignore[index]

    write_rulekiln_cases(
        train_rows,
        target_split="train",
        out_path=generated_dir / "cases.train.jsonl",
        limit=TRAIN_LIMIT,
    )
    write_rulekiln_cases(
        val_rows,
        target_split="validation",
        out_path=generated_dir / "cases.validation.jsonl",
        limit=VAL_LIMIT,
    )

    combined_path = generated_dir / "cases.jsonl"
    write_rulekiln_cases(
        train_rows,
        target_split="train",
        out_path=combined_path,
        limit=TRAIN_LIMIT,
        mode="w",
    )
    write_rulekiln_cases(
        val_rows,
        target_split="validation",
        out_path=combined_path,
        limit=VAL_LIMIT,
        mode="a",
    )

    (generated_dir / "labels.json").write_text(json.dumps(LABEL_NAMES, indent=2) + "\n")

    print(f"Train: up to {TRAIN_LIMIT} cases (some may be skipped)")
    print(f"Validation: up to {VAL_LIMIT} cases (some may be skipped)")
    print(f"Labels: {len(LABEL_NAMES)}")


if __name__ == "__main__":
    main()
