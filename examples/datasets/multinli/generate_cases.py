from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import IO

LABEL_NAMES = ["entailment", "neutral", "contradiction"]

TRAIN_LIMIT = 500
VAL_LIMIT = 300


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
        label_id = int(row["label"])  # type: ignore[arg-type]
        if label_id == -1:
            skipped += 1
            continue
        label = LABEL_NAMES[label_id]
        case: dict[str, object] = {
            "schema_version": "rulekiln.case.v1",
            "id": f"multinli_{target_split}_{written:06d}",
            "split": target_split,
            "task_mode": "classification",
            "input": {
                "premise": str(row["premise"]),
                "hypothesis": str(row["hypothesis"]),
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
                "source": "multinli",
                "genre": str(row.get("genre", "")),
                "label_id": label_id,
            },
            "weight": 1.0,
        }
        f.write(json.dumps(case) + "\n")
        written += 1
    if skipped:
        print(f"Skipped {skipped} cases with label_id=-1", file=sys.stderr)


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
    parser = argparse.ArgumentParser(
        description="Generate RuleKiln cases for MultiNLI"
    )
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
        ds = load_dataset("multi_nli", trust_remote_code=False)

    if args.discover_labels:
        for name in LABEL_NAMES:
            print(name)
        return

    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [dict(r) for r in ds["train"]]  # type: ignore[index]
    val_rows = [dict(r) for r in ds["validation_matched"]]  # type: ignore[index]

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

    (generated_dir / "labels.json").write_text(
        json.dumps(LABEL_NAMES, indent=2) + "\n"
    )

    print(f"Train: up to {TRAIN_LIMIT} cases")
    print(f"Validation: up to {VAL_LIMIT} cases (validation_matched split)")
    print(f"Labels: {len(LABEL_NAMES)}")


if __name__ == "__main__":
    main()
