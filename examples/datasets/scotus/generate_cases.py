from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import IO

TEXT_TRUNCATE = 2000

# From lex_glue/scotus. Verify against dataset features["label"].names.
LABEL_NAMES = [
    "criminal_procedure",
    "civil_rights",
    "first_amendment",
    "due_process",
    "privacy",
    "attorneys",
    "unions",
    "economic_activity",
    "judicial_power",
    "federalism",
    "interstate_relations",
    "federal_taxation",
    "miscellaneous",
    "private_action",
]

TRAIN_LIMIT = 500
VAL_LIMIT = 300


def _write_cases_to_file(
    rows: list[dict[str, object]],
    label_names: list[str],
    target_split: str,
    f: IO[str],
    limit: int,
) -> None:
    for i, row in enumerate(rows[:limit]):
        label_id = int(row["label"])  # type: ignore[arg-type]
        if label_id < 0 or label_id >= len(label_names):
            continue
        label = label_names[label_id]
        text = str(row["text"])
        truncated = len(text) > TEXT_TRUNCATE
        if truncated:
            text = text[:TEXT_TRUNCATE]
        case: dict[str, object] = {
            "schema_version": "rulekiln.case.v1",
            "id": f"scotus_{target_split}_{i:06d}",
            "split": target_split,
            "task_mode": "classification",
            "input": {"excerpt": text},
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
                "source": "scotus",
                "label_id": label_id,
                "truncated": truncated,
            },
            "weight": 1.0,
        }
        f.write(json.dumps(case) + "\n")


def write_rulekiln_cases(
    rows: list[dict[str, object]],
    label_names: list[str],
    *,
    target_split: str,
    out_path: Path,
    limit: int,
    mode: str = "w",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode) as f:
        _write_cases_to_file(rows, label_names, target_split, f, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RuleKiln cases for SCOTUS")
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
        ds = load_dataset("lex_glue", "scotus", trust_remote_code=False)

    label_names: list[str] = ds["train"].features["label"].names  # type: ignore[index]

    if args.discover_labels:
        for name in label_names[:100]:
            print(name)
        return

    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [dict(r) for r in ds["train"]]  # type: ignore[index]
    val_rows = [dict(r) for r in ds["validation"]]  # type: ignore[index]

    write_rulekiln_cases(
        train_rows,
        label_names,
        target_split="train",
        out_path=generated_dir / "cases.train.jsonl",
        limit=TRAIN_LIMIT,
    )
    write_rulekiln_cases(
        val_rows,
        label_names,
        target_split="validation",
        out_path=generated_dir / "cases.validation.jsonl",
        limit=VAL_LIMIT,
    )

    combined_path = generated_dir / "cases.jsonl"
    write_rulekiln_cases(
        train_rows,
        label_names,
        target_split="train",
        out_path=combined_path,
        limit=TRAIN_LIMIT,
        mode="w",
    )
    write_rulekiln_cases(
        val_rows,
        label_names,
        target_split="validation",
        out_path=combined_path,
        limit=VAL_LIMIT,
        mode="a",
    )

    (generated_dir / "labels.json").write_text(
        json.dumps(label_names, indent=2) + "\n"
    )

    print(f"Train: {min(len(train_rows), TRAIN_LIMIT)} cases")
    print(f"Validation: {min(len(val_rows), VAL_LIMIT)} cases")
    print(f"Labels: {len(label_names)}")


if __name__ == "__main__":
    main()
