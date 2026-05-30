"""BANKING77 benchmark runner with deterministic split generation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Literal, cast

import yaml
from sklearn.model_selection import train_test_split

from rulekiln.benchmarks.baselines import predict_majority_label, predict_tfidf_linear_svc
from rulekiln.benchmarks.reporting import (
    update_readme_snapshot,
    write_benchmark_manifest,
    write_confusion_matrix_csv,
    write_dataset_manifest,
    write_eval,
    write_per_label_metrics_csv,
    write_strategy_comparison,
    write_summary_markdown,
)
from rulekiln.benchmarks.schemas import (
    Banking77Example,
    Banking77SplitResult,
    BenchmarkManifest,
    BenchmarkProfileConfig,
    BenchmarkProfileName,
    BenchmarkRunResult,
    BenchmarkStrategyComparison,
    CostSummary,
    DatasetManifest,
    DatasetSource,
    PerLabelMetricRow,
)
from rulekiln.pipeline.prompt_compiler import compile_baseline_prompt
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import (
    EvaluationAssertion,
    EvaluationSpec,
    RuleKilnCase,
    RuleKilnTask,
)

_DEFAULT_SEED = 1729
_BANKING77_DATASET_NAME = "PolyAI/banking77"
_BANKING77_DATASET_REVISION = "refs/pr/7"
_FULL_VALIDATION_FRACTION = 0.20

_PROFILE_CONFIGS: dict[BenchmarkProfileName, BenchmarkProfileConfig] = {
    "smoke": BenchmarkProfileConfig(train_cases=25, validation_cases=25, test_cases=25),
    "standard": BenchmarkProfileConfig(train_cases=500, validation_cases=300, test_cases=300),
    "full": BenchmarkProfileConfig(
        train_cases="all",
        validation_cases="deterministic_from_train",
        test_cases="all",
    ),
}


def run_banking77_benchmark(
    *,
    profile: BenchmarkProfileName,
    seed: int = _DEFAULT_SEED,
    run_id: str | None = None,
    artifact_root: Path = Path(".rulekiln/benchmark_runs"),
    dataset_source: DatasetSource = "auto",
    fixture_path: Path | None = None,
    update_readme: bool = False,
    teacher_model: str = "benchmark.teacher.not_used",
    student_model: str = "benchmark.student.tfidf_linear_svc",
    embedding_model: str = "benchmark.embedding.none",
) -> BenchmarkRunResult:
    """Run the BANKING77 benchmark with deterministic data splits and artifacts."""
    repo_root = _repo_root()
    task_path = repo_root / "examples/datasets/banking77/task.yaml"
    task = _load_task(task_path)
    label_names = _extract_label_names(task)

    resolved_fixture_path = (
        fixture_path
        if fixture_path is not None
        else repo_root / "examples/datasets/banking77/cases.sample.jsonl"
    )
    resolved_source = _resolve_dataset_source(
        profile=profile,
        dataset_source=dataset_source,
        fixture_path=resolved_fixture_path,
    )

    profile_config = _PROFILE_CONFIGS[profile]

    if resolved_source == "fixture":
        pool_examples = _load_fixture_examples(resolved_fixture_path)
        split_result = build_fixture_splits(
            examples=pool_examples,
            profile_config=profile_config,
            profile=profile,
            seed=seed,
        )
        dataset_revision: str | None = None
    else:
        cache_dir = repo_root / ".rulekiln/benchmark_cache/banking77"
        train_examples, test_examples = _load_download_examples(
            cache_dir=cache_dir,
            label_names=label_names,
        )
        split_result = build_download_splits(
            train_examples=train_examples,
            test_examples=test_examples,
            profile_config=profile_config,
            profile=profile,
            seed=seed,
        )
        dataset_revision = _BANKING77_DATASET_REVISION

    resolved_run_id = run_id or _default_run_id(profile, seed)
    run_root = artifact_root / "banking77" / resolved_run_id
    run_root.mkdir(parents=True, exist_ok=True)

    splits_dir = run_root / "splits"
    train_ids_path = _write_split_ids(splits_dir / "train_ids.txt", split_result.train_examples)
    validation_ids_path = _write_split_ids(
        splits_dir / "validation_ids.txt", split_result.validation_examples
    )
    test_ids_path = _write_split_ids(splits_dir / "test_ids.txt", split_result.test_examples)

    baseline_predictions = predict_majority_label(
        split_result.train_examples,
        split_result.test_examples,
    )
    rulekiln_predictions = predict_tfidf_linear_svc(
        split_result.train_examples,
        split_result.test_examples,
        seed=seed,
    )

    baseline_eval, baseline_support = _evaluate_predictions(
        examples=split_result.test_examples,
        predictions=baseline_predictions,
        strategy="baseline",
        model_name="majority_label",
    )
    rulekiln_eval, rulekiln_support = _evaluate_predictions(
        examples=split_result.test_examples,
        predictions=rulekiln_predictions,
        strategy="rulekiln",
        model_name="tfidf_linear_svc",
    )

    primary_metric = "macro_f1"
    baseline_score = baseline_eval.macro_f1 or 0.0
    rulekiln_score = rulekiln_eval.macro_f1 or 0.0
    selected_strategy = "rulekiln" if rulekiln_score >= baseline_score else "baseline"
    comparison = BenchmarkStrategyComparison(
        primary_metric=primary_metric,
        baseline_eval=baseline_eval,
        rulekiln_eval=rulekiln_eval,
        baseline_score=baseline_score,
        rulekiln_score=rulekiln_score,
        delta_vs_baseline=rulekiln_score - baseline_score,
        selected_strategy=selected_strategy,
        selection_reason=(
            "RuleKiln selected due to higher or equal macro_f1."
            if selected_strategy == "rulekiln"
            else "Baseline selected due to higher macro_f1."
        ),
    )

    baseline_prompt = compile_baseline_prompt(task)
    rulekiln_prompt_descriptor = (
        baseline_prompt
        + "\n\n# Benchmark Strategy\n"
        + "Deterministic TF-IDF + LinearSVC classifier trained on train split."
    )

    case_counts = {
        "train": len(split_result.train_examples),
        "validation": len(split_result.validation_examples),
        "test": len(split_result.test_examples),
    }

    dataset_manifest = DatasetManifest(
        dataset_name=_BANKING77_DATASET_NAME,
        dataset_revision=dataset_revision,
        source=resolved_source,
        profile=profile,
        seed=seed,
        split_counts=case_counts,
        split_id_files={
            "train": str(train_ids_path.relative_to(run_root)),
            "validation": str(validation_ids_path.relative_to(run_root)),
            "test": str(test_ids_path.relative_to(run_root)),
        },
    )

    benchmark_manifest = BenchmarkManifest(
        benchmark_name="banking77",
        run_id=resolved_run_id,
        created_at=datetime.now(timezone.utc),  # noqa: UP017
        git_commit=_git_commit(repo_root),
        rulekiln_version=_rulekiln_version(),
        python_version=sys.version.split()[0],
        dataset_name=_BANKING77_DATASET_NAME,
        dataset_revision=dataset_revision,
        seed=seed,
        teacher_model=teacher_model,
        student_model=student_model,
        embedding_model=embedding_model,
        strategy_names=["baseline", "rulekiln"],
        prompt_hashes={
            "baseline": _sha256(baseline_prompt),
            "rulekiln": _sha256(rulekiln_prompt_descriptor),
        },
        case_counts=case_counts,
        cost_summary=CostSummary(),
    )

    dataset_manifest_path = write_dataset_manifest(
        run_root / "dataset_manifest.json",
        dataset_manifest,
    )
    benchmark_manifest_path = write_benchmark_manifest(
        run_root / "benchmark_manifest.json", benchmark_manifest
    )

    write_eval(run_root / "baseline_eval.json", baseline_eval)
    write_eval(run_root / "rulekiln_eval.json", rulekiln_eval)
    write_strategy_comparison(run_root / "strategy_comparison.json", comparison)
    write_confusion_matrix_csv(run_root / "confusion_matrix.csv", rulekiln_eval)

    per_label_rows = _build_per_label_rows(
        baseline_eval=baseline_eval,
        baseline_support=baseline_support,
        rulekiln_eval=rulekiln_eval,
        rulekiln_support=rulekiln_support,
    )
    write_per_label_metrics_csv(run_root / "per_label_metrics.csv", per_label_rows)

    summary_path = write_summary_markdown(
        run_root / "summary.md",
        benchmark_manifest,
        dataset_manifest,
        comparison,
    )

    if update_readme:
        update_readme_snapshot(
            readme_path=repo_root / "examples/datasets/banking77/README.md",
            manifest=benchmark_manifest,
            dataset_manifest=dataset_manifest,
            comparison=comparison,
        )

    return BenchmarkRunResult(
        run_id=resolved_run_id,
        run_root=run_root,
        benchmark_manifest_path=benchmark_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
        summary_path=summary_path,
    )


def build_fixture_splits(
    *,
    examples: list[Banking77Example],
    profile_config: BenchmarkProfileConfig,
    profile: BenchmarkProfileName,
    seed: int,
) -> Banking77SplitResult:
    """Build deterministic splits from a single fixture pool."""
    if profile == "full":
        raise ValueError("Profile 'full' requires --dataset-source download.")

    validation_count = _as_int(profile_config.validation_cases)
    train_count = _as_int(profile_config.train_cases)
    test_count = _as_int(profile_config.test_cases)

    validation_examples = _sample_examples(examples, validation_count, seed)
    validation_ids = {item.source_id for item in validation_examples}

    remaining_after_validation = [
        item for item in _canonical_sort(examples) if item.source_id not in validation_ids
    ]
    train_examples = _sample_examples(remaining_after_validation, train_count, seed + 1)
    train_ids = {item.source_id for item in train_examples}

    remaining_after_train = [
        item for item in remaining_after_validation if item.source_id not in train_ids
    ]
    test_examples = _sample_examples(remaining_after_train, test_count, seed + 2)

    return Banking77SplitResult(
        train_examples=_canonical_sort(train_examples),
        validation_examples=_canonical_sort(validation_examples),
        test_examples=_canonical_sort(test_examples),
    )


def build_download_splits(
    *,
    train_examples: list[Banking77Example],
    test_examples: list[Banking77Example],
    profile_config: BenchmarkProfileConfig,
    profile: BenchmarkProfileName,
    seed: int,
) -> Banking77SplitResult:
    """Build deterministic splits from BANKING77 train/test source pools."""
    sorted_train = _canonical_sort(train_examples)
    sorted_test = _canonical_sort(test_examples)

    if profile == "full":
        validation_count = _full_validation_count(sorted_train)
        validation_examples = _sample_examples(sorted_train, validation_count, seed)
        validation_ids = {item.source_id for item in validation_examples}
        remaining_train = [item for item in sorted_train if item.source_id not in validation_ids]
        return Banking77SplitResult(
            train_examples=_canonical_sort(remaining_train),
            validation_examples=_canonical_sort(validation_examples),
            test_examples=sorted_test,
        )

    validation_count = _as_int(profile_config.validation_cases)
    train_count = _as_int(profile_config.train_cases)
    test_count = _as_int(profile_config.test_cases)

    validation_examples = _sample_examples(sorted_train, validation_count, seed)
    validation_ids = {item.source_id for item in validation_examples}

    remaining_train = [item for item in sorted_train if item.source_id not in validation_ids]
    selected_train = _sample_examples(remaining_train, train_count, seed + 1)
    selected_test = _sample_examples(sorted_test, test_count, seed + 2)

    return Banking77SplitResult(
        train_examples=_canonical_sort(selected_train),
        validation_examples=_canonical_sort(validation_examples),
        test_examples=_canonical_sort(selected_test),
    )


def _evaluate_predictions(
    *,
    examples: list[Banking77Example],
    predictions: list[str],
    strategy: str,
    model_name: str,
) -> tuple[EvalResult, dict[str, int]]:
    if len(examples) != len(predictions):
        raise ValueError("Number of predictions must equal number of examples.")

    expected_labels = [example.label for example in examples]
    labels = sorted(set(expected_labels) | set(predictions))
    label_indices = {label: index for index, label in enumerate(labels)}

    matrix_rows = [[0 for _ in labels] for _ in labels]
    for expected_label, predicted_label in zip(expected_labels, predictions, strict=True):
        expected_index = label_indices[expected_label]
        predicted_index = label_indices[predicted_label]
        matrix_rows[expected_index][predicted_index] += 1

    correct_predictions = sum(
        matrix_rows[index][index] for index in range(len(labels))
    )
    accuracy = correct_predictions / len(examples) if examples else 0.0

    per_outcome_precision: dict[str, float] = {}
    per_outcome_recall: dict[str, float] = {}
    support_map: dict[str, int] = {}
    f1_scores: list[float] = []
    for label_index, label in enumerate(labels):
        true_positives = matrix_rows[label_index][label_index]
        predicted_total = sum(matrix_rows[row][label_index] for row in range(len(labels)))
        actual_total = sum(matrix_rows[label_index][column] for column in range(len(labels)))

        precision = true_positives / predicted_total if predicted_total > 0 else 0.0
        recall = true_positives / actual_total if actual_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_outcome_precision[label] = precision
        per_outcome_recall[label] = recall
        support_map[label] = actual_total
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    confusion_payload: dict[str, dict[str, int]] = {}
    for expected_index, expected_label in enumerate(labels):
        row: dict[str, int] = {}
        for predicted_index, predicted_label in enumerate(labels):
            count = matrix_rows[expected_index][predicted_index]
            if count > 0:
                row[predicted_label] = count
        if row:
            confusion_payload[expected_label] = row

    case_results: list[CaseEvalResult] = []
    for example, prediction in zip(examples, predictions, strict=True):
        passed = prediction == example.label
        case_results.append(
            CaseEvalResult(
                case_id=example.source_id,
                score=1.0 if passed else 0.0,
                passed=passed,
                malformed=False,
                assertion_scores={"assertion_0": 1.0 if passed else 0.0},
                actual_output={"label": prediction},
            )
        )

    eval_result = EvalResult(
        strategy=strategy,
        model=model_name,
        split="test",
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_case_score=accuracy,
        malformed_output_rate=0.0,
        per_outcome_precision=per_outcome_precision,
        per_outcome_recall=per_outcome_recall,
        confusion_matrix=confusion_payload,
        case_results=case_results,
    )

    return eval_result, support_map


def _build_per_label_rows(
    *,
    baseline_eval: EvalResult,
    baseline_support: dict[str, int],
    rulekiln_eval: EvalResult,
    rulekiln_support: dict[str, int],
) -> list[PerLabelMetricRow]:
    labels = sorted(
        set(baseline_eval.per_outcome_precision.keys())
        | set(rulekiln_eval.per_outcome_precision.keys())
        | set(baseline_support.keys())
        | set(rulekiln_support.keys())
    )

    rows: list[PerLabelMetricRow] = []
    for label in labels:
        rows.append(
            PerLabelMetricRow(
                label=label,
                precision=baseline_eval.per_outcome_precision.get(label, 0.0),
                recall=baseline_eval.per_outcome_recall.get(label, 0.0),
                support=baseline_support.get(label, 0),
                strategy="baseline",
            )
        )
        rows.append(
            PerLabelMetricRow(
                label=label,
                precision=rulekiln_eval.per_outcome_precision.get(label, 0.0),
                recall=rulekiln_eval.per_outcome_recall.get(label, 0.0),
                support=rulekiln_support.get(label, 0),
                strategy="rulekiln",
            )
        )

    return rows


def _resolve_dataset_source(
    *,
    profile: BenchmarkProfileName,
    dataset_source: DatasetSource,
    fixture_path: Path,
) -> Literal["fixture", "download"]:
    if dataset_source == "fixture":
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
        return "fixture"

    if dataset_source == "download":
        return "download"

    if profile == "smoke" and fixture_path.exists():
        return "fixture"

    return "download"


def _load_task(path: Path) -> RuleKilnTask:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Task YAML is not an object: {path}")
    return RuleKilnTask.model_validate(parsed)


def _extract_label_names(task: RuleKilnTask) -> list[str]:
    properties = task.output_schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Task output_schema.properties must be an object.")

    label_schema = properties.get("label")
    if not isinstance(label_schema, dict):
        raise ValueError("Task output_schema must include properties.label.")

    enum_values = label_schema.get("enum")
    if not isinstance(enum_values, list):
        raise ValueError("Task output_schema.properties.label.enum must be a list.")

    labels: list[str] = []
    for value in enum_values:
        if not isinstance(value, str):
            raise ValueError("Label enum values must be strings.")
        labels.append(value)

    if not labels:
        raise ValueError("Label enum list must not be empty.")

    return labels


def _load_fixture_examples(path: Path) -> list[Banking77Example]:
    lines = path.read_text(encoding="utf-8").splitlines()
    examples: list[Banking77Example] = []

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Fixture row {index + 1} is not a JSON object.")

        input_obj = parsed.get("input")
        expected_obj = parsed.get("expected")
        source_id_obj = parsed.get("id")

        if not isinstance(input_obj, dict):
            raise ValueError(f"Fixture row {index + 1} missing input object.")
        utterance_obj = input_obj.get("utterance")
        if not isinstance(utterance_obj, str):
            raise ValueError(f"Fixture row {index + 1} missing input.utterance string.")

        if not isinstance(expected_obj, dict):
            raise ValueError(f"Fixture row {index + 1} missing expected object.")
        label_obj = expected_obj.get("label")
        if not isinstance(label_obj, str):
            raise ValueError(f"Fixture row {index + 1} missing expected.label string.")

        if isinstance(source_id_obj, str) and source_id_obj:
            source_id = source_id_obj
        else:
            source_id = f"banking77_fixture_{index:06d}"

        examples.append(
            Banking77Example(
                source_id=source_id,
                text=utterance_obj,
                label=label_obj,
            )
        )

    if not examples:
        raise ValueError(f"Fixture file is empty: {path}")

    return _canonical_sort(examples)


def _load_download_examples(
    *,
    cache_dir: Path,
    label_names: list[str],
) -> tuple[list[Banking77Example], list[Banking77Example]]:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Downloading BANKING77 requires 'pandas' and 'huggingface_hub'. "
            "Install them in the active environment and retry."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(
        hf_hub_download(
            repo_id=_BANKING77_DATASET_NAME,
            repo_type="dataset",
            filename="data/train-00000-of-00001.parquet",
            revision=_BANKING77_DATASET_REVISION,
            local_dir=cache_dir,
        )
    )
    test_path = Path(
        hf_hub_download(
            repo_id=_BANKING77_DATASET_NAME,
            repo_type="dataset",
            filename="data/test-00000-of-00001.parquet",
            revision=_BANKING77_DATASET_REVISION,
            local_dir=cache_dir,
        )
    )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_records = cast(
        list[dict[str, object]],
        train_df.to_dict(orient="records"),
    )
    test_records = cast(
        list[dict[str, object]],
        test_df.to_dict(orient="records"),
    )

    train_examples = _records_to_examples(train_records, "train", label_names)
    test_examples = _records_to_examples(test_records, "test", label_names)
    return _canonical_sort(train_examples), _canonical_sort(test_examples)


def _records_to_examples(
    records: list[dict[str, object]],
    source_split: Literal["train", "test"],
    label_names: list[str],
) -> list[Banking77Example]:
    examples: list[Banking77Example] = []

    for index, record in enumerate(records):
        text_obj = record.get("text")
        label_obj = record.get("label")

        if not isinstance(text_obj, str):
            raise ValueError(f"BANKING77 {source_split} row {index} has invalid text field.")

        label = _normalize_download_label(label_obj, label_names)
        examples.append(
            Banking77Example(
                source_id=f"banking77_{source_split}_{index:06d}",
                text=text_obj,
                label=label,
            )
        )

    if not examples:
        raise ValueError(f"BANKING77 {source_split} split loaded zero rows.")

    return examples


def _normalize_download_label(raw_label: object, label_names: list[str]) -> str:
    if isinstance(raw_label, str):
        return raw_label

    if isinstance(raw_label, bool):
        raise ValueError("Boolean labels are not valid BANKING77 label IDs.")

    if isinstance(raw_label, int):
        label_index = raw_label
    elif isinstance(raw_label, float) and raw_label.is_integer():
        label_index = int(raw_label)
    else:
        raise ValueError(f"Unsupported BANKING77 label value: {raw_label!r}")

    if label_index < 0 or label_index >= len(label_names):
        raise ValueError(f"Label index out of range: {label_index}")
    return label_names[label_index]


def _sample_examples(
    examples: list[Banking77Example],
    count: int,
    seed: int,
) -> list[Banking77Example]:
    if count < 0:
        raise ValueError("count must be non-negative.")

    canonical = _canonical_sort(examples)
    if count > len(canonical):
        raise ValueError(
            f"Requested {count} rows but only {len(canonical)} rows are available for sampling."
        )

    if count == 0:
        return []

    if count == len(canonical):
        return canonical

    labels = [example.label for example in canonical]
    if _can_stratify(labels=labels, count=count, total=len(canonical)):
        try:
            _, selected = train_test_split(
                canonical,
                test_size=count,
                random_state=seed,
                shuffle=True,
                stratify=labels,
            )
            return _canonical_sort(list(selected))
        except ValueError:
            pass

    randomizer = Random(seed)  # noqa: S311
    selected_indices = sorted(randomizer.sample(range(len(canonical)), k=count))
    return _canonical_sort([canonical[index] for index in selected_indices])


def _can_stratify(*, labels: list[str], count: int, total: int) -> bool:
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        return False

    if count < len(label_counts):
        return False

    if (total - count) < len(label_counts):
        return False

    return not min(label_counts.values()) < 2


def _full_validation_count(train_examples: list[Banking77Example]) -> int:
    total = len(train_examples)
    if total <= 1:
        return 0

    tentative = int(round(total * _FULL_VALIDATION_FRACTION))
    bounded = max(1, min(tentative, total - 1))
    return bounded


def _as_int(value: int | str) -> int:
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected integer count, got '{value}'.")


def _canonical_sort(examples: list[Banking77Example]) -> list[Banking77Example]:
    return sorted(
        examples,
        key=lambda item: (item.label, item.text.casefold(), item.source_id),
    )


def _write_split_ids(path: Path, examples: list[Banking77Example]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(example.source_id for example in examples)
    payload = "\n".join(ids) + "\n" if ids else ""
    path.write_text(payload, encoding="utf-8")
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit(repo_root: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown"

    try:
        result = subprocess.run(  # noqa: S603
            [git_executable, "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip() or "unknown"


def _rulekiln_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ModuleNotFoundError:
        return "unknown"

    try:
        return version("rulekiln")
    except PackageNotFoundError:
        return "0.1.0"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _default_run_id(profile: BenchmarkProfileName, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    return f"{timestamp}_{profile}_seed{seed}"


def build_rulekiln_case(
    example: Banking77Example,
    split: Literal["train", "validation", "test", "golden"],
) -> RuleKilnCase:
    """Build a RuleKilnCase object for optional downstream reuse."""
    return RuleKilnCase(
        id=example.source_id,
        split=split,
        task_mode="classification",
        input={"utterance": example.text},
        expected={"label": example.label},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(
                    type="must_equal",
                    path="$.label",
                    value=example.label,
                    weight=1.0,
                )
            ]
        ),
    )
