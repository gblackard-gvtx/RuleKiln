"""Schemas for reproducible benchmark runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rulekiln.schemas.pipeline import EvalResult, PairedComparisonSummary

BenchmarkProfileName = Literal["smoke", "standard", "full"]
DatasetSource = Literal["auto", "fixture", "download"]


class BenchmarkProfileConfig(BaseModel):
    """Profile counts for benchmark data splits."""

    train_cases: int | Literal["all"]
    validation_cases: int | Literal["deterministic_from_train"]
    test_cases: int | Literal["all"]


class CostSummary(BaseModel):
    """Token and cost summary for a benchmark run."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_total_cost_usd: float = 0.0
    teacher_cost_usd: float = 0.0
    student_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    has_estimated_usage: bool = False
    total_model_calls: int = 0


class BenchmarkManifest(BaseModel):
    """Provenance manifest for a benchmark run."""

    benchmark_name: str
    run_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),  # noqa: UP017
    )
    git_commit: str
    rulekiln_version: str
    python_version: str
    dataset_name: str
    dataset_revision: str | None = None
    seed: int
    teacher_model: str
    student_model: str
    embedding_model: str
    strategy_names: list[str] = Field(default_factory=list)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    case_counts: dict[str, int] = Field(default_factory=dict)
    cost_summary: CostSummary = Field(default_factory=CostSummary)


class DatasetManifest(BaseModel):
    """Dataset and split provenance for a benchmark run."""

    dataset_name: str
    dataset_revision: str | None = None
    source: str
    profile: BenchmarkProfileName
    seed: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),  # noqa: UP017
    )
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_id_files: dict[str, str] = Field(default_factory=dict)


class BenchmarkStrategyComparison(BaseModel):
    """Baseline vs RuleKiln benchmark comparison."""

    primary_metric: str
    baseline_eval: EvalResult
    rulekiln_eval: EvalResult
    baseline_score: float
    rulekiln_score: float
    delta_vs_baseline: float
    selected_strategy_id: str | None = None
    selected_strategy_family: str | None = None
    best_distilled_strategy_id: str | None = None
    best_baseline_strategy_id: str | None = None
    best_by_family: dict[str, str] = Field(default_factory=dict)
    paired_comparison: PairedComparisonSummary | None = None
    selected_strategy: str
    selection_reason: str


class Banking77Example(BaseModel):
    """Normalized BANKING77 row used in deterministic splitting and evaluation."""

    source_id: str
    text: str
    label: str


class Banking77SplitResult(BaseModel):
    """Split output with deterministic examples per split."""

    train_examples: list[Banking77Example] = Field(default_factory=list)
    validation_examples: list[Banking77Example] = Field(default_factory=list)
    test_examples: list[Banking77Example] = Field(default_factory=list)


class BenchmarkRunResult(BaseModel):
    """Return object for completed benchmark runs."""

    run_id: str
    run_root: Path
    benchmark_manifest_path: Path
    dataset_manifest_path: Path
    summary_path: Path
