"""CLI entrypoint for reproducible RuleKiln benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rulekiln.benchmarks.banking77 import run_banking77_benchmark
from rulekiln.benchmarks.schemas import BenchmarkProfileName, DatasetSource


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rulekiln-benchmark",
        description="Run reproducible RuleKiln benchmark workflows.",
    )
    subparsers = parser.add_subparsers(dest="benchmark_name", required=True)

    banking77_parser = subparsers.add_parser(
        "banking77",
        help="Run the BANKING77 benchmark.",
    )
    banking77_parser.add_argument(
        "--profile",
        choices=["smoke", "standard", "full"],
        default="smoke",
        help="Benchmark profile size (default: smoke).",
    )
    banking77_parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Deterministic split seed (default: 1729).",
    )
    banking77_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID. If omitted, one is generated from timestamp/profile/seed.",
    )
    banking77_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".rulekiln/benchmark_runs"),
        help="Benchmark artifact root directory.",
    )
    banking77_parser.add_argument(
        "--dataset-source",
        choices=["auto", "fixture", "download"],
        default="auto",
        help="Dataset source selection.",
    )
    banking77_parser.add_argument(
        "--fixture-path",
        type=Path,
        default=None,
        help="Optional fixture path used when dataset-source resolves to fixture.",
    )
    banking77_parser.add_argument(
        "--teacher-model",
        type=str,
        default="benchmark.teacher.not_used",
        help="Teacher model name recorded in benchmark manifest.",
    )
    banking77_parser.add_argument(
        "--student-model",
        type=str,
        default="benchmark.student.tfidf_linear_svc",
        help="Student model name recorded in benchmark manifest.",
    )
    banking77_parser.add_argument(
        "--embedding-model",
        type=str,
        default="benchmark.embedding.none",
        help="Embedding model name recorded in benchmark manifest.",
    )
    banking77_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations used for deterministic confidence intervals.",
    )
    banking77_parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Bootstrap seed for deterministic confidence intervals.",
    )
    banking77_parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap confidence interval computation.",
    )
    banking77_parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update examples/datasets/banking77/README.md snapshot block with this run.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.benchmark_name != "banking77":
        parser.error(f"Unsupported benchmark: {args.benchmark_name}")

    result = run_banking77_benchmark(
        profile=cast(BenchmarkProfileName, args.profile),
        seed=args.seed,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        dataset_source=cast(DatasetSource, args.dataset_source),
        fixture_path=args.fixture_path,
        update_readme=args.update_readme,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        embedding_model=args.embedding_model,
        bootstrap_enabled=not args.no_bootstrap,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )

    print(f"Completed benchmark run: {result.run_id}")
    print(f"Run directory: {result.run_root}")
    print(f"Summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
