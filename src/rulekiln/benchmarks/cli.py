"""CLI entrypoint for reproducible RuleKiln benchmarks."""

from __future__ import annotations

import argparse
import sys
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

    # ── Refinement ablation subcommand ──────────────────────────────────────
    ablation_parser = subparsers.add_parser(
        "refinement-ablation",
        help=(
            "Compare pipeline runs with refinement loop ON vs OFF "
            "and emit refinement_ablation.json. "
            "Provide artifact directories from two completed pipeline runs."
        ),
    )
    ablation_parser.add_argument(
        "--loop-off-dir",
        type=Path,
        required=True,
        help="Artifact directory from the pipeline run with enable_refinement_loop=False.",
    )
    ablation_parser.add_argument(
        "--loop-on-dir",
        type=Path,
        required=True,
        help="Artifact directory from the pipeline run with enable_refinement_loop=True.",
    )
    ablation_parser.add_argument(
        "--output",
        type=Path,
        default=Path("refinement_ablation.json"),
        help="Output path for refinement_ablation.json (default: ./refinement_ablation.json).",
    )
    ablation_parser.add_argument(
        "--benchmark-name",
        type=str,
        default="rulekiln",
        help="Benchmark name recorded in the artifact.",
    )
    ablation_parser.add_argument(
        "--dataset",
        type=str,
        default="unknown",
        help="Dataset name recorded in the artifact.",
    )
    ablation_parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Seed used for both pipeline runs.",
    )
    ablation_parser.add_argument(
        "--strategy",
        type=str,
        default="dbscan",
        help="Strategy to compare (default: dbscan).",
    )

    return parser


def _run_ablation_subcommand(args: argparse.Namespace) -> int:
    from rulekiln.benchmarks.refinement_ablation import (
        build_refinement_ablation,
        load_eval_result_from_artifact,
        write_refinement_ablation_json,
    )

    loop_off_eval = load_eval_result_from_artifact(args.loop_off_dir, args.strategy)
    loop_on_eval = load_eval_result_from_artifact(args.loop_on_dir, args.strategy)

    if loop_off_eval is None:
        print(
            f"Warning: no eval artifact found in {args.loop_off_dir}. "
            "Proceeding with null loop_off metrics.",
            file=sys.stderr,
        )
    if loop_on_eval is None:
        print(
            f"Warning: no eval artifact found in {args.loop_on_dir}. "
            "Proceeding with null loop_on metrics.",
            file=sys.stderr,
        )

    loop_off_iterations = 0
    loop_on_iterations = _count_refinement_iterations(args.loop_on_dir)

    artifact = build_refinement_ablation(
        benchmark_name=args.benchmark_name,
        dataset=args.dataset,
        seed=args.seed,
        loop_off_eval=loop_off_eval,
        loop_on_eval=loop_on_eval,
        loop_off_iterations_run=loop_off_iterations,
        loop_on_iterations_run=loop_on_iterations,
    )

    output_path = write_refinement_ablation_json(args.output, artifact)
    print(f"Wrote refinement_ablation.json to: {output_path}")
    if artifact.loop_helped is not None:
        direction = "improved" if artifact.loop_helped else "did not improve"
        delta = artifact.delta_macro_f1 or 0.0
        print(f"Result: refinement loop {direction} macro_f1 by {delta:+.4f}")
    return 0


def _count_refinement_iterations(artifact_dir: Path) -> int:
    """Count completed refinement iterations from artifact files."""
    outputs = artifact_dir / "outputs"
    if not outputs.is_dir():
        return 0
    count = 0
    while (outputs / f"refinement_iter_{count}.json").exists():
        count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.benchmark_name == "refinement-ablation":
        return _run_ablation_subcommand(args)

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
