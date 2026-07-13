"""Unit tests for the offline clustering-sweep benchmark."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.benchmarks.clustering_sweep import (
    RuleRecord,
    build_variant_texts,
    rank_eligible_arms,
    run_clustering_sweep,
    write_clustering_sweep_json,
)
from rulekiln.db.models import Base, Case, DistillationJob, MicroRule
from rulekiln.providers.contracts import ProviderConfig
from rulekiln.providers.embedding.fake import FakeEmbeddingClient
from rulekiln.schemas.pipeline import ClusteringSweepArtifact, EligibilityThresholds

_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


def _fake_config() -> ProviderConfig:
    return ProviderConfig(profile_name="fake_embed", provider="fake", model="fake-embedding")


def _records() -> list[RuleRecord]:
    return [
        RuleRecord(
            rule_id="r1",
            topic="Permission language",
            condition="contract grants explicit permission",
            expected_outcome="entailment",
            label="entailment",
            hypothesis="Receiving party may share information",
        ),
        RuleRecord(
            rule_id="r2",
            topic="Prohibition language",
            condition="contract explicitly forbids sharing",
            expected_outcome="contradiction",
            label="contradiction",
            hypothesis=None,
        ),
    ]


def test_build_variant_texts_rule_only_matches_pipeline_format() -> None:
    texts = build_variant_texts(_records(), "rule_only")
    assert texts[0] == (
        "Permission language: contract grants explicit permission → entailment"
    )
    assert "[label=" not in texts[0]


def test_build_variant_texts_label_plus_rule() -> None:
    texts = build_variant_texts(_records(), "label_plus_rule")
    assert texts[0].startswith("[label=entailment] Permission language:")
    assert texts[1].startswith("[label=contradiction] Prohibition language:")


def test_build_variant_texts_hypothesis_degrades_per_rule() -> None:
    texts = build_variant_texts(_records(), "label_plus_hypothesis_plus_rule")
    assert texts[0].startswith(
        "[label=entailment] [hypothesis=Receiving party may share information] "
    )
    # r2 has no hypothesis: degrades to the label_plus_rule form.
    assert texts[1].startswith("[label=contradiction] Prohibition language:")
    assert "[hypothesis=" not in texts[1]


async def _seed_job(session: AsyncSession, job_id: str, rule_count: int = 20) -> None:
    labels = ["entailment", "contradiction", "not_mentioned"]
    session.add(
        DistillationJob(
            id=job_id,
            task_id="contractnli",
            task_name="ContractNLI",
            task_mode="classification",
            status="completed",
            stage=None,
            request_json={},
        )
    )
    for i in range(rule_count):
        label = labels[i % len(labels)]
        case_id = f"case-{i}"
        session.add(
            Case(
                id=case_id,
                job_id=job_id,
                task_mode="classification",
                split="train",
                input_json={
                    "premise": f"Contract clause {i}",
                    "hypothesis": f"Hypothesis type {i % 4}",
                },
                expected_json={"label": label},
                evaluation_json={},
                weight=1.0,
            )
        )
        session.add(
            MicroRule(
                id=f"rule-{i}",
                job_id=job_id,
                case_id=case_id,
                topic=f"Topic {i % 5}",
                condition=f"condition variant {i}",
                expected_outcome=label,
            )
        )
    await session.commit()


async def test_run_clustering_sweep_end_to_end(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await _seed_job(db_session, "job-sweep")
    artifact = await run_clustering_sweep(
        session=db_session,
        embedding_client=FakeEmbeddingClient(),
        embedding_config=_fake_config(),
        job_id="job-sweep",
        text_variants=["rule_only", "label_plus_hypothesis_plus_rule"],
        thresholds=EligibilityThresholds(),
    )

    assert artifact.rule_count == 20
    assert artifact.label_counts == {"entailment": 7, "contradiction": 7, "not_mentioned": 6}
    assert {setup.text_variant for setup in artifact.embedding_setups} == {
        "rule_only",
        "label_plus_hypothesis_plus_rule",
    }
    assert all(setup.dimension == 384 for setup in artifact.embedding_setups)
    assert {report.text_variant for report in artifact.k_distance_reports} == {
        "rule_only",
        "label_plus_hypothesis_plus_rule",
    }

    dbscan_arms = [arm for arm in artifact.arms if arm.algorithm == "dbscan"]
    hdbscan_arms = [arm for arm in artifact.arms if arm.algorithm == "hdbscan"]
    # HDBSCAN grid is fixed: 4 sizes x 4 min_samples x 2 methods x 3 epsilons per variant.
    assert len(hdbscan_arms) == 2 * 96
    # DBSCAN arm count depends on k-distance dedup, but every arm carries provenance.
    assert dbscan_arms
    assert all(arm.eps_source_percentile is not None for arm in dbscan_arms)
    assert all(arm.dbscan_params is not None for arm in dbscan_arms)
    assert all(arm.hdbscan_params is not None for arm in hdbscan_arms)

    output_path = write_clustering_sweep_json(
        tmp_path / "sweeps" / "clustering_sweep.json", artifact
    )
    reloaded = ClusteringSweepArtifact.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert reloaded == artifact

    ranked = rank_eligible_arms(artifact)
    assert all(arm.eligibility.eligible for arm in ranked)


async def test_run_clustering_sweep_skips_hypothesis_variant_without_hypotheses(
    db_session: AsyncSession,
) -> None:
    await _seed_job(db_session, "job-nohyp")
    # Strip hypotheses from every case.
    await db_session.execute(update(Case).values(input_json={"premise": "clause"}))
    await db_session.commit()

    artifact = await run_clustering_sweep(
        session=db_session,
        embedding_client=FakeEmbeddingClient(),
        embedding_config=_fake_config(),
        job_id="job-nohyp",
        text_variants=["label_plus_hypothesis_plus_rule"],
        thresholds=EligibilityThresholds(),
    )
    assert artifact.arms == []
    assert any("Skipped variant" in note for note in artifact.notes)


async def test_run_clustering_sweep_requires_rules(db_session: AsyncSession) -> None:
    session = db_session
    session.add(
        DistillationJob(
            id="job-empty",
            task_id="t",
            task_name="T",
            task_mode="classification",
            status="completed",
            stage=None,
            request_json={},
        )
    )
    await session.commit()
    with pytest.raises(ValueError, match="at least 2"):
        await run_clustering_sweep(
            session=session,
            embedding_client=FakeEmbeddingClient(),
            embedding_config=_fake_config(),
            job_id="job-empty",
            text_variants=["rule_only"],
            thresholds=EligibilityThresholds(),
        )


def test_cli_parser_accepts_clustering_sweep() -> None:
    from rulekiln.benchmarks.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "clustering-sweep",
            "--job-id",
            "job-1",
            "--provider-profile",
            "local_mxbai_embed",
            "--embedding-model",
            "mxbai-embed-large-v1.Q6_K.gguf",
            "--variants",
            "rule_only",
            "--max-noise-pct",
            "50",
        ]
    )
    assert args.benchmark_name == "clustering-sweep"
    assert args.variants == "rule_only"
    assert args.max_noise_pct == 50.0


def test_cli_rejects_unknown_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    from rulekiln.benchmarks.cli import _run_clustering_sweep_subcommand

    del monkeypatch
    args = argparse.Namespace(
        job_id="job-1",
        provider_profile="p",
        embedding_model="m",
        variants="rule_only,bogus_variant",
        output=None,
        mlflow=False,
        min_cluster_count=8,
        max_cluster_count=40,
        max_noise_pct=60.0,
        max_largest_cluster_pct=40.0,
    )
    assert _run_clustering_sweep_subcommand(args) == 1
