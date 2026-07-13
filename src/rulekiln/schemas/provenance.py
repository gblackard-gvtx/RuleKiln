"""Clustering provenance schemas (Workstream A).

Typed, JSON-serializable, versioned models that capture *exactly* what clustering
configuration was requested, what was used at runtime, what embeddings/text were
clustered, and what clusters were produced. Shared by the runtime capture path,
persistence (writer), provenance validation, and the UI so there is a single typed
representation of provenance rather than parallel structures.

This module is intentionally a leaf: it imports nothing from ``rulekiln.schemas.pipeline``
so that ``pipeline`` can embed provenance rows without an import cycle.
"""

from typing import Literal

from pydantic import BaseModel, Field

type ClusteringAlgorithm = Literal["dbscan", "hdbscan"]
type ClusteringTextVariant = Literal[
    "rule_only",
    "label_plus_rule",
    "label_plus_hypothesis_plus_rule",
]
type ClusteringLibrary = Literal["sklearn", "hdbscan"]


class ClusteringParameters(BaseModel):
    """Requested or effective clustering parameters.

    The same shape is used for both requested and effective parameters so a reader can
    diff them field-by-field. Irrelevant fields for a given algorithm stay explicitly
    ``None`` (for example ``eps`` for HDBSCAN) rather than being omitted.
    """

    eps: float | None = None
    min_samples: int | None = None
    min_cluster_size: int | None = None
    cluster_selection_method: str | None = None
    cluster_selection_epsilon: float | None = None
    metric: str = "cosine"


class EmbeddingProvenance(BaseModel):
    """What was embedded and how, for the clustered text."""

    provider_profile: str
    provider: str
    model: str
    dimension: int
    dtype: str | None = None
    quantization: str | None = None
    pooling: str | None = None
    query_prefix: str | None = None
    vectors_prenormalized: bool = True
    normalized_before_distance: bool = True
    distance_metric: Literal["cosine"] = "cosine"


class ClusteringInput(BaseModel):
    """Summary of the ordered inputs that were clustered."""

    rule_count: int = 0
    duplicate_rule_count: int = 0
    duplicate_rule_rate: float = 0.0
    label_counts: dict[str, int] = Field(default_factory=dict)
    text_variant: ClusteringTextVariant = "rule_only"


class ClusteringOutput(BaseModel):
    """Cluster-health summary for the produced clustering."""

    cluster_count_excluding_noise: int = 0
    noise_count: int = 0
    noise_pct: float = 0.0
    largest_cluster_size: int = 0
    largest_cluster_pct: float = 0.0
    median_cluster_size: float = 0.0
    small_cluster_count: int = 0
    synthesized_rule_count: int = 0


class ClusteringRuntime(BaseModel):
    """Library and timing information captured during execution."""

    library: ClusteringLibrary | None = None
    library_version: str | None = None
    python_version: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None


class ClusteringSource(BaseModel):
    """Where the requested configuration and inputs came from."""

    task_config_path: str | None = None
    job_task_snapshot_path: str | None = None
    strategy: str


class ClusteringHashes(BaseModel):
    """Canonical hashes tying this provenance record to its persisted artifacts."""

    input_text_hash: str | None = None
    embedding_matrix_hash: str | None = None
    cluster_assignment_hash: str | None = None
    cluster_summary_hash: str | None = None
    synthesized_rules_hash: str | None = None


class ClusteringProvenance(BaseModel):
    """Complete clustering provenance for one algorithm on one job.

    ``job_id`` is not in the abstract schema example but is persisted so provenance can
    be validated as belonging to the correct job (cross-job artifact detection).
    """

    schema_version: Literal["rulekiln.clustering_provenance.v1"] = (
        "rulekiln.clustering_provenance.v1"
    )
    job_id: str | None = None
    algorithm: ClusteringAlgorithm
    text_variant: ClusteringTextVariant = "rule_only"
    requested_parameters: ClusteringParameters = Field(default_factory=ClusteringParameters)
    effective_parameters: ClusteringParameters = Field(default_factory=ClusteringParameters)
    embedding: EmbeddingProvenance
    input: ClusteringInput = Field(default_factory=ClusteringInput)
    output: ClusteringOutput = Field(default_factory=ClusteringOutput)
    runtime: ClusteringRuntime = Field(default_factory=ClusteringRuntime)
    source: ClusteringSource
    hashes: ClusteringHashes = Field(default_factory=ClusteringHashes)


class ClusterAssignmentRecord(BaseModel):
    """One rule's cluster assignment, deterministic and debuggable without vectors."""

    rule_id: str
    source_case_id: str | None = None
    cluster_id: int
    is_noise: bool
    label: str | None = None
    hypothesis_id: str | None = None
    text_hash: str
    embedding_row_index: int


class ClusterSummaryEntry(BaseModel):
    """Aggregate summary for a single cluster."""

    cluster_id: int
    size: int
    label_distribution: dict[str, int] = Field(default_factory=dict)
    hypothesis_distribution: dict[str, int] = Field(default_factory=dict)
    representative_rule_ids: list[str] = Field(default_factory=list)
    medoid_index: int | None = None
    synthesized_rule_id: str | None = None


class ClusterSummary(BaseModel):
    """Cluster summary artifact: per-cluster stats plus noise statistics."""

    schema_version: Literal["rulekiln.cluster_summary.v1"] = "rulekiln.cluster_summary.v1"
    job_id: str | None = None
    algorithm: ClusteringAlgorithm
    strategy: str
    cluster_count: int = 0
    noise_count: int = 0
    noise_pct: float = 0.0
    clusters: list[ClusterSummaryEntry] = Field(default_factory=list)


class StrategyProvenanceRow(BaseModel):
    """Per-strategy provenance + hash references embedded in strategy_comparison.json.

    Clustering fields are ``None`` for non-clustering strategies; prompt/prediction/
    evaluation hashes are populated for every strategy.
    """

    strategy: str
    family: str
    clustering_provenance_ref: str | None = None
    clustering_config_hash: str | None = None
    cluster_assignment_hash: str | None = None
    synthesized_rules_hash: str | None = None
    compiled_prompt_hash: str | None = None
    prediction_hash: str | None = None
    evaluation_hash: str | None = None
