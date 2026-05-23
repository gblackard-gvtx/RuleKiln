"""Centralized application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type ProviderKind = Literal[
    "fake",
    "bedrock",
    "openai",
    "anthropic",
    "vertex_gemini",
    "azure_openai",
    "openai_compatible",
    "custom",
]


class ProviderProfile(BaseModel):
    """Configuration for a named provider profile."""

    provider: ProviderKind
    region: str | None = None
    base_url: str | None = None
    api_key_env_var: str | None = None
    supports_chat: bool = True
    supports_embeddings: bool = False
    timeout_seconds: int = 60
    max_retries: int = 3

    # Rate limiting
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    max_concurrency: int = 3

    # Provider-specific optional metadata
    aws_assume_role_arn: str | None = None
    azure_deployment: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None

    @model_validator(mode="after")
    def validate_profile_requirements(self) -> "ProviderProfile":
        if self.provider == "bedrock" and self.region is None:
            raise ValueError("Bedrock profiles must define region.")

        api_key_providers: set[str] = {"openai", "anthropic", "azure_openai", "openai_compatible"}
        if self.provider in api_key_providers and self.api_key_env_var is None:
            raise ValueError(f"Provider '{self.provider}' must define api_key_env_var.")

        return self


class QualityGateDefaults(BaseModel):
    """Default quality gate thresholds; may be overridden per task."""

    min_metric_delta: float = 0.0
    max_regression_rate: float = 0.10
    max_golden_failures: int = 0
    max_malformed_output_rate: float = 0.01
    max_prompt_tokens: int = 8000
    require_human_approval: bool = True


class AppSettings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    environment: str = "local"
    database_url: str = Field(alias="DATABASE_URL")
    mlflow_tracking_uri: str = Field(alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="rulekiln", alias="MLFLOW_EXPERIMENT_NAME")
    artifact_root: str = Field(default=".rulekiln/runs", alias="ARTIFACT_ROOT")
    import_artifact_root: str = Field(default=".rulekiln/imports", alias="IMPORT_ARTIFACT_ROOT")
    enable_pgvector: bool = Field(default=False, alias="ENABLE_PGVECTOR")
    mlflow_ui_base_url: str | None = Field(default=None, alias="MLFLOW_UI_BASE_URL")
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES")
    max_csv_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        alias="MAX_CSV_UPLOAD_SIZE_BYTES",
    )
    max_csv_rows: int = Field(default=10_000, alias="MAX_CSV_ROWS")

    # Optional API keys — providers read their own key from env by name
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")

    # ── Execution backend ──────────────────────────────────────────────────
    execution_backend: Literal["background_tasks", "postgres_queue"] = Field(
        default="postgres_queue", alias="EXECUTION_BACKEND"
    )
    worker_poll_interval_seconds: int = Field(default=2, alias="WORKER_POLL_INTERVAL_SECONDS")
    worker_lease_seconds: int = Field(default=1800, alias="WORKER_LEASE_SECONDS")

    # ── Provider rate limiting defaults ──────────────────────────────────────
    default_provider_max_concurrency: int = Field(
        default=3, alias="DEFAULT_PROVIDER_MAX_CONCURRENCY"
    )
    default_provider_rate_limit_rpm: int | None = Field(
        default=None, alias="DEFAULT_PROVIDER_RATE_LIMIT_RPM"
    )
    default_provider_rate_limit_tpm: int | None = Field(
        default=None, alias="DEFAULT_PROVIDER_RATE_LIMIT_TPM"
    )

    # ── Rule pruning defaults ─────────────────────────────────────────────
    default_max_rules: int = Field(default=40, alias="DEFAULT_MAX_RULES")
    default_min_rule_support_count: int = Field(default=2, alias="DEFAULT_MIN_RULE_SUPPORT_COUNT")
    default_max_prompt_tokens: int = Field(default=8000, alias="DEFAULT_MAX_PROMPT_TOKENS")

    default_quality_gate: QualityGateDefaults = Field(
        default_factory=QualityGateDefaults,
        alias="DEFAULT_QUALITY_GATE",
    )

    provider_profiles: dict[str, ProviderProfile] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached singleton AppSettings instance."""
    return AppSettings()  # pyright: ignore[reportCallIssue]
