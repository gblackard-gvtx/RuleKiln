"""Provider interface contracts: ProviderConfig, ChatModelClient, EmbeddingClient."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from rulekiln.schemas.usage import ChatCompletionResult, EmbeddingResult

if TYPE_CHECKING:
    from rulekiln.schemas.batch import BatchItem, BatchPollStatus, BatchResult


class ProviderConfig(BaseModel):
    """Resolved provider configuration for a single model call."""

    profile_name: str
    provider: Literal[
        "fake",
        "bedrock",
        "openai",
        "anthropic",
        "vertex_gemini",
        "azure_openai",
        "openai_compatible",
        "custom",
    ]
    model: str
    region: str | None = None
    base_url: str | None = None
    api_key: str | None = None  # resolved from api_key_env_var at config build time
    timeout_seconds: int = 120
    max_retries: int = 3

    # Effective rate limits (resolved from route override → profile → app default)
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None
    max_concurrency: int = 3


class ProviderNotImplementedError(NotImplementedError):
    """Raised when a provider adapter is a stub and not yet implemented."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Provider '{provider}' is not implemented in this build. "
            "Configure a supported provider (fake, openai, openai_compatible, bedrock)."
        )


class ProviderNotConfiguredError(ValueError):
    """Raised when a provider is configured but missing required runtime credentials."""

    def __init__(self, provider: str, detail: str) -> None:
        super().__init__(f"Provider '{provider}' is not configured: {detail}")


class ChatModelClient(ABC):
    """Abstract base for chat / completion providers."""

    @abstractmethod
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        """Call the model and return a ChatCompletionResult with the parsed model and usage."""
        ...


class BatchChatModelClient(ChatModelClient, ABC):
    """Abstract base for chat providers that also support the provider batch API.

    Providers subclass this when they implement asynchronous batch submission
    (e.g. OpenAI Batch API, Anthropic Message Batches).  The per-call
    ``complete_structured`` method from ``ChatModelClient`` must still be
    implemented — it remains the interface for non-batch stages.
    """

    @abstractmethod
    async def submit_batch(
        self,
        items: list[BatchItem],
        config: ProviderConfig,
    ) -> str:
        """Serialize *items* to the provider format, submit, and return the provider batch ID."""
        ...

    @abstractmethod
    async def poll_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
    ) -> BatchPollStatus:
        """Return the current status of an in-flight batch without blocking."""
        ...

    @abstractmethod
    async def collect_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
        *,
        output_schema_class_name: str,
    ) -> BatchResult:
        """Download and parse results for a completed batch.

        Only call after ``poll_batch`` reports ``processing=False``.
        Uses the schema registry to look up *output_schema_class_name* and
        parse each item's response text into a typed ``ChatCompletionResult``.
        """
        ...


class EmbeddingClient(ABC):
    """Abstract base for text embedding providers."""

    @abstractmethod
    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> EmbeddingResult:
        """Embed a list of texts and return an EmbeddingResult with vectors and usage."""
        ...
