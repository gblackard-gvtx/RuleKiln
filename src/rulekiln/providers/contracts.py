"""Provider interface contracts: ProviderConfig, ChatModelClient, EmbeddingClient."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel


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
    timeout_seconds: int = 60
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
    ) -> BaseModel:
        """Call the model and return a structured Pydantic model instance."""
        ...


class EmbeddingClient(ABC):
    """Abstract base for text embedding providers."""

    @abstractmethod
    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        """Embed a list of texts and return a list of float vectors."""
        ...
