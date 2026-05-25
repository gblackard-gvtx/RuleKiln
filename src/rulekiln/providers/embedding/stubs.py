"""Stub embedding providers for deferred providers."""

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotImplementedError,
)
from rulekiln.schemas.usage import EmbeddingResult


class AnthropicEmbeddingClient(EmbeddingClient):
    async def embed_texts(self, *, texts: list[str], config: ProviderConfig) -> EmbeddingResult:
        raise ProviderNotImplementedError("anthropic")


class VertexGeminiEmbeddingClient(EmbeddingClient):
    async def embed_texts(self, *, texts: list[str], config: ProviderConfig) -> EmbeddingResult:
        raise ProviderNotImplementedError("vertex_gemini")


class AzureOpenAIEmbeddingClient(EmbeddingClient):
    async def embed_texts(self, *, texts: list[str], config: ProviderConfig) -> EmbeddingResult:
        raise ProviderNotImplementedError("azure_openai")


class CustomEmbeddingClient(EmbeddingClient):
    async def embed_texts(self, *, texts: list[str], config: ProviderConfig) -> EmbeddingResult:
        raise ProviderNotImplementedError("custom")
