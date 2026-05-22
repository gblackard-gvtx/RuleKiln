"""Embedding provider factory."""

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotImplementedError,
)
from rulekiln.providers.rate_limiter import get_rate_limiter


class _RateLimitedEmbeddingClient(EmbeddingClient):
    """Wraps any EmbeddingClient with rate limiting via the global ProviderRateLimiter."""

    def __init__(self, inner: EmbeddingClient) -> None:
        self._inner = inner

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        limiter = get_rate_limiter()
        await limiter.acquire(config)
        try:
            return await self._inner.embed_texts(texts=texts, config=config)
        finally:
            limiter.release(config)


def get_embedding_client(config: ProviderConfig) -> EmbeddingClient:
    """Return the EmbeddingClient implementation for the given provider config."""
    match config.provider:
        case "fake":
            from rulekiln.providers.embedding.fake import FakeEmbeddingClient
            inner: EmbeddingClient = FakeEmbeddingClient()
        case "openai":
            from rulekiln.providers.embedding.openai_embedding import OpenAIEmbeddingClient
            inner = OpenAIEmbeddingClient()
        case "openai_compatible":
            from rulekiln.providers.embedding.openai_compatible_embedding import OpenAICompatibleEmbeddingClient
            inner = OpenAICompatibleEmbeddingClient()
        case "bedrock":
            from rulekiln.providers.embedding.bedrock_embedding import BedrockEmbeddingClient
            inner = BedrockEmbeddingClient()
        case "anthropic":
            from rulekiln.providers.embedding.stubs import AnthropicEmbeddingClient
            inner = AnthropicEmbeddingClient()
        case "vertex_gemini":
            from rulekiln.providers.embedding.stubs import VertexGeminiEmbeddingClient
            inner = VertexGeminiEmbeddingClient()
        case "azure_openai":
            from rulekiln.providers.embedding.stubs import AzureOpenAIEmbeddingClient
            inner = AzureOpenAIEmbeddingClient()
        case "custom":
            from rulekiln.providers.embedding.stubs import CustomEmbeddingClient
            inner = CustomEmbeddingClient()
        case _:
            raise ProviderNotImplementedError(config.provider)
    return _RateLimitedEmbeddingClient(inner)
