"""Embedding provider factory."""

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotImplementedError,
)


def get_embedding_client(config: ProviderConfig) -> EmbeddingClient:
    """Return the EmbeddingClient implementation for the given provider config."""
    match config.provider:
        case "fake":
            from rulekiln.providers.embedding.fake import FakeEmbeddingClient
            return FakeEmbeddingClient()
        case "openai":
            from rulekiln.providers.embedding.openai_embedding import OpenAIEmbeddingClient
            return OpenAIEmbeddingClient()
        case "openai_compatible":
            from rulekiln.providers.embedding.openai_compatible_embedding import OpenAICompatibleEmbeddingClient
            return OpenAICompatibleEmbeddingClient()
        case "bedrock":
            from rulekiln.providers.embedding.bedrock_embedding import BedrockEmbeddingClient
            return BedrockEmbeddingClient()
        case "anthropic":
            from rulekiln.providers.embedding.stubs import AnthropicEmbeddingClient
            return AnthropicEmbeddingClient()
        case "vertex_gemini":
            from rulekiln.providers.embedding.stubs import VertexGeminiEmbeddingClient
            return VertexGeminiEmbeddingClient()
        case "azure_openai":
            from rulekiln.providers.embedding.stubs import AzureOpenAIEmbeddingClient
            return AzureOpenAIEmbeddingClient()
        case "custom":
            from rulekiln.providers.embedding.stubs import CustomEmbeddingClient
            return CustomEmbeddingClient()
        case _:
            raise ProviderNotImplementedError(config.provider)
