"""OpenAI embedding provider adapter."""

import httpx

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding adapter for OpenAI text-embedding models."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )

        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(
                _OPENAI_EMBED_URL,
                headers={"Authorization": f"Bearer {config.api_key}"},
                json={"model": config.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
