"""OpenAI embedding provider adapter."""

import os

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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderNotConfiguredError("openai", "OPENAI_API_KEY is not set.")

        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(
                _OPENAI_EMBED_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": config.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
