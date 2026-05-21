"""OpenAI-compatible embedding provider (custom base_url)."""

import os

import httpx

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embedding adapter for any OpenAI-compatible endpoint."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        if config.base_url is None:
            raise ProviderNotConfiguredError(
                "openai_compatible", "base_url is required for openai_compatible embedding."
            )
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "dummy")
        url = config.base_url.rstrip("/") + "/embeddings"

        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": config.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
