"""OpenAI-compatible embedding provider (custom base_url)."""

import httpx

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_embedding_call
from rulekiln.schemas.usage import EmbeddingResult


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embedding adapter for any OpenAI-compatible endpoint."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> EmbeddingResult:
        if config.base_url is None:
            raise ProviderNotConfiguredError(
                "openai_compatible", "base_url is required for openai_compatible embedding."
            )
        api_key = config.api_key or "dummy"
        url = config.base_url.rstrip("/") + "/embeddings"

        async def _call() -> EmbeddingResult:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": config.model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                usage_data = data.get("usage", {})
                prompt_tokens = usage_data.get("prompt_tokens") if usage_data else None
                usage = (
                    build_usage_from_provider(
                        input_tokens=prompt_tokens,
                        output_tokens=0,
                        total_tokens=usage_data.get("total_tokens", prompt_tokens)
                        if usage_data
                        else None,
                    )
                    if prompt_tokens is not None
                    else None
                )
                return EmbeddingResult(
                    embeddings=embeddings,
                    usage=usage,
                    raw_model=data.get("model"),
                )

        return await tracked_embedding_call(
            call=_call,
            fallback_input_text=" ".join(texts),
        )
