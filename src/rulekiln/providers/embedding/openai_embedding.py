"""OpenAI embedding provider adapter."""

import httpx

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_embedding_call
from rulekiln.schemas.usage import EmbeddingResult

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding adapter for OpenAI text-embedding models."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> EmbeddingResult:
        if not config.api_key:
            raise ProviderNotConfiguredError(
                "openai",
                "api_key_env_var is not set or the referenced environment variable is empty.",
            )

        async def _call() -> EmbeddingResult:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                resp = await client.post(
                    _OPENAI_EMBED_URL,
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json={"model": config.model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                usage_data = data.get("usage", {})
                prompt_tokens = usage_data.get("prompt_tokens")
                usage = (
                    build_usage_from_provider(
                        input_tokens=prompt_tokens,
                        output_tokens=0,
                        total_tokens=usage_data.get("total_tokens", prompt_tokens),
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
