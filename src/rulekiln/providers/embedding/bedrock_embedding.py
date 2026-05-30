"""AWS Bedrock embedding provider adapter."""

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from rulekiln.providers.estimation import build_usage_from_provider
from rulekiln.providers.tracking import tracked_embedding_call
from rulekiln.schemas.usage import EmbeddingResult


class BedrockEmbeddingClient(EmbeddingClient):
    """Embedding adapter for AWS Bedrock Titan/Cohere embedding models."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> EmbeddingResult:
        if config.region is None:
            raise ProviderNotConfiguredError("bedrock", "region is required for bedrock embedding.")

        async def _call() -> EmbeddingResult:
            try:
                import json

                import boto3  # pyright: ignore[reportMissingModuleSource]
            except ImportError as exc:
                raise ProviderNotConfiguredError(
                    "bedrock", "boto3 is not installed. Add it to dependencies."
                ) from exc

            client = boto3.client("bedrock-runtime", region_name=config.region)
            results: list[list[float]] = []
            for text in texts:
                body = json.dumps({"inputText": text})
                response = client.invoke_model(
                    modelId=config.model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                payload = json.loads(response["body"].read())
                results.append(payload["embedding"])
            usage = build_usage_from_provider(input_tokens=None, output_tokens=0, total_tokens=None)
            return EmbeddingResult(embeddings=results, usage=usage, raw_model=config.model)

        return await tracked_embedding_call(
            call=_call,
            fallback_input_text=" ".join(texts),
        )
