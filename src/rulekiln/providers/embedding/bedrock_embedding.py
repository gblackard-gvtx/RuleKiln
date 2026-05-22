"""AWS Bedrock embedding provider adapter."""

from rulekiln.providers.contracts import (
    EmbeddingClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


class BedrockEmbeddingClient(EmbeddingClient):
    """Embedding adapter for AWS Bedrock Titan/Cohere embedding models."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        if config.region is None:
            raise ProviderNotConfiguredError("bedrock", "region is required for bedrock embedding.")

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
        return results
