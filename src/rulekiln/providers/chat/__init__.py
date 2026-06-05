"""Chat provider factory: maps ProviderConfig to the correct ChatModelClient."""

from __future__ import annotations

from pydantic import BaseModel

from rulekiln.providers.contracts import (
    BatchChatModelClient,
    ChatModelClient,
    ProviderConfig,
    ProviderNotImplementedError,
)
from rulekiln.providers.rate_limiter import get_rate_limiter
from rulekiln.schemas.batch import BatchItem, BatchPollStatus, BatchResult
from rulekiln.schemas.usage import ChatCompletionResult


class _RateLimitedChatClient(ChatModelClient):
    """Wraps any ChatModelClient with rate limiting via the global ProviderRateLimiter."""

    def __init__(self, inner: ChatModelClient) -> None:
        self._inner = inner

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
    ) -> ChatCompletionResult:
        limiter = get_rate_limiter()
        await limiter.acquire(config)
        try:
            return await self._inner.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                config=config,
            )
        finally:
            limiter.release(config)


class _RateLimitedBatchChatClient(BatchChatModelClient):
    """Wraps a BatchChatModelClient preserving the BatchChatModelClient interface.

    Rate-limits the per-call ``complete_structured`` path; batch methods are
    delegated directly because they manage their own concurrency through the
    provider's batch service.
    """

    def __init__(self, inner: BatchChatModelClient) -> None:
        self._inner = inner

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        limiter = get_rate_limiter()
        await limiter.acquire(config)
        try:
            return await self._inner.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                config=config,
            )
        finally:
            limiter.release(config)

    async def submit_batch(
        self,
        items: list[BatchItem],
        config: ProviderConfig,
    ) -> str:
        return await self._inner.submit_batch(items, config)

    async def poll_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
    ) -> BatchPollStatus:
        return await self._inner.poll_batch(batch_id, config)

    async def collect_batch(
        self,
        batch_id: str,
        config: ProviderConfig,
        *,
        output_schema_class_name: str,
    ) -> BatchResult:
        return await self._inner.collect_batch(
            batch_id,
            config,
            output_schema_class_name=output_schema_class_name,
        )


def get_chat_client(config: ProviderConfig) -> ChatModelClient:
    """Return the ChatModelClient implementation for the given provider config."""
    match config.provider:
        case "fake":
            from rulekiln.providers.chat.fake import FakeChatClient

            inner: ChatModelClient = FakeChatClient()
        case "openai":
            from rulekiln.providers.chat.openai_chat import OpenAIChatClient

            inner = OpenAIChatClient()
        case "openai_compatible":
            from rulekiln.providers.chat.openai_compatible_chat import OpenAICompatibleChatClient

            inner = OpenAICompatibleChatClient()
        case "bedrock":
            from rulekiln.providers.chat.bedrock_chat import BedrockChatClient

            inner = BedrockChatClient()
        case "anthropic":
            from rulekiln.providers.chat.anthropic_chat import AnthropicChatClient

            inner = AnthropicChatClient()
        case "vertex_gemini":
            from rulekiln.providers.chat.stubs import VertexGeminiChatClient

            inner = VertexGeminiChatClient()
        case "azure_openai":
            from rulekiln.providers.chat.stubs import AzureOpenAIChatClient

            inner = AzureOpenAIChatClient()
        case "custom":
            from rulekiln.providers.chat.stubs import CustomChatClient

            inner = CustomChatClient()
        case _:
            raise ProviderNotImplementedError(config.provider)
    if isinstance(inner, BatchChatModelClient):
        return _RateLimitedBatchChatClient(inner)
    return _RateLimitedChatClient(inner)
