"""Chat provider factory: maps ProviderConfig to the correct ChatModelClient."""

from pydantic import BaseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotImplementedError,
)
from rulekiln.providers.rate_limiter import get_rate_limiter


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
    ) -> BaseModel:
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
    return _RateLimitedChatClient(inner)
