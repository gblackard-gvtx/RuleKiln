"""Stub chat providers for deferred providers (vertex_gemini, azure_openai, custom)."""

from pydantic import BaseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotImplementedError,
)
from rulekiln.schemas.usage import ChatCompletionResult


class VertexGeminiChatClient(ChatModelClient):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        raise ProviderNotImplementedError("vertex_gemini")


class AzureOpenAIChatClient(ChatModelClient):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        raise ProviderNotImplementedError("azure_openai")


class CustomChatClient(ChatModelClient):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        raise ProviderNotImplementedError("custom")
