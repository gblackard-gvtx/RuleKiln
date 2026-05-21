"""Stub chat providers for deferred providers (anthropic, vertex_gemini, azure_openai, custom)."""

from pydantic import BaseModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotImplementedError,
)


class AnthropicChatClient(ChatModelClient):
    async def complete_structured(self, *, system_prompt: str, user_prompt: str, output_schema: type[BaseModel], config: ProviderConfig) -> BaseModel:
        raise ProviderNotImplementedError("anthropic")


class VertexGeminiChatClient(ChatModelClient):
    async def complete_structured(self, *, system_prompt: str, user_prompt: str, output_schema: type[BaseModel], config: ProviderConfig) -> BaseModel:
        raise ProviderNotImplementedError("vertex_gemini")


class AzureOpenAIChatClient(ChatModelClient):
    async def complete_structured(self, *, system_prompt: str, user_prompt: str, output_schema: type[BaseModel], config: ProviderConfig) -> BaseModel:
        raise ProviderNotImplementedError("azure_openai")


class CustomChatClient(ChatModelClient):
    async def complete_structured(self, *, system_prompt: str, user_prompt: str, output_schema: type[BaseModel], config: ProviderConfig) -> BaseModel:
        raise ProviderNotImplementedError("custom")
