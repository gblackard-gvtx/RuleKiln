"""OpenAI-compatible chat provider (any base_url, e.g. Ollama, Together AI)."""

import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from rulekiln.providers.contracts import (
    ChatModelClient,
    ProviderConfig,
    ProviderNotConfiguredError,
)


class OpenAICompatibleChatClient(ChatModelClient):
    """Chat adapter for OpenAI-compatible endpoints (custom base_url)."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> BaseModel:
        if config.base_url is None:
            raise ProviderNotConfiguredError(
                "openai_compatible", "base_url is required for openai_compatible provider."
            )
        api_key_env = "OPENAI_COMPATIBLE_API_KEY"
        api_key = os.environ.get(api_key_env, "dummy")  # some endpoints accept any key

        model = OpenAIModel(config.model, base_url=config.base_url, api_key=api_key)
        agent: Agent[None, BaseModel] = Agent(  # pyright: ignore[reportUnknownVariableType]
            model,
            result_type=output_schema,  # pyright: ignore[reportArgumentType]
            system_prompt=system_prompt,
        )
        result = await agent.run(user_prompt)
        return result.data  # pyright: ignore[reportReturnType]
