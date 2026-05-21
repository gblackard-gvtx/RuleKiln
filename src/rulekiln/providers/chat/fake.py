"""Fake chat provider for offline testing and local development."""

import json

from pydantic import BaseModel

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig


class FakeChatClient(ChatModelClient):
    """Returns deterministic stub responses populated from output_schema defaults."""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> BaseModel:
        # Build a minimal valid instance by constructing from defaults
        return output_schema.model_validate({})
