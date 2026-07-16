"""Regression tests for provider-configured pydantic-ai retry wiring."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel

from rulekiln.providers.chat.bedrock_chat import BedrockChatClient
from rulekiln.providers.chat.openai_chat import OpenAIChatClient
from rulekiln.providers.chat.openai_compatible_chat import OpenAICompatibleChatClient
from rulekiln.providers.contracts import ProviderConfig


class _DummyOutput(BaseModel):
    """Simple structured output model used by retry wiring tests."""

    ok: bool = True


class _FakeUsage:
    """Minimal pydantic-ai usage object shape expected by adapters."""

    input_tokens: int = 11
    output_tokens: int = 7
    total_tokens: int = 18


class _FakeRunResult:
    """Minimal pydantic-ai run result shape expected by adapters."""

    output: BaseModel = _DummyOutput()
    usage: _FakeUsage = _FakeUsage()


@pytest.mark.asyncio
async def test_openai_chat_uses_configured_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI adapter should pass ProviderConfig.max_retries into Agent(...)."""
    import rulekiln.providers.chat.openai_chat as module

    captured: dict[str, int] = {}

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _FakeModel:
        def __init__(self, _model_name: str, *, provider: object) -> None:
            assert provider is not None

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "OpenAIProvider", _FakeProvider)
    monkeypatch.setattr(module, "OpenAIChatModel", _FakeModel)

    client = OpenAIChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="openai",
            model="gpt-test",
            api_key="test-key",
            max_retries=5,
        ),
    )

    assert captured == {"retries": 5}
    assert isinstance(result.parsed, _DummyOutput)


@pytest.mark.asyncio
async def test_openai_compatible_chat_uses_configured_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible adapter passes max_retries into Agent and uses a native-mode profile."""
    import rulekiln.providers.chat.openai_compatible_chat as module

    captured: dict[str, object] = {}

    class _FakeRunResult:
        output: BaseModel = _DummyOutput()
        usage: _FakeUsage = _FakeUsage()

        def new_messages(self) -> list[object]:
            return []  # no TextPart messages → content=""

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _FakeModel:
        def __init__(
            self, _model_name: str, *, provider: object, profile: object
        ) -> None:
            captured["profile"] = profile

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "OpenAIProvider", _FakeProvider)
    monkeypatch.setattr(module, "OpenAIChatModel", _FakeModel)

    client = OpenAICompatibleChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="openai_compatible",
            model="compat-test",
            base_url="http://localhost:9999/v1",
            api_key="test-key",
            max_retries=4,
        ),
    )

    assert captured["retries"] == 4
    # Verify we're using the native-mode profile (not the default tool-calling mode).
    # OpenAIModelProfile is a TypedDict, so it's a plain dict at runtime (no isinstance check).
    profile = cast("dict[str, object]", captured["profile"])
    assert profile["default_structured_output_mode"] == "native"
    assert isinstance(result.parsed, _DummyOutput)
    assert result.content == ""  # no TextPart messages in stub


@pytest.mark.asyncio
async def test_bedrock_chat_uses_configured_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock adapter should pass ProviderConfig.max_retries into Agent(...)."""
    import rulekiln.providers.chat.bedrock_chat as module

    captured: dict[str, int] = {}

    class _FakeAgent:
        def __init__(
            self,
            _model: object,
            *,
            output_type: type[BaseModel],
            system_prompt: str,
            retries: int,
        ) -> None:
            assert output_type is _DummyOutput
            assert system_prompt == "sys"
            captured["retries"] = retries

        async def run(self, _user_prompt: str) -> _FakeRunResult:
            return _FakeRunResult()

    class _FakeModel:
        def __init__(self, _model_name: str, *, region_name: str) -> None:
            assert region_name == "us-east-1"

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "BedrockConverseModel", _FakeModel)

    client = BedrockChatClient()
    result = await client.complete_structured(
        system_prompt="sys",
        user_prompt="user",
        output_schema=_DummyOutput,
        config=ProviderConfig(
            profile_name="p",
            provider="bedrock",
            model="bedrock-test",
            region="us-east-1",
            max_retries=6,
        ),
    )

    assert captured == {"retries": 6}
    assert isinstance(result.parsed, _DummyOutput)
