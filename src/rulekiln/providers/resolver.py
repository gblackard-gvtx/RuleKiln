"""Provider profile normalization and route resolution."""

import os
from typing import Literal

from rulekiln.config.settings import AppSettings
from rulekiln.providers.contracts import ProviderConfig
from rulekiln.schemas.task_case import ModelRoute


def normalize_profile_name(name: str) -> str:
    """Normalize profile name to lowercase with underscores."""
    return name.strip().lower().replace("-", "_")


def resolve_provider_config(
    provider_profile: str,
    model: str,
    *,
    role: Literal["teacher", "student", "embedding", "judge"],
    settings: AppSettings,
    route: ModelRoute | None = None,
) -> ProviderConfig:
    """Resolve a named provider profile and model into a ProviderConfig.

    Effective rate limits use precedence: route override → profile → app default.
    API key is resolved once from the profile's api_key_env_var.

    Raises:
        ValueError: If profile is unknown or does not support the requested role.
    """
    profile_name = normalize_profile_name(provider_profile)
    profile = settings.provider_profiles.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unknown provider profile '{provider_profile}'. "
            f"Available profiles: {list(settings.provider_profiles.keys())}"
        )

    embedding_roles = {"embedding"}
    chat_roles = {"teacher", "student", "judge"}

    if role in embedding_roles and not profile.supports_embeddings:
        raise ValueError(
            f"Provider profile '{provider_profile}' does not support embeddings. "
            "Set supports_embeddings=true in the profile configuration."
        )

    if role in chat_roles and not profile.supports_chat:
        raise ValueError(
            f"Provider profile '{provider_profile}' does not support chat. "
            "Set supports_chat=true in the profile configuration."
        )

    # Resolve API key once from the named environment variable
    api_key: str | None = None
    if profile.api_key_env_var:
        api_key = os.environ.get(profile.api_key_env_var) or None

    # Rate limit precedence: route override → profile → app default
    effective_rpm: int | None = (
        (route.rate_limit_rpm if route else None)
        or profile.rate_limit_rpm
        or settings.default_provider_rate_limit_rpm
    )
    effective_tpm: int | None = (
        (route.rate_limit_tpm if route else None)
        or profile.rate_limit_tpm
        or settings.default_provider_rate_limit_tpm
    )
    effective_concurrency: int = (
        (route.max_concurrency if route and route.max_concurrency is not None else None)
        or profile.max_concurrency
        or settings.default_provider_max_concurrency
    )

    return ProviderConfig(
        profile_name=profile_name,
        provider=profile.provider,
        model=model,
        region=profile.region,
        base_url=profile.base_url,
        api_key=api_key,
        timeout_seconds=profile.timeout_seconds,
        max_retries=profile.max_retries,
        rate_limit_rpm=effective_rpm,
        rate_limit_tpm=effective_tpm,
        max_concurrency=effective_concurrency,
    )
