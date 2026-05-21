"""Provider profile normalization and route resolution."""

from typing import Literal

from rulekiln.config.settings import AppSettings
from rulekiln.providers.contracts import ProviderConfig


def normalize_profile_name(name: str) -> str:
    """Normalize profile name to lowercase with underscores."""
    return name.strip().lower().replace("-", "_")


def resolve_provider_config(
    provider_profile: str,
    model: str,
    *,
    role: Literal["teacher", "student", "embedding"],
    settings: AppSettings,
) -> ProviderConfig:
    """Resolve a named provider profile and model into a ProviderConfig.

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

    if role == "embedding" and not profile.supports_embeddings:
        raise ValueError(
            f"Provider profile '{provider_profile}' does not support embeddings. "
            "Set supports_embeddings=true in the profile configuration."
        )

    if role in {"teacher", "student"} and not profile.supports_chat:
        raise ValueError(
            f"Provider profile '{provider_profile}' does not support chat. "
            "Set supports_chat=true in the profile configuration."
        )

    return ProviderConfig(
        profile_name=profile_name,
        provider=profile.provider,
        model=model,
        region=profile.region,
        base_url=profile.base_url,
        timeout_seconds=profile.timeout_seconds,
        max_retries=profile.max_retries,
    )
