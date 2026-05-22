"""Unit tests for provider resolver rate limit precedence."""

from __future__ import annotations

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.schemas.task_case import ModelRoute


def _default_profile(**kwargs: object) -> ProviderProfile:
    base = {"provider": "fake"}
    base.update(kwargs)  # type: ignore[arg-type]
    return ProviderProfile(**base)  # type: ignore[arg-type]


def _settings(
    rpm: int | None = None,
    tpm: int | None = None,
    concurrency: int = 3,
    profiles: dict[str, ProviderProfile] | None = None,
) -> AppSettings:
    default_profiles: dict[str, ProviderProfile] = {"default": _default_profile()}
    if profiles:
        default_profiles.update(profiles)
    return AppSettings(
        DATABASE_URL="postgresql+asyncpg://x:y@localhost/db",
        MLFLOW_TRACKING_URI="http://localhost:5000",
        default_provider_rate_limit_rpm=rpm,
        default_provider_rate_limit_tpm=tpm,
        default_provider_max_concurrency=concurrency,
        provider_profiles=default_profiles,
    )


def test_defaults_applied_when_no_route_or_profile() -> None:
    # Profile max_concurrency (3 default) takes precedence over app default;
    # only rpm/tpm app defaults reach through since profile has no rpm by default.
    settings = _settings(rpm=60, tpm=10000, concurrency=4)
    config = resolve_provider_config("default", "gpt-4o", role="teacher", settings=settings)
    assert config.rate_limit_rpm == 60
    assert config.rate_limit_tpm == 10000
    # profile default max_concurrency=3 wins over app default=4
    assert config.max_concurrency == 3


def test_route_overrides_defaults() -> None:
    route = ModelRoute(
        provider_profile="default",
        model="gpt-4o",
        rate_limit_rpm=120,
        max_concurrency=8,
    )
    settings = _settings(rpm=60, concurrency=4)
    config = resolve_provider_config(
        "default", "gpt-4o", role="teacher", settings=settings, route=route
    )
    assert config.rate_limit_rpm == 120
    assert config.max_concurrency == 8


def test_profile_overrides_default_but_not_route() -> None:
    profile = _default_profile(rate_limit_rpm=90, max_concurrency=6)
    route = ModelRoute(
        provider_profile="my_profile",
        model="gpt-4o",
        rate_limit_rpm=None,  # no override on route
        max_concurrency=None,
    )
    settings = _settings(rpm=30, concurrency=2, profiles={"my_profile": profile})
    config = resolve_provider_config(
        "my_profile", "gpt-4o", role="teacher", settings=settings, route=route
    )
    # profile (90) overrides default (30)
    assert config.rate_limit_rpm == 90
    # profile (6) overrides default (2)
    assert config.max_concurrency == 6
