"""FastAPI lifespan handler for startup validation."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from rulekiln.config.settings import get_settings
from rulekiln.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Validate settings and provider profiles on startup."""
    configure_logging()
    settings = get_settings()

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured.")
    if not settings.mlflow_tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not configured.")

    for name, profile in settings.provider_profiles.items():
        logger.info("provider_profile_loaded", name=name, provider=profile.provider)

    logger.info(
        "rulekiln_startup",
        environment=settings.environment,
        enable_pgvector=settings.enable_pgvector,
        provider_profiles=list(settings.provider_profiles.keys()),
    )

    yield

    logger.info("rulekiln_shutdown")
