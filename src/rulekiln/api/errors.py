"""API error schemas and global exception handlers."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rulekiln.observability.logging import get_logger
from rulekiln.providers.contracts import ProviderNotConfiguredError, ProviderNotImplementedError

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI app."""

    @app.exception_handler(ProviderNotImplementedError)
    async def provider_not_implemented(
        request: Request, exc: ProviderNotImplementedError
    ) -> JSONResponse:
        logger.warning("provider_not_implemented", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content=ErrorResponse(error="provider_not_implemented", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(ProviderNotConfiguredError)
    async def provider_not_configured(
        request: Request, exc: ProviderNotConfiguredError
    ) -> JSONResponse:
        logger.warning("provider_not_configured", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error="provider_not_configured", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("validation_error", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(error="validation_error", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_type=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="internal_server_error").model_dump(),
        )
