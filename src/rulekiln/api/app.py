"""FastAPI application factory."""

from fastapi import FastAPI

from rulekiln.api.errors import register_exception_handlers
from rulekiln.api.lifespan import lifespan
from rulekiln.api.routes.distillation_jobs import router as jobs_router
from rulekiln.api.routes.distillation_outputs import router as outputs_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RuleKiln",
        description="Prompt compiler that distils teacher LLM behaviour into a student prompt.",
        version="0.1.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(jobs_router, prefix="/v1")
    app.include_router(outputs_router, prefix="/v1")
    return app


app = create_app()
