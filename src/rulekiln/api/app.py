"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from rulekiln.api.errors import register_exception_handlers
from rulekiln.api.lifespan import lifespan
from rulekiln.api.routes.distillation_jobs import router as jobs_router
from rulekiln.api.routes.distillation_outputs import router as outputs_router
from rulekiln.ui.routes import router as ui_router

_STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="RuleKiln",
        description="Prompt compiler that distils teacher LLM behaviour into a student prompt.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/jobs", status_code=status.HTTP_302_FOUND)

    register_exception_handlers(app)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(jobs_router, prefix="/v1")
    app.include_router(outputs_router, prefix="/v1")
    app.include_router(ui_router)
    return app


app = create_app()
