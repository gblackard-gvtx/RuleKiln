"""Entry-point: run the FastAPI application with uvicorn."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "rulekiln.api.app:app",
        host="0.0.0.0",  # noqa: S104 — bind is controlled by container/env
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
