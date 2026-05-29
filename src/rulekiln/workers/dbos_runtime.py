"""DBOS runtime availability helpers.

This module intentionally keeps DBOS imports optional so non-DBOS backends
continue to function in environments where the DBOS package is not installed.
"""

from __future__ import annotations

import importlib.util
from threading import Lock

from rulekiln.config.settings import AppSettings

_DBOS_LAUNCHED = False
_DBOS_LOCK = Lock()


def is_dbos_available() -> bool:
    """Return whether the DBOS package is importable in the current runtime."""
    return importlib.util.find_spec("dbos") is not None


def require_dbos_available() -> None:
    """Raise a clear runtime error if DBOS backend is requested but unavailable."""
    if is_dbos_available():
        return
    raise RuntimeError(
        "EXECUTION_BACKEND='dbos' requires the 'dbos' package. "
        "Install dependencies and restart the API/worker process."
    )


def ensure_dbos_runtime_launched(settings: AppSettings) -> None:
    """Initialize and launch DBOS once for the current process."""
    require_dbos_available()

    global _DBOS_LAUNCHED  # noqa: PLW0603
    if _DBOS_LAUNCHED:
        return

    with _DBOS_LOCK:
        if _DBOS_LAUNCHED:
            return

        from dbos import DBOS, DBOSConfig  # imported lazily to keep module optional

        config: DBOSConfig = {
            "name": "rulekiln",
            "database_url": settings.database_url,
        }
        DBOS(config=config)
        DBOS.launch()
        _DBOS_LAUNCHED = True
