"""StageGuard: idempotent stage execution wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.repositories.jobs import is_stage_complete, mark_stage_complete, update_job_status
from rulekiln.observability.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


class StageGuard:
    """Wraps stage-marker idempotency checks for a single job.

    Every pipeline stage should go through run_once() instead of calling
    is_stage_complete / _set_stage / mark_stage_complete directly.  This
    centralises the resume semantics so they are easy to test and reason about.
    """

    __slots__ = ("_session", "_job_id")

    def __init__(self, session: AsyncSession, job_id: str) -> None:
        self._session = session
        self._job_id = job_id

    async def is_complete(
        self,
        stage: str,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
    ) -> bool:
        return await is_stage_complete(self._session, self._job_id, stage, strategy, artifact_type)

    async def start(self, stage: str) -> None:
        await update_job_status(self._session, self._job_id, status="running", stage=stage)
        logger.info("stage_started", job_id=self._job_id, stage=stage)

    async def complete(
        self,
        stage: str,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
    ) -> None:
        await mark_stage_complete(self._session, self._job_id, stage, strategy, artifact_type)

    async def run_once(
        self,
        stage: str,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
        fn: Callable[[], Awaitable[T]],
    ) -> T | None:
        """Run fn() exactly once for (stage, strategy, artifact_type).

        If the stage marker already exists (resume after failure), fn is skipped
        and None is returned.  Otherwise, start → fn() → complete, returning the
        result of fn().
        """
        if await self.is_complete(stage, strategy=strategy, artifact_type=artifact_type):
            return None
        await self.start(stage)
        result = await fn()
        await self.complete(stage, strategy=strategy, artifact_type=artifact_type)
        return result
