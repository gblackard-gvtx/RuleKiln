"""DistillationPipelineRunner: entry point for orchestrated pipeline execution.

PR 1 stub — delegates directly to _run() in distillation_worker.  As stages are
extracted in PRs 3–7, _run() shrinks and the orchestrator calls service classes
instead of the monolith.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.pipeline.phase_plan import PipelinePhase


class DistillationPipelineRunner:
    """Thin orchestration layer between the public worker API and stage services.

    Instantiated per-request by the public entry points in distillation_worker.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def run(
        self,
        job_id: str,
        payload: DistillationRequest,
        *,
        phase: PipelinePhase = "full",
    ) -> None:
        # Lazy import breaks the circular dependency:
        # distillation_worker → orchestrator → (lazy) distillation_worker._run
        from rulekiln.workers import distillation_worker as _dw  # noqa: PLC0415

        await _dw._run(self._session, job_id, payload, phase=phase)
