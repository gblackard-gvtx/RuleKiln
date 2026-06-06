"""Class-based pipeline services for the distillation worker.

This package is being built out via a staged strangler refactor.  Each PR moves
one coherent group of code out of distillation_worker.py and into a focused
service class here.  The public entry points in distillation_worker.py remain
unchanged throughout.
"""

from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.context_factory import PipelineContextFactory
from rulekiln.workers.pipeline.orchestrator import DistillationPipelineRunner
from rulekiln.workers.pipeline.phase_plan import PhasePlan, PipelinePhase
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle
from rulekiln.workers.pipeline.stage_guard import StageGuard

__all__ = [
    "DistillationPipelineRunner",
    "PhasePlan",
    "PipelineContext",
    "PipelineContextFactory",
    "PipelinePhase",
    "ProviderBundle",
    "StageGuard",
]
