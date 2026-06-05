"""Classroom evaluator: run evaluation across multiple students concurrently.

Each student is evaluated independently. Results are returned as a mapping
from student_id to EvalResult so the caller can inspect per-student metrics.

The anchor student (identified by ClassroomConfig.anchor_student_id) drives the
closed-loop conflict-resolution loop. Non-anchor students run only once at the
final iteration; callers must pass only the anchor student during loop iterations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from rulekiln.pipeline.evaluator import evaluate_prompt
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.classroom import ClassroomConfig, StudentConfig
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask

CaseResultPersistFn = Callable[[str, CaseEvalResult], Coroutine[Any, Any, None]]


async def evaluate_classroom(
    *,
    system_prompt: str,
    cases: list[RuleKilnCase],
    task: RuleKilnTask,
    classroom_config: ClassroomConfig,
    get_chat_client: Callable[[StudentConfig], ChatModelClient],
    get_provider_config: Callable[[StudentConfig], ProviderConfig],
    strategy: str,
    split: str = "validation",
    bootstrap_enabled: bool = True,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 1729,
    on_student_case_result: CaseResultPersistFn | None = None,
    anchor_only: bool = False,
) -> dict[str, EvalResult]:
    """Evaluate *system_prompt* against all students in the classroom concurrently.

    Returns a mapping of ``student_id -> EvalResult``.  Results are keyed by ID,
    not index, so callers can reliably look up the anchor student's result.

    When ``anchor_only=True`` only the anchor student is evaluated (used during
    loop iterations to avoid paying non-anchor inference cost mid-loop).
    """
    students = [classroom_config.anchor_student] if anchor_only else classroom_config.students

    async def _eval_one(student: StudentConfig) -> tuple[str, EvalResult]:
        chat_client = get_chat_client(student)
        provider_config = get_provider_config(student)

        on_case_result = None
        if on_student_case_result is not None:
            student_id = student.id
            _persist = on_student_case_result  # narrow away Optional for closure capture

            async def _on_result(result: CaseEvalResult) -> None:
                await _persist(student_id, result)

            on_case_result = _on_result

        result = await evaluate_prompt(
            system_prompt,
            cases,
            task,
            chat_client,
            provider_config,
            strategy=f"{strategy}_{student.id}",
            split=split,
            bootstrap_enabled=bootstrap_enabled,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            on_case_result=on_case_result,
        )
        return student.id, result

    tasks = [_eval_one(s) for s in students]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)


def anchor_eval(
    classroom_results: dict[str, EvalResult],
    classroom_config: ClassroomConfig,
) -> EvalResult | None:
    """Return the anchor student's EvalResult from a classroom result dict."""
    return classroom_results.get(classroom_config.anchor_student_id)
