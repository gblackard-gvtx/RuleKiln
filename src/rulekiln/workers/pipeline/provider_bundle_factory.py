"""ProviderBundleFactory: resolves all provider configs and instantiates clients."""

from __future__ import annotations

from rulekiln.config.settings import AppSettings
from rulekiln.providers.chat import get_chat_client
from rulekiln.providers.embedding import get_embedding_client
from rulekiln.providers.resolver import resolve_provider_config
from rulekiln.schemas.job import DistillationRequest
from rulekiln.workers.pipeline.provider_bundle import ProviderBundle


class ProviderBundleFactory:
    """Resolves provider configs and instantiates all clients for one pipeline run."""

    @staticmethod
    def create(payload: DistillationRequest, settings: AppSettings) -> ProviderBundle:
        tc = payload.teacher_config

        teacher_config = resolve_provider_config(
            payload.teacher.provider_profile,
            payload.teacher.model,
            role="teacher",
            settings=settings,
        )
        student_config = resolve_provider_config(
            payload.student.provider_profile,
            payload.student.model,
            role="student",
            settings=settings,
        )
        embedding_config = resolve_provider_config(
            payload.embedding.provider_profile,
            payload.embedding.model,
            role="embedding",
            settings=settings,
        )
        judge_route = payload.judge or payload.teacher
        judge_config = resolve_provider_config(
            judge_route.provider_profile,
            judge_route.model,
            role="judge",
            settings=settings,
        )

        extraction_teacher_config = (
            resolve_provider_config(
                tc.for_phase("instruction_extraction").provider,
                tc.for_phase("instruction_extraction").model,
                role="teacher",
                settings=settings,
            )
            if tc is not None
            else teacher_config
        )
        synthesis_teacher_config = (
            resolve_provider_config(
                tc.for_phase("cluster_consolidation").provider,
                tc.for_phase("cluster_consolidation").model,
                role="teacher",
                settings=settings,
            )
            if tc is not None
            else teacher_config
        )
        refinement_teacher_config = (
            resolve_provider_config(
                tc.for_phase("conflict_resolution").provider,
                tc.for_phase("conflict_resolution").model,
                role="teacher",
                settings=settings,
            )
            if tc is not None
            else teacher_config
        )

        teacher_chat = get_chat_client(teacher_config)
        student_chat = get_chat_client(student_config)
        embedding_client = get_embedding_client(embedding_config)
        judge_chat = get_chat_client(judge_config)
        extraction_teacher_chat = (
            get_chat_client(extraction_teacher_config) if tc is not None else teacher_chat
        )
        synthesis_teacher_chat = (
            get_chat_client(synthesis_teacher_config) if tc is not None else teacher_chat
        )
        refinement_teacher_chat = (
            get_chat_client(refinement_teacher_config) if tc is not None else teacher_chat
        )

        return ProviderBundle(
            teacher_config=teacher_config,
            student_config=student_config,
            embedding_config=embedding_config,
            judge_config=judge_config,
            extraction_teacher_config=extraction_teacher_config,
            synthesis_teacher_config=synthesis_teacher_config,
            refinement_teacher_config=refinement_teacher_config,
            teacher_chat=teacher_chat,
            student_chat=student_chat,
            judge_chat=judge_chat,
            extraction_teacher_chat=extraction_teacher_chat,
            synthesis_teacher_chat=synthesis_teacher_chat,
            refinement_teacher_chat=refinement_teacher_chat,
            embedding_client=embedding_client,
            eval_student_id=student_config.model,
        )
