"""ProviderBundle: all resolved provider configs and chat clients for a job."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rulekiln.providers.contracts import ChatModelClient, EmbeddingClient, ProviderConfig


class ProviderBundle(BaseModel):
    """Resolved provider configs and instantiated clients for a pipeline run.

    Built once per job after validation and threaded to every stage that calls
    an LLM or embedding provider.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    teacher_config: ProviderConfig
    student_config: ProviderConfig
    embedding_config: ProviderConfig
    judge_config: ProviderConfig
    extraction_teacher_config: ProviderConfig
    synthesis_teacher_config: ProviderConfig
    refinement_teacher_config: ProviderConfig

    teacher_chat: ChatModelClient
    student_chat: ChatModelClient
    judge_chat: ChatModelClient
    extraction_teacher_chat: ChatModelClient
    synthesis_teacher_chat: ChatModelClient
    refinement_teacher_chat: ChatModelClient
    embedding_client: EmbeddingClient

    eval_student_id: str
