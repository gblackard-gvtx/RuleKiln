"""Multipart form parsers for the operator UI."""

from typing import Annotated

from fastapi import File, Form, UploadFile


class NewJobForm:
    """Dependency class that parses the new-job multipart form submission."""

    def __init__(
        self,
        task_file: Annotated[UploadFile, File(description="task.yaml")],
        cases_file: Annotated[UploadFile, File(description="cases.jsonl")],
        teacher_profile: Annotated[str, Form()],
        teacher_model: Annotated[str, Form()],
        student_profile: Annotated[str, Form()],
        student_model: Annotated[str, Form()],
        embedding_profile: Annotated[str, Form()],
        embedding_model: Annotated[str, Form()],
        judge_profile: Annotated[str | None, Form()] = None,
        judge_model: Annotated[str | None, Form()] = None,
        baseline_prompt: Annotated[str | None, Form()] = None,
    ) -> None:
        self.task_file = task_file
        self.cases_file = cases_file
        self.teacher_profile = teacher_profile
        self.teacher_model = teacher_model
        self.student_profile = student_profile
        self.student_model = student_model
        self.embedding_profile = embedding_profile
        self.embedding_model = embedding_model
        self.judge_profile = judge_profile
        self.judge_model = judge_model
        self.baseline_prompt = baseline_prompt
