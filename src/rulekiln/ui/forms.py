"""Multipart form parsers for the operator UI."""

from typing import Annotated

from fastapi import File, Form, Request, UploadFile


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


class CsvUploadForm:
    """Parses the CSV upload form (step 1)."""

    def __init__(
        self,
        csv_file: Annotated[UploadFile, File(description="CSV file to import")],
        task_id: Annotated[str, Form()],
        task_name: Annotated[str, Form()],
        task_mode: Annotated[str, Form()],
        description: Annotated[str, Form()] = "",
    ) -> None:
        self.csv_file = csv_file
        self.task_id = task_id
        self.task_name = task_name
        self.task_mode = task_mode
        self.description = description


class CsvMappingForm:
    """Parses the per-column mapping form (step 2 → step 3).

    Field name convention: col__{column_name}__{field}
    e.g. col__transcript__role, col__transcript__path, col__transcript__create_assertion
    """

    def __init__(self, request: Request) -> None:
        self._request = request
        self.import_id: str = ""
        self.column_mappings: list[dict[str, str | None]] = []

    async def parse(self) -> None:
        form_data = await self._request.form()
        self.import_id = str(form_data.get("import_id", ""))

        column_names: list[str] = []
        for key in form_data:
            if key.startswith("col__") and key.endswith("__role"):
                column_names.append(key[len("col__") : -len("__role")])

        for column_name in column_names:
            role = str(form_data.get(f"col__{column_name}__role", "ignore"))
            path_value = str(form_data.get(f"col__{column_name}__path", "")).strip() or None
            assertion_type = (
                str(form_data.get(f"col__{column_name}__assertion_type", "")).strip() or None
            )
            create_assertion = bool(form_data.get(f"col__{column_name}__create_assertion", ""))
            self.column_mappings.append(
                {
                    "column_name": column_name,
                    "role": role,
                    "path": path_value,
                    "assertion_type": assertion_type,
                    "create_assertion": str(create_assertion),
                }
            )
