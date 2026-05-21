"""Request shape validation helpers."""

from rulekiln.config.settings import AppSettings, get_settings
from rulekiln.schemas.job import DistillationRequest


class RequestValidationError(ValueError):
    pass


def validate_distillation_request(
    payload: DistillationRequest,
    settings: AppSettings | None = None,
) -> None:
    """Validate provider profiles and case list beyond Pydantic schema checks."""
    if settings is None:
        settings = get_settings()

    if len(payload.cases) == 0:
        raise RequestValidationError("cases list must not be empty.")

    for route_name, route in [
        ("teacher", payload.teacher),
        ("student", payload.student),
        ("embedding", payload.embedding),
    ]:
        if route.provider_profile not in settings.provider_profiles:
            raise RequestValidationError(
                f"{route_name} provider_profile '{route.provider_profile}' "
                "is not configured in PROVIDER_PROFILES."
            )

    if payload.judge is not None:
        if payload.judge.provider_profile not in settings.provider_profiles:
            raise RequestValidationError(
                f"judge provider_profile '{payload.judge.provider_profile}' "
                "is not configured in PROVIDER_PROFILES."
            )
