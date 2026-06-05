"""Registry mapping output schema class names to Pydantic types for batch result parsing.

All Pydantic models used as output_schema in batch-eligible pipeline stages must be
registered here before batch jobs are submitted. Use @register_schema on the class or
call register_schema(cls) explicitly at import time.
"""

from __future__ import annotations

from pydantic import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


class BatchSchemaRegistryError(LookupError):
    """Raised when a schema class name has not been registered."""


def register_schema(cls: type[BaseModel]) -> type[BaseModel]:
    """Register a Pydantic model class for use in batch result parsing.

    Can be used as a decorator or called directly::

        @register_schema
        class ExtractionOutput(BaseModel): ...

        register_schema(SomeOtherModel)
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def get_schema_class(name: str) -> type[BaseModel]:
    """Return the registered schema class for *name*.

    Raises BatchSchemaRegistryError if the class has not been registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise BatchSchemaRegistryError(
            f"Unknown batch schema: {name!r}. "
            "Ensure the class is decorated with @register_schema at import time."
        ) from exc
