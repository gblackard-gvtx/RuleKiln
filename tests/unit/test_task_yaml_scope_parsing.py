"""Guard against the YAML scope-colon parsing bug.

An unquoted ``task_scope``/``non_scope`` list item that contains a colon (e.g.
``Classify into exactly one of: a, b, c``) is parsed by YAML as a single-key
mapping rather than a string, which produces garbled Rules sections downstream.
This test asserts every shipped ``examples/**/task.yaml`` parses cleanly with
string-only scope items so a regression is caught in CI rather than in a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_YAML_FILES = sorted((_REPO_ROOT / "examples").rglob("task.yaml"))
_SCOPE_KEYS = ("task_scope", "non_scope")


def test_task_yaml_files_exist() -> None:
    """Sanity check that the glob actually found task files to validate."""
    assert _TASK_YAML_FILES, "no examples/**/task.yaml files found to validate"


@pytest.mark.parametrize(
    "task_yaml",
    _TASK_YAML_FILES,
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_task_scope_items_are_strings(task_yaml: Path) -> None:
    """Every scope item must parse as a plain string, never a dict/list."""
    parsed = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{task_yaml} did not parse to a mapping"

    for key in _SCOPE_KEYS:
        items = parsed.get(key)
        if items is None:
            continue
        assert isinstance(items, list), (
            f"{task_yaml}: {key} must be a list, got {type(items).__name__}"
        )
        non_strings = [item for item in items if not isinstance(item, str)]
        assert not non_strings, (
            f"{task_yaml}: {key} contains non-string items {non_strings!r}. "
            "A list item with a colon must be quoted so YAML parses it as a string."
        )
