"""Recursive JSON walk: anonymize strings, preserve unknown keys and tool schemas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from enigmatic.presidio_ops.images import redact_data_url
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]

SKIP_KEY_NAMES = {"model", "encoding_format"}
SCHEMA_KEYS = {"parameters", "input_schema", "json_schema", "schema"}


def _is_schema_path(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    if path[-1] in SCHEMA_KEYS:
        return True
    if "json_schema" in path:
        return True
    # tools[].function.parameters / functions[].parameters
    if "parameters" in path and ("tools" in path or "functions" in path):
        return True
    if "input_schema" in path and "tools" in path:
        return True
    return False


def walk(
    value: JSONValue,
    pipeline: Pipeline,
    mapping: SessionMapping,
    path: tuple[str, ...] = (),
) -> JSONValue:
    """Anonymize every JSON string except model ids and tool/function schemas."""
    if isinstance(value, str):
        if path and path[-1] in SKIP_KEY_NAMES:
            return value
        if _is_schema_path(path):
            return value
        if value.startswith("data:image"):
            return redact_data_url(value, mapping)
        return pipeline.anonymize_text(value, mapping)
    if isinstance(value, list):
        return [walk(item, pipeline, mapping, path) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            child_path = path + (key,)
            if key in SKIP_KEY_NAMES:
                out[key] = item
                continue
            out[key] = walk(item, pipeline, mapping, child_path)
        return out
    return value


def collect_strings(value: JSONValue, path: tuple[str, ...] = ()) -> list[str]:
    """Test helper: strings the walker would send to Presidio."""
    found: list[str] = []

    def visit(node: JSONValue, node_path: tuple[str, ...]) -> None:
        if isinstance(node, str):
            if node_path and node_path[-1] in SKIP_KEY_NAMES:
                return
            if _is_schema_path(node_path):
                return
            found.append(node)
            return
        if isinstance(node, list):
            for item in node:
                visit(item, node_path)
            return
        if isinstance(node, dict):
            for key, item in node.items():
                visit(item, node_path + (key,))

    visit(value, path)
    return found


AnonymizeFn = Callable[[str], str]
