"""Recursive JSON walk: anonymize strings, preserve unknown keys and tool schemas."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from enigmatic.presidio_ops.images import redact_data_url
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]

# Values under these keys are ids, enums, or opaque blobs. Changing them breaks the
# request (encrypted reasoning, call ids that must pair up, enum validation).
SKIP_KEY_NAMES = frozenset(
    {
        "model",
        "encoding_format",
        "encrypted_content",
        "signature",
        "id",
        "call_id",
        "tool_call_id",
        "tool_use_id",
        "item_id",
        "file_id",
        "previous_response_id",
        "response_id",
        "conversation",
        "role",
        "type",
        "status",
        "include",
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning_effort",
        "effort",
        "verbosity",
        "truncation",
        "tool_choice",
        "detail",
        "media_type",
        "mime_type",
    }
)
SCHEMA_KEYS = {"parameters", "input_schema", "json_schema", "schema"}
SCHEMA_TEXT_KEYS = {"description", "example", "examples"}
# Tool names must match between the tool list, calls, and outputs.
TOOL_CONTEXT_KEYS = frozenset({"tools", "tool_calls", "function", "function_call"})
TOOL_ITEM_TYPES = frozenset(
    {
        "function",
        "function_call",
        "function_call_output",
        "custom",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_use",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
    }
)
IMAGE_PART_TYPES = {"image_url": "text", "input_image": "input_text"}
JSON_TEXT_KEYS = frozenset({"arguments"})


def _is_schema_text(path: tuple[str, ...]) -> bool:
    return any(part in SCHEMA_TEXT_KEYS for part in path)


def _skip_string(path: tuple[str, ...]) -> bool:
    """Skip model ids and schema structure. Still scan descriptions and examples."""
    if path and path[-1] in SKIP_KEY_NAMES:
        return True
    if _is_schema_text(path):
        return False
    return _is_schema_path(path)


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


def _skip_key(key: str, parent: dict[str, Any], path: tuple[str, ...]) -> bool:
    """Keys whose whole value passes through untouched."""
    if key in SKIP_KEY_NAMES:
        return True
    if key == "name" and (parent.get("type") in TOOL_ITEM_TYPES or TOOL_CONTEXT_KEYS.intersection(path)):
        return True
    # Custom tool grammars (Codex apply_patch) are structure, not prose.
    if key == "format" and "tools" in path:
        return True
    if key == "summary" and path[-1:] == ("reasoning",) and isinstance(parent.get(key), str):
        return True
    # Raw base64 payloads: Anthropic image/document sources and input_audio.
    if key == "data" and path[-1:] in {("source",), ("input_audio",)}:
        return True
    return False


class _Walker:
    def __init__(
        self,
        on_text: Callable[[str], str],
        on_image: Callable[[str], str | None] | None,
    ) -> None:
        self._on_text = on_text
        self._on_image = on_image

    def value(self, value: JSONValue, path: tuple[str, ...]) -> JSONValue:
        if isinstance(value, str):
            return self.string(value, path)
        if isinstance(value, list):
            return [self.value(item, path) for item in value]
        if isinstance(value, dict):
            return self.mapping(value, path)
        return value

    def string(self, value: str, path: tuple[str, ...]) -> str:
        if _skip_string(path):
            return value
        if value.startswith("data:"):
            # Image data URLs outside a typed image part still go through redaction.
            if value.startswith("data:image") and self._on_image is not None:
                redacted = self._on_image(value)
                return redacted if redacted is not None else value
            return value
        if path and path[-1] in JSON_TEXT_KEYS:
            return self.json_text(value)
        return self._on_text(value)

    def json_text(self, value: str) -> str:
        """Scan decoded JSON values so restored originals can be re-escaped exactly."""
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return self._on_text(value)
        if not isinstance(decoded, dict | list):
            return self._on_text(value)
        walked = self.data(decoded)
        if walked == decoded:
            return value
        return json.dumps(walked, ensure_ascii=False, separators=(",", ":"))

    def data(self, value: JSONValue) -> JSONValue:
        """Tool-call arguments: every string is user data regardless of its key."""
        if isinstance(value, str):
            return self._on_text(value)
        if isinstance(value, list):
            return [self.data(item) for item in value]
        if isinstance(value, dict):
            return {key: self.data(item) for key, item in value.items()}
        return value

    def mapping(self, value: dict[str, Any], path: tuple[str, ...]) -> JSONValue:
        kind = value.get("type")
        if kind in IMAGE_PART_TYPES:
            return self.image_part(value, str(kind))
        if kind == "image" and isinstance(value.get("source"), dict):
            return self.anthropic_image(value)
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _skip_key(key, value, path):
                out[key] = item
                continue
            out[key] = self.value(item, path + (key,))
        return out

    def image_part(self, part: dict[str, Any], kind: str) -> dict[str, Any]:
        raw = part.get("image_url")
        url = raw.get("url") if isinstance(raw, dict) else raw
        if not isinstance(url, str) or not url.startswith("data:image") or self._on_image is None:
            return part
        redacted = self._on_image(url)
        if redacted is None:
            return part
        if not redacted.startswith("data:"):
            return {"type": IMAGE_PART_TYPES[kind], "text": redacted}
        out = dict(part)
        out["image_url"] = {**raw, "url": redacted} if isinstance(raw, dict) else redacted
        return out

    def anthropic_image(self, block: dict[str, Any]) -> dict[str, Any]:
        source = block["source"]
        if source.get("type") != "base64" or self._on_image is None:
            return block
        media = str(source.get("media_type") or "image/png")
        redacted = self._on_image(f"data:{media};base64,{source.get('data', '')}")
        if redacted is None:
            return block
        if not redacted.startswith("data:"):
            return {"type": "text", "text": redacted}
        header, _, b64 = redacted.partition(",")
        new_media = header[len("data:") :].split(";", 1)[0] or media
        return {**block, "source": {**source, "media_type": new_media, "data": b64}}


def walk(
    value: JSONValue,
    pipeline: Pipeline,
    mapping: SessionMapping,
    path: tuple[str, ...] = (),
) -> JSONValue:
    """Anonymize every JSON string except ids, enums, opaque blobs, and tool schemas."""
    walker = _Walker(
        on_text=lambda text: pipeline.anonymize_text(text, mapping),
        on_image=lambda url: redact_data_url(url, mapping),
    )
    return walker.value(value, path)


def collect_strings(value: JSONValue, path: tuple[str, ...] = ()) -> list[str]:
    """Test helper: strings the walker would send to Presidio."""
    found: list[str] = []

    def record(text: str) -> str:
        found.append(text)
        return text

    _Walker(on_text=record, on_image=None).value(value, path)
    return found


AnonymizeFn = Callable[[str], str]
