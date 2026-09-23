"""Field-aware placeholder restore for OpenAI Chat, OpenAI Responses, and Anthropic SSE.

Each logical text stream gets its own split-token buffer, so a partial `<EMAIL`
held back from one choice, tool call, or output item never leaks into another.
Buffers are flushed as a well-formed delta event right before the event that
closes that stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.restore import StreamRestorer

Dialect = Literal["chat", "responses", "anthropic"]
Event = tuple[str | None, Any]
BufferKey = tuple[Any, ...]

CHAT_TEXT_FIELDS = ("content", "refusal", "reasoning_content", "reasoning")
CHAT_META_KEYS = ("id", "object", "created", "model", "system_fingerprint", "service_tier")
RESPONSES_KEY_FIELDS = ("item_id", "output_index", "content_index", "summary_index")
RESPONSES_OPAQUE_DELTAS = frozenset({"response.audio.delta"})
RESPONSES_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})
ANTHROPIC_DELTA_FIELDS = (("text", False), ("thinking", False), ("partial_json", True))


class SseRestorer:
    def __init__(
        self,
        mapping: SessionMapping,
        dialect: Dialect,
        on_response_id: Callable[[str], None] | None = None,
    ) -> None:
        self._mapping = mapping
        self._dialect = dialect
        self._on_response_id = on_response_id
        self._buffers: dict[BufferKey, StreamRestorer] = {}
        self._templates: dict[BufferKey, dict[str, Any]] = {}
        self._chat_meta: dict[str, Any] = {}

    def restore(self, payload: Any, event: str | None = None) -> list[Event]:
        """Restore one parsed SSE payload. May prepend synthetic flush events."""
        if not isinstance(payload, dict):
            return [(event, self._mapping.restore_complete_any(payload))]
        if self._dialect == "chat":
            return self._chat(payload, event)
        if self._dialect == "responses":
            return self._responses(payload, event)
        return self._anthropic(payload, event)

    def finish(self) -> list[Event]:
        """Flush every buffer still holding text at end of stream."""
        if self._dialect == "chat":
            return self._chat_finish()
        return self._flush_matching(lambda _key: True)

    def _buffer(self, key: BufferKey, *, escape_json: bool = False) -> StreamRestorer:
        existing = self._buffers.get(key)
        if existing is None:
            existing = StreamRestorer(self._mapping, escape_json=escape_json)
            self._buffers[key] = existing
        return existing

    def _pop_leftover(self, key: BufferKey) -> str:
        restorer = self._buffers.pop(key, None)
        return restorer.flush() if restorer is not None else ""

    # Chat Completions

    def _chat(self, payload: dict[str, Any], event: str | None) -> list[Event]:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return [(event, self._mapping.restore_complete_any(payload))]
        for key in CHAT_META_KEYS:
            if key in payload:
                self._chat_meta[key] = payload[key]
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._chat_delta(index, delta)
            if choice.get("finish_reason") is not None:
                if not isinstance(delta, dict):
                    delta = {}
                    choice["delta"] = delta
                self._chat_flush_choice(index, delta)
        return [(event, payload)]

    def _chat_delta(self, index: Any, delta: dict[str, Any]) -> None:
        for field in CHAT_TEXT_FIELDS:
            value = delta.get(field)
            if isinstance(value, str):
                delta[field] = self._buffer(("chat", index, field)).feed(value)
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                self._chat_function(("chat", index, "tool", call.get("index", 0)), call.get("function"))
        self._chat_function(("chat", index, "function_call"), delta.get("function_call"))

    def _chat_function(self, key: BufferKey, fn: Any) -> None:
        if not isinstance(fn, dict):
            return
        if isinstance(fn.get("name"), str):
            fn["name"] = self._mapping.restore_complete(fn["name"])
        if isinstance(fn.get("arguments"), str):
            fn["arguments"] = self._buffer(key, escape_json=True).feed(fn["arguments"])

    def _chat_flush_choice(self, index: Any, delta: dict[str, Any]) -> None:
        for key in [key for key in self._buffers if key[1] == index]:
            leftover = self._pop_leftover(key)
            if leftover:
                _merge_chat_leftover(delta, key, leftover)

    def _chat_finish(self) -> list[Event]:
        deltas: dict[Any, dict[str, Any]] = {}
        for key in list(self._buffers):
            leftover = self._pop_leftover(key)
            if leftover:
                _merge_chat_leftover(deltas.setdefault(key[1], {}), key, leftover)
        return [
            (
                None,
                {
                    **self._chat_meta,
                    "choices": [{"index": index, "delta": delta, "finish_reason": None}],
                },
            )
            for index, delta in deltas.items()
        ]

    # Responses

    def _responses(self, payload: dict[str, Any], event: str | None) -> list[Event]:
        kind = payload.get("type")
        if not isinstance(kind, str):
            return [(event, self._mapping.restore_complete_any(payload))]
        self._note_response_id(kind, payload)
        if (
            kind.endswith(".delta")
            and isinstance(payload.get("delta"), str)
            and kind not in RESPONSES_OPAQUE_DELTAS
        ):
            key = _responses_key(kind[: -len(".delta")], payload)
            restorer = self._buffer(key, escape_json="arguments" in kind)
            payload["delta"] = restorer.feed(payload["delta"])
            self._templates[key] = {
                "type": kind,
                **{field: payload[field] for field in RESPONSES_KEY_FIELDS if field in payload},
            }
            return [(event, payload)]
        flushed: list[Event] = []
        sequence = payload.get("sequence_number")
        if kind.endswith(".done"):
            target = _responses_key(kind[: -len(".done")], payload)
            flushed = self._flush_matching(lambda key: key == target, sequence)
        elif kind == "response.output_item.done":
            item = payload.get("item")
            item_id = item.get("id") if isinstance(item, dict) else None
            flushed = self._flush_matching(lambda key: key[1] == item_id, sequence)
        elif kind in RESPONSES_TERMINAL:
            flushed = self._flush_matching(lambda _key: True, sequence)
        return [*flushed, (event, self._mapping.restore_complete_any(payload))]

    def _note_response_id(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_response_id is None or kind != "response.created":
            return
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            self._on_response_id(response["id"])

    # Anthropic Messages

    def _anthropic(self, payload: dict[str, Any], event: str | None) -> list[Event]:
        kind = payload.get("type")
        if kind == "content_block_delta" and isinstance(payload.get("delta"), dict):
            index = payload.get("index", 0)
            delta = payload["delta"]
            for field, escape_json in ANTHROPIC_DELTA_FIELDS:
                if isinstance(delta.get(field), str):
                    key: BufferKey = ("anthropic", index, field)
                    delta[field] = self._buffer(key, escape_json=escape_json).feed(delta[field])
                    self._templates[key] = {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": delta.get("type")},
                        "_field": field,
                    }
            return [(event, payload)]
        flushed: list[Event] = []
        if kind == "content_block_stop":
            index = payload.get("index", 0)
            flushed = self._flush_matching(lambda key: key[1] == index)
        elif kind in {"message_delta", "message_stop"}:
            flushed = self._flush_matching(lambda _key: True)
        return [*flushed, (event, self._mapping.restore_complete_any(payload))]

    # Shared flush for Responses and Anthropic

    def _flush_matching(
        self,
        predicate: Callable[[BufferKey], bool],
        sequence: Any = None,
    ) -> list[Event]:
        events: list[Event] = []
        for key in [key for key in self._buffers if predicate(key)]:
            template = self._templates.pop(key, None)
            leftover = self._pop_leftover(key)
            if not leftover or template is None:
                continue
            events.append(self._flush_event(template, leftover, sequence))
        return events

    def _flush_event(self, template: dict[str, Any], leftover: str, sequence: Any) -> Event:
        if self._dialect == "anthropic":
            field = template["_field"]
            body = {key: value for key, value in template.items() if key != "_field"}
            body["delta"] = {**template["delta"], field: leftover}
            return "content_block_delta", body
        body = {**template, "delta": leftover}
        if sequence is not None:
            body["sequence_number"] = sequence
        return str(template["type"]), body


def _responses_key(base: str, payload: dict[str, Any]) -> BufferKey:
    return (base, *(payload.get(field) for field in RESPONSES_KEY_FIELDS))


def _merge_chat_leftover(delta: dict[str, Any], key: BufferKey, leftover: str) -> None:
    field = key[2]
    if field in CHAT_TEXT_FIELDS:
        delta[field] = (delta.get(field) or "") + leftover
        return
    if field == "function_call":
        fn = delta.setdefault("function_call", {})
        fn["arguments"] = (fn.get("arguments") or "") + leftover
        return
    calls = delta.setdefault("tool_calls", [])
    for call in calls:
        if isinstance(call, dict) and call.get("index", 0) == key[3]:
            fn = call.setdefault("function", {})
            fn["arguments"] = (fn.get("arguments") or "") + leftover
            return
    calls.append({"index": key[3], "function": {"arguments": leftover}})
