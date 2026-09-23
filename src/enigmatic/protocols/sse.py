"""SSE helpers, text-to-stream adapters, and streaming protocol translation."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from enigmatic.protocols.translate import (
    anthropic_stop_reason,
    custom_tool_input,
    custom_tool_names,
    openai_finish_reason,
    openai_usage_from_anthropic,
    responses_usage_from_chat,
)

DONE = "[DONE]"
SseItem = tuple[str | None, Any]


def format_sse(payload: dict[str, Any], event: str | None = None) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def openai_chat_stream_from_text(text: str, model: str) -> Iterator[str]:
    for chunk in chat_chunks_from_text(text, model):
        yield format_sse(chunk)
    yield f"data: {DONE}\n\n"


def chat_chunks_from_text(text: str, model: str) -> Iterator[dict[str, Any]]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model}
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    step = 48
    for i in range(0, len(text), step):
        piece = text[i : i + step]
        yield {**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def anthropic_stream_from_text(text: str, model: str) -> Iterator[str]:
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    yield format_sse(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        event="message_start",
    )
    yield format_sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        event="content_block_start",
    )
    step = 48
    for i in range(0, len(text), step):
        yield format_sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[i : i + step]}},
            event="content_block_delta",
        )
    yield format_sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    yield format_sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
        event="message_delta",
    )
    yield format_sse({"type": "message_stop"}, event="message_stop")


def responses_stream_from_text(text: str, model: str) -> Iterator[str]:
    builder = ResponsesStreamBuilder(model)
    for chunk in chat_chunks_from_text(text, model):
        for event, payload in builder.feed(chunk):
            yield format_sse(payload, event=event)
    for event, payload in builder.finish():
        yield format_sse(payload, event=event)


def _error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "upstream error")
    return str(error or "upstream error")


# OpenAI Chat chunks -> Anthropic Messages events


async def map_openai_sse_to_anthropic(events: AsyncIterator[SseItem], model: str) -> AsyncIterator[bytes]:
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    started = False
    block_index = -1
    open_block: tuple[str, Any] | None = None
    tool_blocks: dict[Any, int] = {}
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}

    def emit(payload: dict[str, Any]) -> bytes:
        return format_sse(payload, event=str(payload["type"])).encode()

    def start_message() -> bytes:
        return emit(
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": dict(usage),
                },
            }
        )

    def close_block() -> Iterator[bytes]:
        nonlocal open_block
        if open_block is not None:
            yield emit({"type": "content_block_stop", "index": block_index})
            open_block = None

    async for _event, payload in events:
        if payload == DONE:
            break
        if not isinstance(payload, dict):
            continue
        if "error" in payload and "choices" not in payload:
            if not started:
                yield start_message()
            yield emit({"type": "error", "error": {"type": "api_error", "message": _error_message(payload)}})
            return
        if not started:
            started = True
            yield start_message()
        if isinstance(payload.get("usage"), dict):
            usage["input_tokens"] = int(payload["usage"].get("prompt_tokens") or 0)
            usage["output_tokens"] = int(payload["usage"].get("completion_tokens") or 0)
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            if open_block is None or open_block[0] != "text":
                for chunk in close_block():
                    yield chunk
                block_index += 1
                open_block = ("text", None)
                yield emit(
                    {"type": "content_block_start", "index": block_index, "content_block": {"type": "text", "text": ""}}
                )
            yield emit(
                {"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": content}}
            )
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            key = call.get("index", 0)
            fn = call.get("function") or {}
            if key not in tool_blocks:
                for chunk in close_block():
                    yield chunk
                block_index += 1
                tool_blocks[key] = block_index
                open_block = ("tool", key)
                yield emit(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": str(fn.get("name") or ""),
                            "input": {},
                        },
                    }
                )
            arguments = fn.get("arguments")
            if isinstance(arguments, str) and arguments:
                yield emit(
                    {
                        "type": "content_block_delta",
                        "index": tool_blocks[key],
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    }
                )
        finish = choice.get("finish_reason")
        if finish:
            stop_reason = anthropic_stop_reason(str(finish))
    if not started:
        yield start_message()
    for chunk in close_block():
        yield chunk
    yield emit(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
        }
    )
    yield emit({"type": "message_stop"})


# Anthropic Messages events -> OpenAI Chat chunks


async def anthropic_events_to_chat_chunks(
    events: AsyncIterator[SseItem],
    model: str,
    include_usage: bool = False,
) -> AsyncIterator[dict[str, Any] | str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    tool_index: dict[Any, int] = {}
    finish = "stop"
    anthropic_usage: dict[str, Any] = {}
    finished = False

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def closing() -> Iterator[dict[str, Any] | str]:
        yield chunk({}, finish)
        if include_usage:
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": openai_usage_from_anthropic(anthropic_usage),
            }
        yield DONE

    async for _event, payload in events:
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "message_start":
            message = payload.get("message") or {}
            anthropic_usage.update(message.get("usage") or {})
            yield chunk({"role": "assistant", "content": ""})
        elif kind == "content_block_start":
            block = payload.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = len(tool_index)
                tool_index[payload.get("index", 0)] = index
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": str(block.get("id") or ""),
                                "type": "function",
                                "function": {"name": str(block.get("name") or ""), "arguments": ""},
                            }
                        ]
                    }
                )
            elif block.get("type") == "text" and block.get("text"):
                yield chunk({"content": str(block["text"])})
        elif kind == "content_block_delta":
            delta = payload.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta" and delta.get("text"):
                yield chunk({"content": str(delta["text"])})
            elif delta_type == "input_json_delta" and delta.get("partial_json"):
                index = tool_index.get(payload.get("index", 0), 0)
                yield chunk(
                    {"tool_calls": [{"index": index, "function": {"arguments": str(delta["partial_json"])}}]}
                )
            elif delta_type == "thinking_delta" and delta.get("thinking"):
                yield chunk({"reasoning_content": str(delta["thinking"])})
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            if delta.get("stop_reason"):
                finish = openai_finish_reason(str(delta["stop_reason"]))
            anthropic_usage.update(payload.get("usage") or {})
        elif kind == "message_stop":
            finished = True
            for item in closing():
                yield item
            return
        elif kind == "error":
            error = payload.get("error") or {}
            yield {
                "error": {
                    "message": str(error.get("message") or "upstream error"),
                    "type": str(error.get("type") or "server_error"),
                    "param": None,
                    "code": None,
                }
            }
            return
    if not finished:
        for item in closing():
            yield item


async def map_anthropic_sse_to_openai(
    events: AsyncIterator[SseItem],
    model: str,
    include_usage: bool = False,
) -> AsyncIterator[bytes]:
    async for item in anthropic_events_to_chat_chunks(events, model, include_usage):
        if item == DONE:
            yield f"data: {DONE}\n\n".encode()
        elif isinstance(item, dict):
            yield format_sse(item).encode()


# OpenAI Chat chunks -> OpenAI Responses events


class ResponsesStreamBuilder:
    """Turn Chat Completion chunks into the Responses streaming event lifecycle."""

    def __init__(self, model: str, request: dict[str, Any] | None = None) -> None:
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.model = model
        self.created_at = int(time.time())
        self._request = request or {}
        self._custom = custom_tool_names(self._request)
        self._sequence = 0
        self._started = False
        self._output: list[dict[str, Any]] = []
        self._message: dict[str, Any] | None = None
        self._message_text = ""
        self._calls: dict[Any, dict[str, Any]] = {}
        self._call_order: list[Any] = []
        self._finish: str | None = None
        self._usage: dict[str, Any] | None = None
        self._failed: dict[str, Any] | None = None

    def _event(self, payload: dict[str, Any]) -> SseItem:
        payload["sequence_number"] = self._sequence
        self._sequence += 1
        return str(payload["type"]), payload

    def _response(self, status: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": list(self._output) if status != "in_progress" else [],
            "parallel_tool_calls": bool(self._request.get("parallel_tool_calls", True)),
            "tool_choice": self._request.get("tool_choice", "auto"),
            "tools": self._request.get("tools") or [],
            "error": self._failed,
            "incomplete_details": None,
            "usage": None,
        }
        if status == "incomplete":
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        if status != "in_progress":
            body["usage"] = responses_usage_from_chat(self._usage or {})
        return body

    def _start(self) -> list[SseItem]:
        if self._started:
            return []
        self._started = True
        return [
            self._event({"type": "response.created", "response": self._response("in_progress")}),
            self._event({"type": "response.in_progress", "response": self._response("in_progress")}),
        ]

    def feed(self, chunk: dict[str, Any]) -> list[SseItem]:
        events = self._start()
        if "error" in chunk and "choices" not in chunk:
            self._failed = {"code": "server_error", "message": _error_message(chunk)}
            return events
        if isinstance(chunk.get("usage"), dict):
            self._usage = chunk["usage"]
        choices = chunk.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            events.extend(self._text_delta(content))
        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                events.extend(self._call_delta(call))
        if choice.get("finish_reason"):
            self._finish = str(choice["finish_reason"])
        return events

    def _text_delta(self, text: str) -> list[SseItem]:
        events: list[SseItem] = []
        if self._message is None:
            self._message = {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            self._message["_index"] = len(self._output) + len(self._calls)
            events.append(
                self._event(
                    {
                        "type": "response.output_item.added",
                        "output_index": self._message["_index"],
                        "item": _public(self._message),
                    }
                )
            )
            events.append(
                self._event(
                    {
                        "type": "response.content_part.added",
                        "item_id": self._message["id"],
                        "output_index": self._message["_index"],
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
            )
        self._message_text += text
        events.append(
            self._event(
                {
                    "type": "response.output_text.delta",
                    "item_id": self._message["id"],
                    "output_index": self._message["_index"],
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
        )
        return events

    def _call_delta(self, call: dict[str, Any]) -> list[SseItem]:
        events: list[SseItem] = []
        key = call.get("index", 0)
        fn = call.get("function") or {}
        item = self._calls.get(key)
        if item is None:
            events.extend(self._close_message())
            name = str(fn.get("name") or "")
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
            index = len(self._output) + len(self._calls)
            if name in self._custom:
                item = {"id": f"ctc_{uuid.uuid4().hex}", "type": "custom_tool_call", "status": "in_progress"}
                item.update({"call_id": call_id, "name": name, "input": "", "_arguments": ""})
            else:
                item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "in_progress"}
                item.update({"call_id": call_id, "name": name, "arguments": ""})
            item["_index"] = index
            self._calls[key] = item
            self._call_order.append(key)
            events.append(
                self._event(
                    {"type": "response.output_item.added", "output_index": item["_index"], "item": _public(item)}
                )
            )
        arguments = fn.get("arguments")
        if isinstance(arguments, str) and arguments and item["type"] == "custom_tool_call":
            item["_arguments"] += arguments
        elif isinstance(arguments, str) and arguments:
            item["arguments"] += arguments
            events.append(
                self._event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item["id"],
                        "output_index": item["_index"],
                        "delta": arguments,
                    }
                )
            )
        return events

    def _close_message(self) -> list[SseItem]:
        message = self._message
        if message is None:
            return []
        self._message = None
        part = {"type": "output_text", "text": self._message_text, "annotations": []}
        message["content"] = [part]
        message["status"] = "completed"
        index = message["_index"]
        self._output.append(_public(message))
        text = self._message_text
        self._message_text = ""
        return [
            self._event(
                {
                    "type": "response.output_text.done",
                    "item_id": message["id"],
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                }
            ),
            self._event(
                {
                    "type": "response.content_part.done",
                    "item_id": message["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": part,
                }
            ),
            self._event({"type": "response.output_item.done", "output_index": index, "item": _public(message)}),
        ]

    def _close_calls(self) -> list[SseItem]:
        events: list[SseItem] = []
        for key in self._call_order:
            item = self._calls[key]
            item["status"] = "completed"
            if item["type"] == "custom_tool_call":
                item["input"] = custom_tool_input(item["_arguments"] or "{}")
                location = {"item_id": item["id"], "output_index": item["_index"]}
                events.append(
                    self._event({"type": "response.custom_tool_call_input.delta", **location, "delta": item["input"]})
                )
                events.append(
                    self._event({"type": "response.custom_tool_call_input.done", **location, "input": item["input"]})
                )
            else:
                events.append(
                    self._event(
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": item["id"],
                            "output_index": item["_index"],
                            "name": item["name"],
                            "arguments": item["arguments"],
                        }
                    )
                )
            events.append(
                self._event(
                    {"type": "response.output_item.done", "output_index": item["_index"], "item": _public(item)}
                )
            )
            self._output.append(_public(item))
        self._calls = {}
        self._call_order = []
        return events

    def finish(self) -> list[SseItem]:
        events = self._start()
        events.extend(self._close_message())
        events.extend(self._close_calls())
        if self._failed is not None:
            events.append(self._event({"type": "response.failed", "response": self._response("failed")}))
        elif self._finish == "length":
            events.append(self._event({"type": "response.incomplete", "response": self._response("incomplete")}))
        else:
            events.append(self._event({"type": "response.completed", "response": self._response("completed")}))
        return events


def _public(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


async def responses_events_from_chat_chunks(
    chunks: AsyncIterator[dict[str, Any] | str] | Iterable[dict[str, Any] | str],
    model: str,
    request: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    builder = ResponsesStreamBuilder(model, request)
    if isinstance(chunks, AsyncIterator):
        async for chunk in chunks:
            if isinstance(chunk, dict):
                for event, payload in builder.feed(chunk):
                    yield format_sse(payload, event=event).encode()
    else:
        for chunk in chunks:
            if isinstance(chunk, dict):
                for event, payload in builder.feed(chunk):
                    yield format_sse(payload, event=event).encode()
    for event, payload in builder.finish():
        yield format_sse(payload, event=event).encode()
