"""SSE helpers and text-to-stream adapters."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any


def format_sse(payload: dict[str, Any], event: str | None = None) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def openai_chat_stream_from_text(text: str, model: str) -> Iterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    yield format_sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    step = 48
    for i in range(0, len(text), step):
        piece = text[i : i + step]
        yield format_sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
        )
    yield format_sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


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
    yield format_sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, event="message_delta")
    yield format_sse({"type": "message_stop"}, event="message_stop")


def responses_stream_from_text(text: str, model: str) -> Iterator[str]:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    yield format_sse({"type": "response.created", "response": {"id": response_id, "model": model, "status": "in_progress"}})
    yield format_sse({"type": "response.output_text.delta", "delta": text})
    yield format_sse({"type": "response.completed", "response": {"id": response_id, "status": "completed"}})


async def map_openai_sse_to_anthropic(lines: AsyncIterator[str], model: str) -> AsyncIterator[bytes]:
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    started = False
    async for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = ((payload.get("choices") or [{}])[0].get("delta") or {})
        content = delta.get("content")
        if not started:
            started = True
            yield format_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                    },
                },
                event="message_start",
            ).encode()
            yield format_sse(
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                event="content_block_start",
            ).encode()
        if isinstance(content, str) and content:
            yield format_sse(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}},
                event="content_block_delta",
            ).encode()
    if started:
        yield format_sse({"type": "content_block_stop", "index": 0}, event="content_block_stop").encode()
        yield format_sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, event="message_delta").encode()
        yield format_sse({"type": "message_stop"}, event="message_stop").encode()


async def map_anthropic_sse_to_openai(lines: AsyncIterator[str], model: str) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    role_sent = False
    async for line in lines:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        kind = payload.get("type")
        if kind == "content_block_delta":
            text = (payload.get("delta") or {}).get("text", "")
            if not role_sent:
                role_sent = True
                yield format_sse(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                    }
                ).encode()
            elif text:
                yield format_sse(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                ).encode()
        elif kind == "message_stop":
            yield format_sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ).encode()
            yield b"data: [DONE]\n\n"
