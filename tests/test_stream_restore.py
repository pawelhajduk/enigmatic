from __future__ import annotations

import json
from typing import Any

import pytest

from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.protocols.stream_restore import SseRestorer
from enigmatic.providers.http import restored_sse

PEM = "-----BEGIN PRIVATE KEY-----\nabc\"def\n-----END PRIVATE KEY-----"


def _mapping() -> SessionMapping:
    mapping = SessionMapping()
    assert mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com") == "<EMAIL_ADDRESS_1>"
    assert mapping.placeholder_for("PEM_KEY", PEM) == "<PEM_KEY_1>"
    return mapping


def _chat_chunk(delta: dict[str, Any], finish: str | None = None, index: int = 0) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }


def _run(restorer: SseRestorer, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in payloads:
        out.extend(item for _event, item in restorer.restore(payload))
    out.extend(item for _event, item in restorer.finish())
    return out


def _chat_text(chunks: list[dict[str, Any]], field: str = "content") -> str:
    return "".join(
        choice["delta"].get(field) or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )


def _chat_arguments(chunks: list[dict[str, Any]], index: int) -> str:
    return "".join(
        (call.get("function") or {}).get("arguments") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for call in choice["delta"].get("tool_calls") or []
        if call.get("index") == index
    )


def test_chat_split_tokens_do_not_cross_fields_or_tool_calls() -> None:
    restorer = SseRestorer(_mapping(), "chat")
    chunks = _run(
        restorer,
        [
            _chat_chunk({"role": "assistant", "content": "Mail <EMAIL_"}),
            _chat_chunk({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "write", "arguments": '{"to":"<EMAIL'}}]}),
            _chat_chunk({"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "save", "arguments": '{"key":"<PEM_'}}]}),
            _chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '_ADDRESS_1>"}'}}]}),
            _chat_chunk({"tool_calls": [{"index": 1, "function": {"arguments": 'KEY_1>"}'}}]}),
            _chat_chunk({"content": "ADDRESS_1> done"}),
            _chat_chunk({}, finish="tool_calls"),
        ],
    )
    assert _chat_text(chunks) == "Mail ada@example.com done"
    assert json.loads(_chat_arguments(chunks, 0)) == {"to": "ada@example.com"}
    assert json.loads(_chat_arguments(chunks, 1)) == {"key": PEM}


def test_chat_trailing_partial_is_flushed_into_the_finish_chunk() -> None:
    restorer = SseRestorer(_mapping(), "chat")
    chunks = _run(restorer, [_chat_chunk({"content": "a < b and <X"}), _chat_chunk({}, finish="stop")])
    assert _chat_text(chunks) == "a < b and <X"
    assert all(isinstance(chunk, dict) and "choices" in chunk for chunk in chunks)


def test_chat_reasoning_content_is_restored() -> None:
    restorer = SseRestorer(_mapping(), "chat")
    chunks = _run(
        restorer,
        [_chat_chunk({"reasoning_content": "user <EMAIL_ADD"}), _chat_chunk({"reasoning_content": "RESS_1>"}), _chat_chunk({}, "stop")],
    )
    assert _chat_text(chunks, "reasoning_content") == "user ada@example.com"


def _resp(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, **fields}


def test_responses_deltas_done_events_and_snapshots_are_restored() -> None:
    bound: list[str] = []
    restorer = SseRestorer(_mapping(), "responses", on_response_id=bound.append)
    text_loc = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    patch_loc = {"item_id": "ctc_1", "output_index": 1}
    call_loc = {"item_id": "fc_1", "output_index": 2}
    events = _run(
        restorer,
        [
            _resp("response.created", response={"id": "resp_1", "status": "in_progress", "output": []}),
            _resp("response.output_text.delta", delta="Hi <EMAIL_ADD", **text_loc),
            _resp("response.custom_tool_call_input.delta", delta="+ owner: <EMAIL", **patch_loc),
            _resp("response.output_text.delta", delta="RESS_1>", **text_loc),
            _resp("response.output_text.done", text="Hi <EMAIL_ADDRESS_1>", **text_loc),
            _resp("response.custom_tool_call_input.delta", delta="_ADDRESS_1>\n", **patch_loc),
            _resp("response.custom_tool_call_input.done", input="+ owner: <EMAIL_ADDRESS_1>\n", **patch_loc),
            _resp("response.function_call_arguments.delta", delta='{"k":"<PEM_KEY', **call_loc),
            _resp("response.function_call_arguments.delta", delta='_1>","t":"<X', **call_loc),
            _resp("response.function_call_arguments.done", arguments='{"k":"<PEM_KEY_1>","t":"<X"}', **call_loc),
            _resp(
                "response.completed",
                response={
                    "id": "resp_1",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "Hi <EMAIL_ADDRESS_1>"}]},
                        {"type": "function_call", "arguments": '{"k":"<PEM_KEY_1>"}'},
                    ],
                },
            ),
        ],
    )
    assert bound == ["resp_1"]

    def deltas(kind: str) -> str:
        return "".join(event["delta"] for event in events if event["type"] == kind)

    assert deltas("response.output_text.delta") == "Hi ada@example.com"
    assert deltas("response.custom_tool_call_input.delta") == "+ owner: ada@example.com\n"
    assert deltas("response.function_call_arguments.delta") == '{"k":' + json.dumps(PEM) + ',"t":"<X'
    done = next(event for event in events if event["type"] == "response.function_call_arguments.done")
    assert json.loads(done["arguments"])["k"] == PEM
    kinds = [event["type"] for event in events]
    # The held "<X" is flushed as a delta right before its done event.
    assert kinds.index("response.function_call_arguments.done") - 1 == max(
        i for i, kind in enumerate(kinds) if kind == "response.function_call_arguments.delta"
    )
    completed = events[-1]["response"]["output"]
    assert completed[0]["content"][0]["text"] == "Hi ada@example.com"
    assert json.loads(completed[1]["arguments"]) == {"k": PEM}


def test_anthropic_input_json_delta_is_json_escaped() -> None:
    restorer = SseRestorer(_mapping(), "anthropic")
    events = _run(
        restorer,
        [
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"k":"<PEM_'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'KEY_1>"}'}},
            {"type": "content_block_stop", "index": 1},
        ],
    )
    joined = "".join(event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta")
    assert json.loads(joined) == {"k": PEM}


class _StreamResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, lines: list[str]) -> None:
        self._body = ("\n".join(lines) + "\n").encode()

    async def aiter_bytes(self) -> Any:
        for i in range(0, len(self._body), 7):
            yield self._body[i : i + 7]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_restored_sse_keeps_event_lines_and_never_emits_bare_strings() -> None:
    mapping = _mapping()
    lines = [
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","item_id":"m","output_index":0,"content_index":0,"delta":"x <EMAIL_ADD"}',
        "",
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","item_id":"m","output_index":0,"content_index":0,"delta":"RESS_1> <Y"}',
        "",
    ]
    raw = b"".join([chunk async for chunk in restored_sse(_StreamResponse(lines), mapping, "responses")])  # type: ignore[arg-type]
    blocks = [block for block in raw.decode().split("\n\n") if block]
    payloads = []
    for block in blocks:
        event_line, data_line = block.split("\n")
        assert event_line.startswith("event: response.output_text.delta")
        payloads.append(json.loads(data_line[len("data: ") :]))
    assert all(isinstance(payload, dict) for payload in payloads)
    assert "".join(payload["delta"] for payload in payloads) == "x ada@example.com <Y"


def test_legacy_completions_stream_text_is_restored() -> None:
    restorer = SseRestorer(_mapping(), "chat")
    chunks = _run(
        restorer,
        [
            {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "to <EMAIL_ADD", "finish_reason": None}]},
            {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "RESS_1> <Z", "finish_reason": "stop"}]},
        ],
    )
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "to ada@example.com <Z"
    assert all("delta" not in chunk["choices"][0] for chunk in chunks)


@pytest.mark.asyncio
async def test_multibyte_characters_split_across_network_chunks_survive() -> None:
    from enigmatic.providers.http import iter_sse_lines

    class OneByteAtATime:
        async def aiter_bytes(self) -> Any:
            data = 'data: {"t":"café 日本 🎉"}\n'.encode()
            for i in range(len(data)):
                yield data[i : i + 1]

    lines = [line async for line in iter_sse_lines(OneByteAtATime())]  # type: ignore[arg-type]
    assert lines == ['data: {"t":"café 日本 🎉"}']


def test_non_streaming_restore_escapes_json_arguments() -> None:
    mapping = _mapping()
    restored = mapping.restore_complete_any(
        {"tool_calls": [{"function": {"arguments": '{"k":"<PEM_KEY_1>"}'}}], "content": "<PEM_KEY_1>"}
    )
    assert isinstance(restored, dict)
    assert json.loads(restored["tool_calls"][0]["function"]["arguments"]) == {"k": PEM}
    assert restored["content"] == PEM
