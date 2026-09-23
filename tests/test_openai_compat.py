"""OpenAI API surface compatibility: paths, headers, errors, streaming, sessions."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from enigmatic.config import EnigmaticConfig, HttpProfile
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.server import create_app

EMAIL = "ada@example.com"


class FakePipeline:
    def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
        if EMAIL in text:
            return text.replace(EMAIL, mapping.placeholder_for("EMAIL_ADDRESS", EMAIL))
        return text


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        sse: list[str] | None = None,
        headers: dict[str, str] | None = None,
        raw: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        if raw is not None:
            self._raw = raw
        elif sse is not None:
            self._raw = ("\n".join(sse) + "\n").encode()
        else:
            self._raw = json.dumps(body).encode()

    async def aread(self) -> bytes:
        return self._raw

    async def aiter_bytes(self) -> Any:
        for i in range(0, len(self._raw), 5):
            yield self._raw[i : i + 5]

    async def aclose(self) -> None:
        return None


class Upstream:
    """Scripted upstream: a queue of responses plus a log of requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[FakeResponse] = []

    def install(self, monkeypatch: Any) -> None:
        upstream = self

        class FakeHttp:
            def __init__(self, profile: Any) -> None:
                self.profile = profile

            async def request(self, method: str, path: str, body: Any, stream: bool, **kwargs: Any) -> FakeResponse:
                upstream.calls.append(
                    {"method": method, "path": path, "body": body, "stream": stream, "profile": self.profile, **kwargs}
                )
                return upstream.replies.pop(0)

            async def close(self) -> None:
                return None

        import enigmatic.server as server_mod

        monkeypatch.setattr(server_mod, "HttpProvider", FakeHttp)


def _client(**overrides: Any) -> TestClient:
    settings: dict[str, Any] = {
        "api_key_env": None,
        "default_profile": "openai",
        "http": {
            "openai": HttpProfile(type="openai", base_url="https://api.openai.com/v1"),
            "anthropic": HttpProfile(type="anthropic", base_url="https://api.anthropic.com"),
        },
        **overrides,
    }
    cfg = EnigmaticConfig(**settings)
    return TestClient(create_app(cfg, pipeline=FakePipeline()))  # type: ignore[arg-type]


@pytest.fixture
def upstream(monkeypatch: Any) -> Upstream:
    fake = Upstream()
    fake.install(monkeypatch)
    return fake


def _sse(events: list[dict[str, Any]], named: bool = True, done: bool = False) -> list[str]:
    lines: list[str] = []
    for event in events:
        if named:
            lines.append(f"event: {event['type']}")
        lines.append("data: " + json.dumps(event))
        lines.append("")
    if done:
        lines += ["data: [DONE]", ""]
    return lines


def _parse_sse(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def test_bare_paths_without_v1_prefix(upstream: Upstream) -> None:
    upstream.replies.append(FakeResponse(body={"id": "resp_1", "object": "response", "output": []}))
    upstream.replies.append(FakeResponse(body={"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}))
    client = _client()
    assert client.post("/responses", json={"model": "gpt-5", "input": "hi"}).status_code == 200
    assert client.post("/chat/completions", json={"model": "gpt-5", "messages": []}).status_code == 200
    assert upstream.calls[0]["path"] == "/v1/responses"
    assert upstream.calls[1]["path"] == "/v1/chat/completions"


def test_models_have_created_and_model_retrieve(upstream: Upstream) -> None:
    upstream.replies.append(FakeResponse(body={"object": "list", "data": [{"id": "gpt-5", "object": "model", "owned_by": "openai"}]}))
    upstream.replies.append(FakeResponse(body={"id": "gpt-5", "object": "model", "created": 1, "owned_by": "openai"}))
    client = _client()
    listed = client.get("/v1/models").json()["data"]
    assert all({"id", "object", "created", "owned_by"} <= set(item) for item in listed)
    assert client.get("/v1/models/openai/gpt-5").json()["id"] == "gpt-5"
    assert upstream.calls[-1]["path"] == "/v1/models/gpt-5"
    assert client.get("/models/anthropic/default").json()["owned_by"] == "anthropic"


def test_headers_forwarded_both_ways_and_upstream_errors_keep_shape(upstream: Upstream) -> None:
    upstream.replies.append(
        FakeResponse(
            429,
            body={"error": {"message": "Rate limit for <EMAIL_ADDRESS_1>", "type": "rate_limit_error", "param": None, "code": "rate_limit_exceeded"}},
            headers={"retry-after": "3", "x-request-id": "req_1", "x-ratelimit-remaining-tokens": "0", "set-cookie": "no"},
        )
    )
    client = _client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5", "messages": [{"role": "user", "content": EMAIL}]},
        headers={"OpenAI-Beta": "responses=v1", "session_id": "s1", "Cookie": "secret"},
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    assert response.headers["x-request-id"] == "req_1"
    assert response.headers["x-ratelimit-remaining-tokens"] == "0"
    assert "set-cookie" not in response.headers
    assert response.json()["error"]["message"] == f"Rate limit for {EMAIL}"
    forwarded = upstream.calls[0]["extra_headers"]
    assert forwarded["openai-beta"] == "responses=v1"
    assert forwarded["session_id"] == "s1"
    assert "cookie" not in {name.lower() for name in forwarded}


def test_non_json_upstream_error_is_wrapped_in_openai_shape(upstream: Upstream) -> None:
    upstream.replies.append(FakeResponse(502, raw=b"<html>bad gateway</html>"))
    response = _client().post("/v1/responses", json={"model": "gpt-5", "input": "hi"})
    assert response.status_code == 502
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["type"] == "server_error"


def test_unknown_path_and_client_errors_use_spec_error_shape() -> None:
    client = _client()
    missing = client.post("/v1/nope", json={})
    assert missing.status_code == 404
    assert set(missing.json()["error"]) == {"message", "type", "param", "code"}
    bad = client.post("/v1/chat/completions", content=b"[1]", headers={"content-type": "application/json"})
    assert bad.status_code == 400
    assert bad.json()["error"]["type"] == "invalid_request_error"
    anthropic_missing = client.post("/v1/messages", content=b"nope")
    assert anthropic_missing.json()["type"] == "error"


def test_responses_stream_restores_text_and_custom_tool_input(upstream: Upstream) -> None:
    events = [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_s", "status": "in_progress", "output": []}},
        {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "m", "output_index": 0, "content_index": 0, "delta": "to <EMAIL_ADDR"},
        {"type": "response.output_text.delta", "sequence_number": 2, "item_id": "m", "output_index": 0, "content_index": 0, "delta": "ESS_1>"},
        {"type": "response.custom_tool_call_input.delta", "sequence_number": 3, "item_id": "c", "output_index": 1, "delta": "+owner=<EMAIL_ADDRESS_1>"},
        {"type": "response.completed", "sequence_number": 4, "response": {"id": "resp_s", "status": "completed", "output": []}},
    ]
    upstream.replies.append(FakeResponse(sse=_sse(events)))
    response = _client().post(
        "/v1/responses",
        json={"model": "gpt-5", "stream": True, "input": [{"role": "user", "content": f"mail {EMAIL}"}]},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = _parse_sse(response.text)
    text = "".join(e["delta"] for e in parsed if e["type"] == "response.output_text.delta")
    patch = "".join(e["delta"] for e in parsed if e["type"] == "response.custom_tool_call_input.delta")
    assert text == f"to {EMAIL}"
    assert patch == f"+owner={EMAIL}"
    assert "event: response.completed" in response.text


def test_previous_response_id_reuses_the_vault(upstream: Upstream) -> None:
    upstream.replies.append(FakeResponse(body={"id": "resp_a", "object": "response", "output": []}))
    upstream.replies.append(
        FakeResponse(body={"id": "resp_b", "object": "response", "output": [{"type": "message", "content": [{"type": "output_text", "text": "<EMAIL_ADDRESS_1>"}]}]})
    )
    upstream.replies.append(FakeResponse(body={"id": "resp_a", "object": "response", "output": [{"type": "message", "content": [{"type": "output_text", "text": "<EMAIL_ADDRESS_1>"}]}]}))
    client = _client()
    client.post("/v1/responses", json={"model": "gpt-5", "input": f"remember {EMAIL}"})
    # The second turn never mentions the email; only the stored upstream history does.
    second = client.post("/v1/responses", json={"model": "gpt-5", "previous_response_id": "resp_a", "input": "what was it?"})
    assert second.json()["output"][0]["content"][0]["text"] == EMAIL
    fetched = client.get("/v1/responses/resp_a", params={"include[]": "reasoning.encrypted_content"})
    assert fetched.json()["output"][0]["content"][0]["text"] == EMAIL
    assert upstream.calls[-1]["method"] == "GET"
    assert upstream.calls[-1]["path"] == "/v1/responses/resp_a"
    assert ("include[]", "reasoning.encrypted_content") in upstream.calls[-1]["query"]


def test_compact_and_input_tokens_pass_through(upstream: Upstream) -> None:
    upstream.replies.append(
        FakeResponse(body={"id": "cmp_1", "object": "response.compaction", "output": [{"type": "compaction", "encrypted_content": "gAAA"}], "usage": {}})
    )
    upstream.replies.append(FakeResponse(body={"object": "response.input_tokens", "input_tokens": 12}))
    client = _client()
    compact = client.post("/v1/responses/compact", json={"model": "gpt-5", "input": [{"role": "user", "content": EMAIL}]})
    assert compact.status_code == 200
    assert compact.json()["output"][0]["encrypted_content"] == "gAAA"
    assert "<EMAIL_ADDRESS_1>" in json.dumps(upstream.calls[0]["body"])
    assert upstream.calls[0]["path"] == "/v1/responses/compact"
    tokens = client.post("/responses/input_tokens", json={"model": "gpt-5", "input": "hi"})
    assert tokens.json()["input_tokens"] == 12
    unsupported = client.post("/v1/responses/compact", json={"model": "anthropic/claude-sonnet-4-5", "input": "hi"})
    assert unsupported.status_code == 501


def test_chat_to_anthropic_stream_carries_tool_calls_and_usage(upstream: Upstream) -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "cache_read_input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Writing <EMAIL_ADDRESS_1>"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "write", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"to":"<EMAIL_'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'ADDRESS_1>"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    upstream.replies.append(FakeResponse(sse=_sse(events)))
    response = _client().post(
        "/v1/chat/completions",
        json={
            "model": "anthropic/claude-sonnet-4-5",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": EMAIL}],
            "tools": [{"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}}],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        },
    )
    sent = upstream.calls[0]["body"]
    assert sent["system"] == "rules"
    assert sent["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    chunks = _parse_sse(response.text)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    args = "".join(
        call["function"].get("arguments") or ""
        for c in chunks
        if c.get("choices")
        for call in c["choices"][0]["delta"].get("tool_calls") or []
    )
    assert content == f"Writing {EMAIL}"
    assert json.loads(args) == {"to": EMAIL}
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and c["choices"][0]["finish_reason"]]
    assert finishes == ["tool_calls"]
    usage = next(c["usage"] for c in chunks if c.get("usage"))
    assert usage["prompt_tokens"] == 15 and usage["prompt_tokens_details"]["cached_tokens"] == 5
    assert response.text.rstrip().endswith("data: [DONE]")


def test_responses_to_anthropic_stream_emits_full_lifecycle(upstream: Upstream) -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "apply_patch", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"input":"+ <EMAIL_ADDRESS_1>"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    upstream.replies.append(FakeResponse(sse=_sse(events)))
    response = _client().post(
        "/v1/responses",
        json={
            "model": "anthropic/claude-sonnet-4-5",
            "stream": True,
            "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}}],
            "input": [
                {"type": "function_call", "call_id": "call_0", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_0", "output": "ok"},
                {"role": "user", "content": [{"type": "input_text", "text": f"patch {EMAIL}"}]},
            ],
        },
    )
    sent = upstream.calls[0]["body"]
    assert sent["messages"][0]["content"][0]["type"] == "tool_use"
    assert sent["messages"][1]["content"][0]["type"] == "tool_result"
    assert sent["tools"][0]["name"] == "apply_patch"
    parsed = _parse_sse(response.text)
    kinds = [event["type"] for event in parsed]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert "response.output_item.added" in kinds and kinds[-1] == "response.completed"
    assert [event["sequence_number"] for event in parsed] == list(range(len(parsed)))
    done = next(event for event in parsed if event["type"] == "response.custom_tool_call_input.done")
    assert done["input"] == f"+ {EMAIL}"
    item = parsed[-1]["response"]["output"][0]
    assert item["type"] == "custom_tool_call" and item["call_id"] == "toolu_1"


def test_responses_to_anthropic_rejects_previous_response_id() -> None:
    response = _client().post(
        "/v1/responses",
        json={"model": "anthropic/claude-sonnet-4-5", "previous_response_id": "resp_x", "input": "hi"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "previous_response_id"


def test_mid_stream_failure_ends_with_an_error_event(upstream: Upstream, monkeypatch: Any) -> None:
    class Broken(FakeResponse):
        async def aiter_bytes(self) -> Any:
            yield b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            raise RuntimeError("connection reset")

    upstream.replies.append(Broken(sse=[]))
    response = _client().post("/v1/chat/completions", json={"model": "gpt-5", "stream": True, "messages": []})
    parsed = _parse_sse(response.text)
    assert parsed[0]["choices"][0]["delta"]["content"] == "hi"
    assert parsed[-1]["error"]["type"] == "server_error"
    assert "connection reset" not in response.text


def test_anthropic_x_api_key_passes_the_gate(monkeypatch: Any, upstream: Upstream) -> None:
    monkeypatch.setenv("ENIGMATIC_TEST_KEY", "sk-local")
    upstream.replies.append(FakeResponse(body={"id": "msg", "type": "message", "content": [], "usage": {}}))
    client = _client(api_key_env="ENIGMATIC_TEST_KEY")
    denied = client.post("/v1/messages", json={"model": "anthropic/claude", "messages": []})
    assert denied.status_code == 401 and denied.json()["type"] == "error"
    allowed = client.post(
        "/v1/messages",
        json={"model": "anthropic/claude", "messages": []},
        headers={"x-api-key": "sk-local", "anthropic-beta": "tools-2024"},
    )
    assert allowed.status_code == 200
    assert upstream.calls[0]["extra_headers"] == {"anthropic-beta": "tools-2024"}
