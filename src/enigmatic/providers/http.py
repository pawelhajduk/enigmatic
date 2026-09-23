"""HTTP OpenAI-compatible and Anthropic passthrough."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from enigmatic.config import HttpProfile
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.protocols.sse import (
    map_anthropic_sse_to_openai,
    map_openai_sse_to_anthropic,
)
from enigmatic.protocols.stream_restore import Dialect, SseRestorer

logger = logging.getLogger("enigmatic.http")

DEFAULT_READ_TIMEOUT = 600.0
# Long read window for token streams and slow reasoning. Connect stays short so a dead
# upstream fails fast. The read default matches the OpenAI SDK's 600 s request timeout.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=DEFAULT_READ_TIMEOUT, write=30.0, pool=10.0)
# Inbound headers that carry OpenAI or Anthropic request semantics and are safe to forward.
FORWARD_REQUEST_HEADERS = {
    "openai": (
        "openai-beta",
        "openai-organization",
        "openai-project",
        "idempotency-key",
        "x-client-request-id",
        "session_id",
        "conversation_id",
        "originator",
    ),
    "anthropic": ("anthropic-beta", "anthropic-version", "idempotency-key"),
}
RETURN_RESPONSE_HEADERS = (
    "x-request-id",
    "request-id",
    "openai-processing-ms",
    "openai-version",
    "retry-after",
    "retry-after-ms",
)
RETURN_RESPONSE_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
MAX_SSE_LINE_BYTES = 1024 * 1024


class HttpProviderError(Exception):
    def __init__(self, status_code: int, body: bytes, headers: dict[str, str]) -> None:
        super().__init__(f"upstream HTTP {status_code}")
        self.status_code = status_code
        self.body = body
        self.headers = headers


def _api_key(profile: HttpProfile) -> str:
    if not profile.api_key_env:
        return ""
    return os.environ.get(profile.api_key_env, "")


def build_headers(profile: HttpProfile, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    key = _api_key(profile)
    if key:
        if profile.auth_header.lower() == "authorization":
            headers["authorization"] = f"Bearer {key}"
        elif profile.auth_header.lower() == "api-key":
            headers["api-key"] = key
        else:
            headers[profile.auth_header] = key
    if profile.type == "anthropic":
        headers["anthropic-version"] = profile.api_version or "2023-06-01"
        if key:
            headers["x-api-key"] = key
    if extra:
        for name, value in extra.items():
            lowered = name.lower()
            if lowered in {"host", "content-length", "authorization", "x-api-key", "api-key"}:
                continue
            headers[lowered] = value
    return headers


def forwardable_request_headers(headers: Any, upstream_type: str) -> dict[str, str]:
    """Client headers to pass upstream. `headers` is any case-insensitive mapping."""
    allowed = FORWARD_REQUEST_HEADERS["anthropic" if upstream_type == "anthropic" else "openai"]
    out: dict[str, str] = {}
    for name in allowed:
        value = headers.get(name)
        if value:
            out[name] = value
    return out


def returnable_response_headers(response: Any) -> dict[str, str]:
    """Upstream headers clients use for retries, rate limits, and support ids."""
    headers = getattr(response, "headers", None)
    if not headers:
        return {}
    out: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in RETURN_RESPONSE_HEADERS or lowered.startswith(RETURN_RESPONSE_HEADER_PREFIXES):
            out[lowered] = value
    return out


def join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    # OpenAI-compat bases usually already include /v1
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[3:]
    if base.endswith("/v1") and path == "/v1":
        return base
    return base + path


class HttpProvider:
    def __init__(self, profile: HttpProfile, client: httpx.AsyncClient | None = None) -> None:
        self.profile = profile
        self._client = client

    async def _client_obj(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = DEFAULT_TIMEOUT
            if self.profile.read_timeout is not None:
                timeout = httpx.Timeout(
                    connect=DEFAULT_TIMEOUT.connect,
                    read=self.profile.read_timeout,
                    write=DEFAULT_TIMEOUT.write,
                    pool=DEFAULT_TIMEOUT.pool,
                )
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        stream: bool,
        extra_headers: dict[str, str] | None = None,
        query: dict[str, str] | list[tuple[str, str]] | None = None,
    ) -> httpx.Response:
        client = await self._client_obj()
        url = join_url(self.profile.base_url, path)
        headers = build_headers(self.profile, extra_headers)
        params = list(query.items()) if isinstance(query, dict) else list(query or [])
        has_version = any(name == "api-version" for name, _value in params)
        if self.profile.api_version and not has_version and self.profile.type != "anthropic":
            params.append(("api-version", self.profile.api_version))
        logger.info("http %s %s stream=%s", method, url, stream)
        request = client.build_request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode("utf-8") if body is not None else None,
            params=params or None,
        )
        response = await client.send(request, stream=stream)
        return response

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for raw in response.aiter_bytes():
        buffer += raw.decode("utf-8", errors="replace")
        if len(buffer) > MAX_SSE_LINE_BYTES and "\n" not in buffer:
            raise HttpProviderError(502, b"upstream SSE line too large", {})
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if len(line) > MAX_SSE_LINE_BYTES:
                raise HttpProviderError(502, b"upstream SSE line too large", {})
            yield line
    if len(buffer) > MAX_SSE_LINE_BYTES:
        raise HttpProviderError(502, b"upstream SSE line too large", {})
    if buffer:
        yield buffer


async def close_response(response: httpx.Response) -> None:
    close = getattr(response, "aclose", None)
    if close is not None:
        await close()


SseItem = tuple[str | None, Any]
DONE = "[DONE]"


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[SseItem]:
    """Yield `(event_name, payload)`; payload is parsed JSON, `DONE`, or raw text."""
    event_name: str | None = None
    async for raw_line in iter_sse_lines(response):
        line = raw_line.rstrip("\r")
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == DONE:
            yield event_name, DONE
        else:
            try:
                yield event_name, json.loads(data)
            except json.JSONDecodeError:
                yield event_name, data
        event_name = None


def format_sse_item(event: str | None, payload: Any) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {data}\n\n".encode()
    return f"data: {data}\n\n".encode()


async def restored_events(
    response: httpx.Response,
    mapping: SessionMapping,
    dialect: Dialect,
    on_response_id: Callable[[str], None] | None = None,
) -> AsyncIterator[SseItem]:
    """Upstream SSE with placeholders restored per logical text stream."""
    restorer = SseRestorer(mapping, dialect, on_response_id)
    uses_events = False
    finished = False
    try:
        async for event, payload in iter_sse_events(response):
            uses_events = uses_events or event is not None
            if payload == DONE:
                for item in restorer.finish():
                    yield item
                finished = True
                yield None, DONE
                continue
            if isinstance(payload, str):
                yield event, mapping.restore_complete(payload)
                continue
            for synth_event, restored in restorer.restore(payload, event):
                yield (synth_event if uses_events else None), restored
        if not finished:
            for synth_event, restored in restorer.finish():
                yield (synth_event if uses_events else None), restored
    finally:
        await close_response(response)


async def restored_sse(
    response: httpx.Response,
    mapping: SessionMapping,
    dialect: Dialect = "chat",
    on_response_id: Callable[[str], None] | None = None,
) -> AsyncIterator[bytes]:
    async for event, payload in restored_events(response, mapping, dialect, on_response_id):
        yield format_sse_item(event, payload)


async def translate_and_restore_openai_to_anthropic(
    response: httpx.Response,
    mapping: SessionMapping,
    model: str,
) -> AsyncIterator[bytes]:
    async for chunk in map_openai_sse_to_anthropic(restored_events(response, mapping, "chat"), model):
        yield chunk


async def translate_and_restore_anthropic_to_openai(
    response: httpx.Response,
    mapping: SessionMapping,
    model: str,
    include_usage: bool = False,
) -> AsyncIterator[bytes]:
    events = restored_events(response, mapping, "anthropic")
    async for chunk in map_anthropic_sse_to_openai(events, model, include_usage=include_usage):
        yield chunk
