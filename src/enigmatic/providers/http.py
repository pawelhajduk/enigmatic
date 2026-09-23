"""HTTP OpenAI-compatible and Anthropic passthrough."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from enigmatic.config import HttpProfile
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.protocols.sse import (
    map_anthropic_sse_to_openai,
    map_openai_sse_to_anthropic,
)
from enigmatic.restore import StreamRestorer

logger = logging.getLogger("enigmatic.http")

# Long read window for token streams. Connect stays short so a dead upstream fails fast.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
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
            if lowered in {"host", "content-length", "authorization", "x-api-key"}:
                continue
            headers[name] = value
    return headers


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
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        stream: bool,
        extra_headers: dict[str, str] | None = None,
        query: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = await self._client_obj()
        url = join_url(self.profile.base_url, path)
        headers = build_headers(self.profile, extra_headers)
        params = dict(query or {})
        if self.profile.api_version and "api-version" not in params and self.profile.type != "anthropic":
            params["api-version"] = self.profile.api_version
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


_STREAM_TEXT_KEYS = {"content", "text", "arguments"}


def restore_sse_data_line(line: str, restorer: StreamRestorer) -> str:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return line if line.endswith("\n") else line + "\n"
    payload = stripped[5:].strip()
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    try:
        parsed: Any = json.loads(payload)
    except json.JSONDecodeError:
        restored = restorer.feed(payload)
        return f"data: {restored}\n\n"
    restored_obj = _restore_stream_text_fields(parsed, restorer)
    return f"data: {json.dumps(restored_obj, ensure_ascii=False)}\n\n"


def _restore_stream_text_fields(value: Any, restorer: StreamRestorer) -> Any:
    """Restore only token-stream fields so ids and roles are not mixed into the buffer."""
    if isinstance(value, list):
        return [_restore_stream_text_fields(item, restorer) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _STREAM_TEXT_KEYS and isinstance(item, str):
                out[key] = restorer.feed(item)
            else:
                out[key] = _restore_stream_text_fields(item, restorer)
        return out
    return value


async def restored_sse(
    response: httpx.Response,
    mapping: SessionMapping,
) -> AsyncIterator[bytes]:
    restorer = StreamRestorer(mapping)
    event_prefix = ""
    try:
        async for line in iter_sse_lines(response):
            if line.startswith("event:"):
                event_prefix = line if line.endswith("\n") else line + "\n"
                continue
            if line.startswith("data:"):
                out = restore_sse_data_line(line, restorer)
                if event_prefix:
                    yield (event_prefix + out).encode("utf-8")
                    event_prefix = ""
                else:
                    yield out.encode("utf-8")
                continue
            if not line.strip():
                continue
        leftover = restorer.flush()
        if leftover:
            yield f"data: {json.dumps(leftover, ensure_ascii=False)}\n\n".encode()
    finally:
        await close_response(response)


async def translate_and_restore_openai_to_anthropic(
    response: httpx.Response,
    mapping: SessionMapping,
    model: str,
) -> AsyncIterator[bytes]:
    restorer = StreamRestorer(mapping)

    async def restored_lines() -> AsyncIterator[str]:
        try:
            async for line in iter_sse_lines(response):
                if line.startswith("data:"):
                    yield restore_sse_data_line(line, restorer).rstrip("\n")
                else:
                    yield line
        finally:
            await close_response(response)

    async for chunk in map_openai_sse_to_anthropic(restored_lines(), model):
        yield chunk


async def translate_and_restore_anthropic_to_openai(
    response: httpx.Response,
    mapping: SessionMapping,
    model: str,
) -> AsyncIterator[bytes]:
    restorer = StreamRestorer(mapping)

    async def restored_lines() -> AsyncIterator[str]:
        try:
            async for line in iter_sse_lines(response):
                if line.startswith("data:"):
                    yield restore_sse_data_line(line, restorer).rstrip("\n")
                else:
                    yield line
        finally:
            await close_response(response)

    async for chunk in map_anthropic_sse_to_openai(restored_lines(), model):
        yield chunk
