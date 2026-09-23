"""FastAPI app: inbound OpenAI/Anthropic surface, anonymize, route, restore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from enigmatic import __version__
from enigmatic.auth import openai_auth_error, token_from_authorization, tokens_equal
from enigmatic.config import EnigmaticConfig, load_config
from enigmatic.errors import (
    anthropic_error_body,
    error_message,
    is_anthropic_error,
    is_openai_error,
    openai_error_body,
    openai_error_type,
    stream_error_bytes,
)
from enigmatic.openai_walk import walk
from enigmatic.presidio_ops.mapping import STORE, SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline, build_pipeline
from enigmatic.protocols.packing import pack_anthropic, pack_openai_chat, pack_responses
from enigmatic.protocols.sse import (
    anthropic_events_to_chat_chunks,
    anthropic_stream_from_text,
    openai_chat_stream_from_text,
    responses_events_from_chat_chunks,
    responses_stream_from_text,
)
from enigmatic.protocols.translate import (
    anthropic_message_from_openai,
    anthropic_message_from_text,
    anthropic_to_openai_chat,
    openai_completion_from_anthropic,
    openai_completion_from_text,
    openai_to_anthropic,
    responses_from_chat_completion,
    responses_input_to_messages,
    responses_output_from_text,
    strip_model_prefix,
)
from enigmatic.providers.http import (
    HttpProvider,
    HttpProviderError,
    forwardable_request_headers,
    restored_events,
    restored_sse,
    returnable_response_headers,
    translate_and_restore_anthropic_to_openai,
    translate_and_restore_openai_to_anthropic,
)
from enigmatic.providers.invoke import invoke_agent
from enigmatic.providers.router import Route, RouteError, Router

logger = logging.getLogger("enigmatic.server")

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_AGENT_CONCURRENCY = 2
_SESSION_HEADER_MAX = 256
# Client headers that identify one conversation. Used as an implicit vault key.
_IMPLICIT_SESSION_HEADERS = ("x-enigmatic-session", "session_id", "conversation_id")
_STREAM_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


class PayloadTooLarge(Exception):
    pass


class AgentBusy(Exception):
    pass


class ClientError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.param = param


class UpstreamPayloadError(Exception):
    pass


def session_store_key(
    authorization: str | None,
    session_header: str | None,
    *,
    gate_enabled: bool,
) -> str | None:
    """Opaque map key. None means a one-request map that is not stored.

    An authenticated caller is namespaced by their token, so a shared
    `X-Enigmatic-Session` cannot cross credentials. With the gate off and no
    session header, requests do not share the old `"default"` vault.
    """
    client = (session_header or "").strip()[:_SESSION_HEADER_MAX]
    token = token_from_authorization(authorization)
    if gate_enabled and token:
        material = token.encode("utf-8") + b"\0" + client.encode("utf-8")
        return hashlib.sha256(material).hexdigest()
    if client and not gate_enabled:
        return hashlib.sha256(b"open\0" + client.encode("utf-8")).hexdigest()
    return None


NOT_IMPLEMENTED = (
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/images/variations",
    "/v1/files",
    "/v1/batches",
    "/v1/fine-tuning/jobs",
    "/v1/fine_tuning/jobs",
    "/v1/conversations",
)


def _authorization(request: Request) -> str | None:
    """Bearer/Basic from Authorization, or Anthropic's `x-api-key` as a bearer."""
    header = request.headers.get("authorization")
    if header:
        return header
    api_key = request.headers.get("x-api-key")
    return f"Bearer {api_key}" if api_key else None


def _implicit_session(request: Request, body: dict[str, Any] | None) -> str | None:
    for name in _IMPLICIT_SESSION_HEADERS:
        value = request.headers.get(name)
        if value:
            return value
    cache_key = (body or {}).get("prompt_cache_key")
    return cache_key if isinstance(cache_key, str) and cache_key else None


def session_id_from_request(request: Request, *, gate_enabled: bool) -> str | None:
    return session_store_key(
        _authorization(request),
        request.headers.get("x-enigmatic-session"),
        gate_enabled=gate_enabled,
    )


def _is_anthropic_path(request: Request) -> bool:
    return request.url.path.rstrip("/").endswith("/messages")


def create_app(
    config: EnigmaticConfig | None = None,
    pipeline: Pipeline | None = None,
) -> FastAPI:
    cfg = config or load_config()
    pipeline = pipeline or build_pipeline(cfg)
    router = Router(cfg)
    http_clients: dict[str, HttpProvider] = {}
    agent_gate = asyncio.Semaphore(MAX_AGENT_CONCURRENCY)
    gate_enabled = bool(cfg.resolved_api_key)
    started_at = int(time.time())

    def namespace_for(request: Request) -> str:
        token = token_from_authorization(_authorization(request))
        if gate_enabled and token:
            return hashlib.sha256(b"ns\0" + token.encode("utf-8")).hexdigest()
        return "open"

    def session_for(
        request: Request,
        body: dict[str, Any] | None = None,
        *,
        persistent: bool = False,
    ) -> tuple[SessionMapping, str | None]:
        """Vault for this request, most specific first.

        A `previous_response_id` we produced reuses that response's vault, so
        placeholders stored upstream still restore. Otherwise an explicit or
        implicit client session key is used. `persistent` requests (Responses)
        get a fresh stored vault so later turns can chain to them.
        """
        previous = (body or {}).get("previous_response_id")
        if isinstance(previous, str) and previous:
            bound = STORE.lookup_response(namespace_for(request), previous)
            if bound is not None:
                return STORE.get(bound[0]), bound[0]
        key = session_store_key(
            _authorization(request),
            _implicit_session(request, body),
            gate_enabled=gate_enabled,
        )
        if key is None and persistent:
            key = f"resp-{uuid.uuid4().hex}"
        if key is None:
            return STORE.ephemeral(), None
        return STORE.get(key), key

    def http_for(route: Route) -> HttpProvider:
        assert route.http is not None
        existing = http_clients.get(route.profile_id)
        if existing is None:
            existing = HttpProvider(route.http)
            http_clients[route.profile_id] = existing
        return existing

    def upstream_headers(request: Request, route: Route) -> dict[str, str]:
        return forwardable_request_headers(request.headers, route.kind)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        for client in http_clients.values():
            await client.close()

    app = FastAPI(
        title="Enigmatic",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def api_route(path: str, methods: list[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Serve `path` under `/v1` and bare, for clients whose base URL omits `/v1`."""

        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            for prefix in ("/v1", ""):
                app.add_api_route(
                    prefix + path,
                    handler,
                    methods=methods,
                    response_model=None,
                    include_in_schema=False,
                )
            return handler

        return register

    def check_gate(request: Request) -> JSONResponse | None:
        expected = cfg.resolved_api_key
        if not expected:
            return None
        token = token_from_authorization(_authorization(request))
        if token is not None and tokens_equal(token, expected):
            return None
        body = openai_auth_error(provided=token is not None)
        if _is_anthropic_path(request):
            body = anthropic_error_body(body["error"]["message"], 401)
        return JSONResponse(
            body,
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Enigmatic"'},
        )

    async def read_body(request: Request) -> dict[str, Any]:
        limit = cfg.max_body_bytes or MAX_BODY_BYTES
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            raise PayloadTooLarge()
        raw = await request.body()
        if len(raw) > limit:
            raise PayloadTooLarge()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClientError("Invalid JSON object") from exc
        if not isinstance(payload, dict):
            raise ClientError("Invalid JSON object")
        return payload

    def anonymize(body: dict[str, Any], mapping: SessionMapping) -> dict[str, Any]:
        walked = walk(body, pipeline, mapping)
        if not isinstance(walked, dict):
            raise TypeError("payload must be an object")
        return walked

    async def anonymize_async(body: dict[str, Any], mapping: SessionMapping) -> dict[str, Any]:
        return await asyncio.to_thread(anonymize, body, mapping)

    def resolve(model: str, inbound: str) -> Route:
        try:
            return router.resolve(model, inbound)
        except KeyError:
            raise ClientError("No provider profile for the requested model", code="model_not_found") from None

    def require_openai_http(route: Route, feature: str) -> None:
        if route.kind != "openai" or route.http is None:
            raise ClientError(
                f"{feature} requires an OpenAI-compatible HTTP upstream",
                status_code=501,
                code="unsupported_upstream",
            )

    async def read_json(response: Any) -> Any:
        raw = await response.aread()
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UpstreamPayloadError() from exc

    async def upstream_error(response: Any, mapping: SessionMapping, inbound: str) -> JSONResponse:
        """Relay an upstream error in the inbound API's error shape."""
        raw = await response.aread()
        status = int(response.status_code)
        try:
            payload: Any = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        message = error_message(payload, raw)
        if inbound == "anthropic":
            body = payload if is_anthropic_error(payload) else anthropic_error_body(message, status)
        else:
            body = payload if is_openai_error(payload) else openai_error_body(message, openai_error_type(status))
        return JSONResponse(body, status, headers=returnable_response_headers(response))

    def stream_response(
        chunks: AsyncIterator[bytes],
        dialect: str,
        upstream: Any = None,
    ) -> StreamingResponse:
        async def guarded() -> AsyncIterator[bytes]:
            try:
                async for chunk in chunks:
                    yield chunk
            except httpx.TimeoutException:
                logger.warning("upstream stream timed out")
                yield stream_error_bytes(dialect, "Upstream stream timed out")
            except Exception as exc:
                logger.warning("upstream stream failed: %s", type(exc).__name__)
                yield stream_error_bytes(dialect, "Upstream stream failed")

        headers = {**_STREAM_HEADERS, **returnable_response_headers(upstream)}
        return StreamingResponse(guarded(), media_type="text/event-stream", headers=headers)

    async def relay(
        response: Any,
        mapping: SessionMapping,
        want_stream: bool,
        dialect: str,
        on_response_id: Callable[[str], None] | None = None,
    ) -> Response:
        """Restore an OpenAI-shaped upstream response (JSON or SSE) for the client."""
        if response.status_code >= 400:
            return await upstream_error(response, mapping, "openai")
        if want_stream:
            return stream_response(
                restored_sse(response, mapping, dialect, on_response_id),  # type: ignore[arg-type]
                dialect,
                response,
            )
        payload = mapping.restore_complete_any(await read_json(response))
        if on_response_id is not None and isinstance(payload, dict) and isinstance(payload.get("id"), str):
            on_response_id(payload["id"])
        return JSONResponse(payload, response.status_code, headers=returnable_response_headers(response))

    def binder(request: Request, key: str | None, profile_id: str) -> Callable[[str], None] | None:
        if key is None:
            return None
        namespace = namespace_for(request)
        return lambda response_id: STORE.bind_response(namespace, response_id, key, profile_id)

    def synthetic_model(model_id: str, owner: str) -> dict[str, Any]:
        return {"id": model_id, "object": "model", "created": started_at, "owned_by": owner}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @api_route("/models", ["GET"])
    async def list_models(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        models: list[dict[str, Any]] = []
        for name in cfg.http:
            models.append(synthetic_model(f"{name}/default", name))
        for name in {**cfg.acp, **cfg.jsonl}:
            models.append(synthetic_model(f"{name}/default", name))
        default = cfg.http.get(cfg.default_profile)
        if default and default.type != "anthropic":
            try:
                route = router.resolve(f"{cfg.default_profile}/default", "openai")
                if route.http:
                    provider = http_for(route)
                    response = await provider.request("GET", "/v1/models", None, stream=False)
                    raw = await response.aread()
                    if response.status_code < 400:
                        payload = json.loads(raw.decode("utf-8"))
                        for item in payload.get("data") or []:
                            if isinstance(item, dict) and "id" in item:
                                models.append({"object": "model", "created": started_at, "owned_by": "", **item})
            except Exception as exc:
                logger.info("upstream /models skipped: %s", exc)
        return JSONResponse({"object": "list", "data": models})

    @api_route("/models/{model:path}", ["GET"])
    async def retrieve_model(request: Request, model: str) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        prefix, bare = strip_model_prefix(model)
        if bare == "default" and (prefix in cfg.http or prefix in cfg.acp or prefix in cfg.jsonl):
            return JSONResponse(synthetic_model(model, prefix))
        route = resolve(model, "openai")
        if route.kind != "openai" or route.http is None:
            return JSONResponse(synthetic_model(model, route.profile_id))
        response = await http_for(route).request(
            "GET", f"/v1/models/{quote(bare, safe='/')}", None, stream=False
        )
        if response.status_code >= 400:
            return await upstream_error(response, STORE.ephemeral(), "openai")
        return JSONResponse(await read_json(response), headers=returnable_response_headers(response))

    async def handle_openai_chat(request: Request, body: dict[str, Any]) -> Response:
        mapping, _key = session_for(request, body)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        want_stream = bool(anon.get("stream"))
        if route.kind in {"acp", "jsonl"}:
            return await _agent_chat(route, anon, mapping, want_stream, inbound="openai")
        provider = http_for(route)
        headers = upstream_headers(request, route)
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        if route.kind == "anthropic":
            translated = openai_to_anthropic(outbound)
            response = await provider.request(
                "POST", "/v1/messages", translated, stream=want_stream, extra_headers=headers
            )
            if response.status_code >= 400:
                return await upstream_error(response, mapping, "openai")
            if want_stream:
                include_usage = bool((outbound.get("stream_options") or {}).get("include_usage"))
                return stream_response(
                    translate_and_restore_anthropic_to_openai(response, mapping, route.model, include_usage),
                    "chat",
                    response,
                )
            payload = mapping.restore_complete_any(await read_json(response))
            if not isinstance(payload, dict):
                raise UpstreamPayloadError()
            return JSONResponse(
                openai_completion_from_anthropic(payload, route.model),
                headers=returnable_response_headers(response),
            )
        response = await provider.request(
            "POST", "/v1/chat/completions", outbound, stream=want_stream, extra_headers=headers
        )
        return await relay(response, mapping, want_stream, "chat")

    async def handle_anthropic(request: Request, body: dict[str, Any]) -> Response:
        mapping, _key = session_for(request, body)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "anthropic")
        want_stream = bool(anon.get("stream"))
        if route.kind in {"acp", "jsonl"}:
            return await _agent_chat(route, anon, mapping, want_stream, inbound="anthropic")
        provider = http_for(route)
        headers = upstream_headers(request, route)
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        if route.kind == "openai":
            translated = anthropic_to_openai_chat(outbound)
            response = await provider.request(
                "POST", "/v1/chat/completions", translated, stream=want_stream, extra_headers=headers
            )
            if response.status_code >= 400:
                return await upstream_error(response, mapping, "anthropic")
            if want_stream:
                return stream_response(
                    translate_and_restore_openai_to_anthropic(response, mapping, route.model),
                    "anthropic",
                    response,
                )
            payload = mapping.restore_complete_any(await read_json(response))
            if not isinstance(payload, dict):
                raise UpstreamPayloadError()
            return JSONResponse(
                anthropic_message_from_openai(payload, route.model),
                headers=returnable_response_headers(response),
            )
        response = await provider.request(
            "POST", "/v1/messages", outbound, stream=want_stream, extra_headers=headers
        )
        if response.status_code >= 400:
            return await upstream_error(response, mapping, "anthropic")
        if want_stream:
            return stream_response(restored_sse(response, mapping, "anthropic"), "anthropic", response)
        payload = mapping.restore_complete_any(await read_json(response))
        return JSONResponse(payload, headers=returnable_response_headers(response))

    async def _agent_chat(
        route: Route,
        anon: dict[str, Any],
        mapping: SessionMapping,
        want_stream: bool,
        inbound: str,
    ) -> JSONResponse | StreamingResponse:
        if inbound == "anthropic":
            prompt = pack_anthropic(anon)
        else:
            prompt = pack_openai_chat(anon)
        text = await _run_agent(route, prompt, mapping)
        model = route.model or str(anon.get("model") or route.profile_id)
        if inbound == "anthropic":
            if want_stream:
                return StreamingResponse(
                    _bytes_from_iter(anthropic_stream_from_text(text, model)),
                    media_type="text/event-stream",
                )
            return JSONResponse(anthropic_message_from_text(text, model))
        if want_stream:
            return StreamingResponse(
                _bytes_from_iter(openai_chat_stream_from_text(text, model)),
                media_type="text/event-stream",
            )
        return JSONResponse(openai_completion_from_text(text, model))

    async def _run_agent(route: Route, prompt: str, mapping: SessionMapping) -> str:
        try:
            await asyncio.wait_for(agent_gate.acquire(), timeout=0.05)
        except TimeoutError as exc:
            raise AgentBusy("Too many agent CLI requests") from exc
        try:
            text = await invoke_agent(route, prompt)
            return mapping.restore_complete(text)
        finally:
            agent_gate.release()

    @api_route("/chat/completions", ["POST"])
    async def chat_completions(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        logger.info(
            "chat.completions session=%s",
            session_id_from_request(request, gate_enabled=gate_enabled) or "ephemeral",
        )
        return await handle_openai_chat(request, body)

    @api_route("/completions", ["POST"])
    async def completions(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping, _key = session_for(request, body)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        if route.kind in {"acp", "jsonl"}:
            text = await _run_agent(route, pack_openai_chat(anon), mapping)
            return JSONResponse(
                {
                    "id": "cmpl-enigmatic",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": route.model,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
                }
            )
        require_openai_http(route, "Legacy completions")
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        want_stream = bool(outbound.get("stream"))
        response = await http_for(route).request(
            "POST",
            "/v1/completions",
            outbound,
            stream=want_stream,
            extra_headers=upstream_headers(request, route),
        )
        return await relay(response, mapping, want_stream, "chat")

    @api_route("/embeddings", ["POST"])
    async def embeddings(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping, _key = session_for(request, body)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        if route.kind in {"acp", "jsonl"}:
            return JSONResponse(
                openai_error_body(
                    "Embeddings are not supported on agent-CLI upstreams",
                    code="unsupported_upstream",
                ),
                501,
            )
        require_openai_http(route, "Embeddings")
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        response = await http_for(route).request(
            "POST",
            "/v1/embeddings",
            outbound,
            stream=False,
            extra_headers=upstream_headers(request, route),
        )
        if response.status_code >= 400:
            return await upstream_error(response, mapping, "openai")
        return JSONResponse(await read_json(response), headers=returnable_response_headers(response))

    @api_route("/responses", ["POST"])
    async def responses(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping, key = session_for(request, body, persistent=True)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        want_stream = bool(anon.get("stream"))
        model = route.model or str(anon.get("model") or "")
        if route.kind != "openai" and anon.get("previous_response_id"):
            raise ClientError(
                "previous_response_id requires an OpenAI Responses upstream; resend the full input",
                param="previous_response_id",
            )
        if route.kind in {"acp", "jsonl"}:
            text = await _run_agent(route, pack_responses(anon), mapping)
            if want_stream:
                return StreamingResponse(
                    _bytes_from_iter(responses_stream_from_text(text, model)),
                    media_type="text/event-stream",
                )
            return JSONResponse(responses_output_from_text(text, model))
        provider = http_for(route)
        headers = upstream_headers(request, route)
        outbound = dict(anon)
        outbound["model"] = model or outbound.get("model")
        if route.kind == "anthropic":
            translated = openai_to_anthropic(responses_input_to_messages(outbound))
            response = await provider.request(
                "POST", "/v1/messages", translated, stream=want_stream, extra_headers=headers
            )
            if response.status_code >= 400:
                return await upstream_error(response, mapping, "openai")
            if want_stream:
                chunks = anthropic_events_to_chat_chunks(
                    restored_events(response, mapping, "anthropic"), model, include_usage=True
                )
                return stream_response(
                    responses_events_from_chat_chunks(chunks, model, request=outbound),
                    "responses",
                    response,
                )
            payload = mapping.restore_complete_any(await read_json(response))
            if not isinstance(payload, dict):
                raise UpstreamPayloadError()
            completion = openai_completion_from_anthropic(payload, model)
            return JSONResponse(
                responses_from_chat_completion(completion, model, request=outbound),
                headers=returnable_response_headers(response),
            )
        response = await provider.request(
            "POST", "/v1/responses", outbound, stream=want_stream, extra_headers=headers
        )
        return await relay(response, mapping, want_stream, "responses", binder(request, key, route.profile_id))

    async def responses_passthrough(request: Request, path: str, feature: str) -> Response:
        """Anonymized POST to an OpenAI-only Responses sub-endpoint (compact, input_tokens)."""
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping, key = session_for(request, body, persistent=True)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        require_openai_http(route, feature)
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        response = await http_for(route).request(
            "POST", path, outbound, stream=False, extra_headers=upstream_headers(request, route)
        )
        return await relay(response, mapping, False, "responses", binder(request, key, route.profile_id))

    @api_route("/responses/compact", ["POST"])
    async def responses_compact(request: Request) -> Response:
        return await responses_passthrough(request, "/v1/responses/compact", "Responses compaction")

    @api_route("/responses/input_tokens", ["POST"])
    async def responses_input_tokens(request: Request) -> Response:
        return await responses_passthrough(request, "/v1/responses/input_tokens", "Input token counting")

    async def response_resource(request: Request, response_id: str, suffix: str = "") -> Response:
        """GET/DELETE/cancel on a stored response, restored with the vault that created it."""
        gate = check_gate(request)
        if gate:
            return gate
        bound = STORE.lookup_response(namespace_for(request), response_id)
        profile_id = bound[1] if bound else cfg.default_profile
        profile = cfg.http.get(profile_id)
        if profile is None or profile.type == "anthropic":
            raise ClientError(f"No response found with id '{response_id}'.", status_code=404)
        mapping = STORE.get(bound[0]) if bound else STORE.ephemeral()
        route = Route(kind="openai", profile_id=profile_id, model="", http=profile)
        want_stream = request.method == "GET" and request.query_params.get("stream") == "true"
        response = await http_for(route).request(
            request.method,
            f"/v1/responses/{quote(response_id, safe='')}{suffix}",
            None,
            stream=want_stream,
            extra_headers=upstream_headers(request, route),
            query=list(request.query_params.multi_items()),
        )
        return await relay(response, mapping, want_stream, "responses")

    @api_route("/responses/{response_id}", ["GET", "DELETE"])
    async def response_get_or_delete(request: Request, response_id: str) -> Response:
        return await response_resource(request, response_id)

    @api_route("/responses/{response_id}/cancel", ["POST"])
    async def response_cancel(request: Request, response_id: str) -> Response:
        return await response_resource(request, response_id, "/cancel")

    @api_route("/responses/{response_id}/input_items", ["GET"])
    async def response_input_items(request: Request, response_id: str) -> Response:
        return await response_resource(request, response_id, "/input_items")

    @api_route("/messages", ["POST"])
    async def messages(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        logger.info(
            "anthropic.messages session=%s",
            session_id_from_request(request, gate_enabled=gate_enabled) or "ephemeral",
        )
        return await handle_anthropic(request, body)

    async def _not_implemented_handler(request: Request) -> JSONResponse:
        gate = check_gate(request)
        if gate:
            return gate
        return JSONResponse(
            openai_error_body(
                f"{request.url.path} is not implemented in Enigmatic",
                code="unsupported_endpoint",
            ),
            501,
        )

    for path in NOT_IMPLEMENTED:
        for variant in (path, path.removeprefix("/v1"), path + "/{rest:path}", path.removeprefix("/v1") + "/{rest:path}"):
            app.add_api_route(
                variant,
                _not_implemented_handler,
                methods=["GET", "POST", "DELETE"],
                include_in_schema=False,
            )

    def error_response(
        request: Request,
        message: str,
        status: int,
        error_type: str | None = None,
        *,
        code: str | None = None,
        param: str | None = None,
    ) -> JSONResponse:
        if _is_anthropic_path(request):
            return JSONResponse(anthropic_error_body(message, status), status)
        body = openai_error_body(message, error_type or openai_error_type(status), code=code, param=param)
        return JSONResponse(body, status)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            message = f"Unknown request URL: {request.method} {request.url.path}."
            return error_response(request, message, 404, code="unknown_url")
        return error_response(request, str(exc.detail), exc.status_code)

    @app.exception_handler(HttpProviderError)
    async def http_err(request: Request, exc: HttpProviderError) -> JSONResponse:
        return error_response(request, str(exc), exc.status_code, code="upstream_error")

    @app.exception_handler(httpx.TimeoutException)
    async def upstream_timeout(request: Request, _exc: httpx.TimeoutException) -> JSONResponse:
        return error_response(request, "Upstream request timed out", 504, code="upstream_timeout")

    @app.exception_handler(httpx.TransportError)
    async def upstream_unreachable(request: Request, _exc: httpx.TransportError) -> JSONResponse:
        return error_response(request, "Could not reach the upstream provider", 502, code="upstream_unreachable")

    @app.exception_handler(UpstreamPayloadError)
    async def upstream_payload(request: Request, _exc: UpstreamPayloadError) -> JSONResponse:
        return error_response(request, "Upstream returned an invalid payload", 502, code="upstream_error")

    @app.exception_handler(PayloadTooLarge)
    async def payload_too_large(request: Request, _exc: PayloadTooLarge) -> JSONResponse:
        return error_response(request, "Request body is too large", 413, "invalid_request_error", code="request_too_large")

    @app.exception_handler(AgentBusy)
    async def agent_busy(request: Request, _exc: AgentBusy) -> JSONResponse:
        return error_response(request, "Too many agent CLI requests", 429, code="rate_limit_exceeded")

    @app.exception_handler(RouteError)
    async def route_error(request: Request, exc: RouteError) -> JSONResponse:
        return error_response(request, str(exc), 400, "invalid_request_error")

    @app.exception_handler(ClientError)
    async def client_error(request: Request, exc: ClientError) -> JSONResponse:
        return error_response(
            request, str(exc), exc.status_code, "invalid_request_error", code=exc.code, param=exc.param
        )

    @app.exception_handler(Exception)
    async def any_err(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request failed")
        return error_response(request, "Internal error", 500, "server_error")

    app.state.config = cfg
    app.state.pipeline = pipeline
    return app


async def _bytes_from_iter(chunks: Any) -> AsyncIterator[bytes]:
    for chunk in chunks:
        if isinstance(chunk, bytes):
            yield chunk
        else:
            yield str(chunk).encode("utf-8")
