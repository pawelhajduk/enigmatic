"""FastAPI app: inbound OpenAI/Anthropic surface, anonymize, route, restore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from enigmatic import __version__
from enigmatic.auth import openai_auth_error, token_from_authorization, tokens_equal
from enigmatic.config import EnigmaticConfig, load_config
from enigmatic.openai_walk import walk
from enigmatic.presidio_ops.mapping import STORE, SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline, build_pipeline
from enigmatic.protocols.packing import pack_anthropic, pack_openai_chat, pack_responses
from enigmatic.protocols.sse import (
    anthropic_stream_from_text,
    openai_chat_stream_from_text,
    responses_stream_from_text,
)
from enigmatic.protocols.translate import (
    anthropic_message_from_openai,
    anthropic_message_from_text,
    anthropic_to_openai_chat,
    openai_completion_from_anthropic,
    openai_completion_from_text,
    openai_to_anthropic,
    responses_input_to_messages,
    responses_output_from_text,
)
from enigmatic.providers.acp import run_acp_prompt
from enigmatic.providers.http import (
    HttpProvider,
    HttpProviderError,
    restored_sse,
    translate_and_restore_anthropic_to_openai,
    translate_and_restore_openai_to_anthropic,
)
from enigmatic.providers.jsonl import jsonl_denies_tools, run_jsonl_prompt
from enigmatic.providers.router import Route, RouteError, Router

logger = logging.getLogger("enigmatic.server")

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_AGENT_CONCURRENCY = 2
_SESSION_HEADER_MAX = 256


class PayloadTooLarge(Exception):
    pass


class AgentBusy(Exception):
    pass


class ClientError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


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
)

def session_id_from_request(request: Request, *, gate_enabled: bool) -> str | None:
    return session_store_key(
        request.headers.get("authorization"),
        request.headers.get("x-enigmatic-session"),
        gate_enabled=gate_enabled,
    )


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

    def mapping_for(request: Request) -> SessionMapping:
        key = session_id_from_request(request, gate_enabled=gate_enabled)
        if key is None:
            return STORE.ephemeral()
        return STORE.get(key)

    def http_for(route: Route) -> HttpProvider:
        assert route.http is not None
        existing = http_clients.get(route.profile_id)
        if existing is None:
            existing = HttpProvider(route.http)
            http_clients[route.profile_id] = existing
        return existing

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

    def check_gate(request: Request) -> JSONResponse | None:
        expected = cfg.resolved_api_key
        if not expected:
            return None
        token = token_from_authorization(request.headers.get("authorization"))
        if token is not None and tokens_equal(token, expected):
            return None
        return JSONResponse(
            openai_auth_error(provided=token is not None),
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Enigmatic"'},
        )

    async def read_body(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise PayloadTooLarge()
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
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
            raise ClientError("No provider profile for the requested model") from None

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/v1/models", response_model=None)
    async def list_models(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        models: list[dict[str, Any]] = []
        for name in cfg.http:
            models.append({"id": f"{name}/default", "object": "model", "owned_by": name})
        for name in {**cfg.acp, **cfg.jsonl}:
            models.append({"id": f"{name}/default", "object": "model", "owned_by": name})
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
                                models.append(item)
            except Exception as exc:
                logger.info("upstream /models skipped: %s", exc)
        return JSONResponse({"object": "list", "data": models})

    async def handle_openai_chat(request: Request, body: dict[str, Any]) -> JSONResponse | StreamingResponse:
        mapping = mapping_for(request)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        want_stream = bool(anon.get("stream"))
        if route.kind in {"acp", "jsonl"}:
            return await _agent_chat(route, anon, mapping, want_stream, inbound="openai")
        assert route.http is not None
        provider = http_for(route)
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        if route.kind == "anthropic":
            translated = openai_to_anthropic(outbound)
            response = await provider.request("POST", "/v1/messages", translated, stream=want_stream)
            if response.status_code >= 400:
                raw = await response.aread()
                return JSONResponse(json.loads(raw.decode() or "{}") or {"error": raw.decode()}, response.status_code)
            if want_stream:
                return StreamingResponse(
                    translate_and_restore_anthropic_to_openai(response, mapping, route.model),
                    media_type="text/event-stream",
                )
            raw = await response.aread()
            payload = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
            if not isinstance(payload, dict):
                return JSONResponse({"error": "invalid upstream payload"}, 502)
            return JSONResponse(openai_completion_from_anthropic(payload, route.model))
        response = await provider.request("POST", "/v1/chat/completions", outbound, stream=want_stream)
        return await _http_openai_result(response, mapping, want_stream)

    async def handle_anthropic(request: Request, body: dict[str, Any]) -> JSONResponse | StreamingResponse:
        mapping = mapping_for(request)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "anthropic")
        want_stream = bool(anon.get("stream"))
        if route.kind in {"acp", "jsonl"}:
            return await _agent_chat(route, anon, mapping, want_stream, inbound="anthropic")
        assert route.http is not None
        provider = http_for(route)
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        if route.kind == "openai":
            translated = anthropic_to_openai_chat(outbound)
            response = await provider.request("POST", "/v1/chat/completions", translated, stream=want_stream)
            if response.status_code >= 400:
                raw = await response.aread()
                return JSONResponse(json.loads(raw.decode() or "{}") or {"error": raw.decode()}, response.status_code)
            if want_stream:
                return StreamingResponse(
                    translate_and_restore_openai_to_anthropic(response, mapping, route.model),
                    media_type="text/event-stream",
                )
            raw = await response.aread()
            payload = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
            if not isinstance(payload, dict):
                return JSONResponse({"error": "invalid upstream payload"}, 502)
            return JSONResponse(anthropic_message_from_openai(payload, route.model))
        response = await provider.request("POST", "/v1/messages", outbound, stream=want_stream)
        if response.status_code >= 400:
            raw = await response.aread()
            return JSONResponse(json.loads(raw.decode() or "{}") or {"error": raw.decode()}, response.status_code)
        if want_stream:
            return StreamingResponse(restored_sse(response, mapping), media_type="text/event-stream")
        raw = await response.aread()
        payload = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
        return JSONResponse(payload if isinstance(payload, dict) else {"error": "invalid upstream"})

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
            return await _run_agent_unlocked(route, prompt, mapping)
        finally:
            agent_gate.release()

    async def _run_agent_unlocked(route: Route, prompt: str, mapping: SessionMapping) -> str:
        if route.kind == "acp" and route.acp is not None:
            try:
                text = await run_acp_prompt(route.acp, prompt, model=route.model or None)
                return mapping.restore_complete(text)
            except Exception as exc:
                logger.info("ACP failed, JSONL fallback: %s", exc)
                if route.jsonl is None:
                    raise
        if route.jsonl is None:
            raise RuntimeError(f"No JSONL profile for {route.profile_id}")
        if not jsonl_denies_tools(route.jsonl):
            raise RouteError(f"Agent profile {route.profile_id} does not deny tools")
        parser = "claude" if route.profile_id == "claude" else "copilot"
        text = await run_jsonl_prompt(route.jsonl, prompt, model=route.model or None, parser=parser)
        return mapping.restore_complete(text)

    async def _http_openai_result(
        response: Any,
        mapping: SessionMapping,
        want_stream: bool,
    ) -> JSONResponse | StreamingResponse:
        if response.status_code >= 400:
            raw = await response.aread()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
            return JSONResponse(payload, response.status_code)
        if want_stream:
            return StreamingResponse(restored_sse(response, mapping), media_type="text/event-stream")
        raw = await response.aread()
        payload = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
        return JSONResponse(payload if isinstance(payload, dict) else {"error": "invalid upstream"})

    @app.post("/v1/chat/completions", response_model=None)
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

    @app.post("/v1/completions", response_model=None)
    async def completions(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping = mapping_for(request)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        if route.kind in {"acp", "jsonl"}:
            text = await _run_agent(route, pack_openai_chat(anon), mapping)
            return JSONResponse(
                {
                    "id": "cmpl-enigmatic",
                    "object": "text_completion",
                    "model": route.model,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                }
            )
        assert route.http is not None
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        response = await http_for(route).request(
            "POST", "/v1/completions", outbound, stream=bool(outbound.get("stream"))
        )
        return await _http_openai_result(response, mapping, bool(outbound.get("stream")))

    @app.post("/v1/embeddings", response_model=None)
    async def embeddings(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping = mapping_for(request)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        if route.kind in {"acp", "jsonl"}:
            return JSONResponse(
                {
                    "error": {
                        "message": "Embeddings are not supported on agent-CLI upstreams",
                        "type": "not_implemented",
                    }
                },
                501,
            )
        assert route.http is not None
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        response = await http_for(route).request("POST", "/v1/embeddings", outbound, stream=False)
        raw = await response.aread()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"error": raw.decode("utf-8", errors="replace")}
        return JSONResponse(payload, response.status_code)

    @app.post("/v1/responses", response_model=None)
    async def responses(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping = mapping_for(request)
        anon = await anonymize_async(body, mapping)
        route = resolve(str(anon.get("model") or ""), "openai")
        want_stream = bool(anon.get("stream"))
        if route.kind in {"acp", "jsonl"}:
            text = await _run_agent(route, pack_responses(anon), mapping)
            model = route.model or str(anon.get("model") or "")
            if want_stream:
                return StreamingResponse(
                    _bytes_from_iter(responses_stream_from_text(text, model)),
                    media_type="text/event-stream",
                )
            return JSONResponse(responses_output_from_text(text, model))
        assert route.http is not None
        outbound = dict(anon)
        outbound["model"] = route.model or outbound.get("model")
        if route.kind == "anthropic":
            chat = responses_input_to_messages(outbound)
            translated = openai_to_anthropic(chat)
            response = await http_for(route).request("POST", "/v1/messages", translated, stream=False)
            raw = await response.aread()
            payload = mapping.restore_complete_any(json.loads(raw.decode("utf-8")))
            if not isinstance(payload, dict):
                return JSONResponse({"error": "invalid upstream"}, 502)
            text_parts = [
                block.get("text", "")
                for block in payload.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return JSONResponse(responses_output_from_text("".join(text_parts), route.model))
        response = await http_for(route).request("POST", "/v1/responses", outbound, stream=want_stream)
        return await _http_openai_result(response, mapping, want_stream)

    @app.post("/v1/messages", response_model=None)
    @app.post("/messages", response_model=None)
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
            {
                "error": {
                    "message": f"{request.url.path} is not implemented in Enigmatic v1",
                    "type": "not_implemented",
                }
            },
            501,
        )

    for path in NOT_IMPLEMENTED:
        app.add_api_route(
            path,
            _not_implemented_handler,
            methods=["GET", "POST", "DELETE"],
            include_in_schema=False,
        )

    def error_body(message: str, kind: str) -> dict[str, Any]:
        return {"error": {"message": message, "type": kind}}

    @app.exception_handler(HttpProviderError)
    async def http_err(_request: Request, exc: HttpProviderError) -> JSONResponse:
        return JSONResponse(error_body(str(exc), "upstream_error"), exc.status_code)

    @app.exception_handler(PayloadTooLarge)
    async def payload_too_large(_request: Request, _exc: PayloadTooLarge) -> JSONResponse:
        return JSONResponse(error_body("Request body is too large", "invalid_request_error"), 413)

    @app.exception_handler(AgentBusy)
    async def agent_busy(_request: Request, _exc: AgentBusy) -> JSONResponse:
        return JSONResponse(error_body("Too many agent CLI requests", "rate_limit_error"), 429)

    @app.exception_handler(RouteError)
    async def route_error(_request: Request, exc: RouteError) -> JSONResponse:
        return JSONResponse(error_body(str(exc), "invalid_request_error"), 400)

    @app.exception_handler(ClientError)
    async def client_error(_request: Request, exc: ClientError) -> JSONResponse:
        return JSONResponse(error_body(str(exc), "invalid_request_error"), exc.status_code)

    @app.exception_handler(Exception)
    async def any_err(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request failed")
        return JSONResponse(error_body("Internal error", "internal_error"), 500)

    app.state.config = cfg
    app.state.pipeline = pipeline
    return app


async def _bytes_from_iter(chunks: Any) -> AsyncIterator[bytes]:
    for chunk in chunks:
        if isinstance(chunk, bytes):
            yield chunk
        else:
            yield str(chunk).encode("utf-8")
