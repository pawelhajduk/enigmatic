"""FastAPI app: inbound OpenAI/Anthropic surface, anonymize, route, restore."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from enigmatic import __version__
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
from enigmatic.providers.jsonl import run_jsonl_prompt
from enigmatic.providers.router import Route, Router

logger = logging.getLogger("enigmatic.server")

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

def session_id_from_request(request: Request) -> str:
    header = request.headers.get("x-enigmatic-session")
    if header:
        return header.strip()
    auth = request.headers.get("authorization")
    if auth:
        return hashlib.sha256(auth.encode("utf-8")).hexdigest()[:24]
    return "default"


def create_app(
    config: EnigmaticConfig | None = None,
    pipeline: Pipeline | None = None,
) -> FastAPI:
    cfg = config or load_config()
    pipeline = pipeline or build_pipeline(cfg)
    router = Router(cfg)
    http_clients: dict[str, HttpProvider] = {}

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
        if not cfg.api_key:
            return None
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {cfg.api_key}":
            return JSONResponse({"error": {"message": "Invalid local API key", "type": "auth"}}, 401)
        return None

    async def read_body(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def anonymize(body: dict[str, Any], mapping: SessionMapping) -> dict[str, Any]:
        walked = walk(body, pipeline, mapping)
        if not isinstance(walked, dict):
            raise TypeError("payload must be an object")
        return walked

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
        mapping = STORE.get(session_id_from_request(request))
        anon = anonymize(body, mapping)
        route = router.resolve(str(anon.get("model") or ""), "openai")
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
        mapping = STORE.get(session_id_from_request(request))
        anon = anonymize(body, mapping)
        route = router.resolve(str(anon.get("model") or ""), "anthropic")
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
        logger.info("chat.completions session=%s", session_id_from_request(request))
        return await handle_openai_chat(request, body)

    @app.post("/v1/completions", response_model=None)
    async def completions(request: Request) -> Response:
        gate = check_gate(request)
        if gate:
            return gate
        body = await read_body(request)
        mapping = STORE.get(session_id_from_request(request))
        anon = anonymize(body, mapping)
        route = router.resolve(str(anon.get("model") or ""), "openai")
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
        mapping = STORE.get(session_id_from_request(request))
        anon = anonymize(body, mapping)
        route = router.resolve(str(anon.get("model") or ""), "openai")
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
        mapping = STORE.get(session_id_from_request(request))
        anon = anonymize(body, mapping)
        route = router.resolve(str(anon.get("model") or ""), "openai")
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
        logger.info("anthropic.messages session=%s", session_id_from_request(request))
        return await handle_anthropic(request, body)

    def _not_implemented_handler(request: Request) -> JSONResponse:
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

    @app.exception_handler(HttpProviderError)
    async def http_err(_request: Request, exc: HttpProviderError) -> JSONResponse:
        return JSONResponse({"error": {"message": str(exc)}}, exc.status_code)

    @app.exception_handler(Exception)
    async def any_err(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request failed")
        return JSONResponse({"error": {"message": str(exc), "type": type(exc).__name__}}, 500)

    app.state.config = cfg
    app.state.pipeline = pipeline
    return app


async def _bytes_from_iter(chunks: Any) -> AsyncIterator[bytes]:
    for chunk in chunks:
        if isinstance(chunk, bytes):
            yield chunk
        else:
            yield str(chunk).encode("utf-8")
