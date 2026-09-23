"""Error bodies in the shapes OpenAI and Anthropic clients parse."""

from __future__ import annotations

import json
from typing import Any

_ANTHROPIC_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def openai_error_type(status: int) -> str:
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "server_error"
    return "invalid_request_error"


def openai_error_body(
    message: str,
    error_type: str = "invalid_request_error",
    *,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def anthropic_error_body(message: str, status: int) -> dict[str, Any]:
    error_type = _ANTHROPIC_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}


def error_message(payload: object, raw: bytes = b"") -> str:
    """Best-effort human message from any upstream error body."""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str) and error:
            return error
        if payload.get("message"):
            return str(payload["message"])
        if payload.get("detail"):
            return str(payload["detail"])
    text = raw.decode("utf-8", errors="replace").strip()
    return text[:2000] or "Upstream request failed"


def is_openai_error(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("error"), dict) and payload.get("type") != "error"


def is_anthropic_error(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("type") == "error" and isinstance(payload.get("error"), dict)


def stream_error_bytes(dialect: str, message: str) -> bytes:
    """Final SSE event telling the client the stream failed after it started."""
    if dialect == "anthropic":
        body = {"type": "error", "error": {"type": "api_error", "message": message}}
        return f"event: error\ndata: {json.dumps(body)}\n\n".encode()
    if dialect == "responses":
        body = {"type": "error", "code": "server_error", "message": message, "param": None, "sequence_number": 0}
        return f"event: error\ndata: {json.dumps(body)}\n\n".encode()
    return f"data: {json.dumps(openai_error_body(message, 'server_error'))}\n\n".encode()
