"""OpenAI-compatible access tokens for inbound requests.

OpenAI clients send `Authorization: Bearer <token>`. The API also accepts HTTP
Basic auth with a blank username and the token as the password.
"""

from __future__ import annotations

import base64
import binascii
import secrets
from typing import Any


def token_from_authorization(header: str | None) -> str | None:
    """Return the access token from an Authorization header, if one is present."""
    if header is None:
        return None
    scheme, _, rest = header.strip().partition(" ")
    credential = rest.strip()
    if not credential:
        return None
    scheme_name = scheme.lower()
    if scheme_name == "bearer":
        return credential
    if scheme_name == "basic":
        return _basic_token(credential)
    return None


def _basic_token(credential: str) -> str | None:
    try:
        decoded = base64.b64decode(credential, validate=False).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    username, separator, password = decoded.partition(":")
    if not separator:
        return None
    token = password if password else username
    token = token.strip()
    return token or None


def tokens_equal(provided: str, expected: str) -> bool:
    return secrets.compare_digest(provided, expected)


def openai_auth_error(*, provided: bool) -> dict[str, Any]:
    """Error body in the shape OpenAI clients already parse."""
    if provided:
        message = "Incorrect API key provided."
        code: str | None = "invalid_api_key"
    else:
        message = (
            "You must provide an API key in an Authorization header using Bearer auth "
            "(Authorization: Bearer <token>)."
        )
        code = None
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        }
    }
