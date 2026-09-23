from __future__ import annotations

import base64

from enigmatic.auth import openai_auth_error, token_from_authorization


def test_token_from_authorization_accepts_openai_bearer() -> None:
    assert token_from_authorization("Bearer sk-test") == "sk-test"
    assert token_from_authorization("bearer   sk-test") == "sk-test"
    assert token_from_authorization(None) is None
    assert token_from_authorization("Token sk-test") is None


def test_token_from_authorization_accepts_openai_basic_password() -> None:
    blank_user = base64.b64encode(b":sk-basic").decode("ascii")
    token_user = base64.b64encode(b"sk-user:").decode("ascii")

    assert token_from_authorization(f"Basic {blank_user}") == "sk-basic"
    assert token_from_authorization(f"Basic {token_user}") == "sk-user"


def test_openai_auth_error_shape() -> None:
    missing = openai_auth_error(provided=False)
    invalid = openai_auth_error(provided=True)

    assert missing["error"]["type"] == "invalid_request_error"
    assert missing["error"]["param"] is None
    assert missing["error"]["code"] is None
    assert "Authorization" in missing["error"]["message"]
    assert invalid["error"]["code"] == "invalid_api_key"
    assert "sk-" not in invalid["error"]["message"]
