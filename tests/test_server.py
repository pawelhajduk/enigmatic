from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from enigmatic.config import EnigmaticConfig, HttpProfile
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.server import create_app


class FakePipeline:
    entities = ["EMAIL_ADDRESS"]
    language = "en"
    score_threshold = 0.5

    def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
        needle = "ada@example.com"
        if needle in text:
            token = mapping.placeholder_for("EMAIL_ADDRESS", needle)
            return text.replace(needle, token)
        return text


def _app(http_url: str) -> TestClient:
    cfg = EnigmaticConfig(
        listen_host="127.0.0.1",
        listen_port=47821,
        default_profile="openai",
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url=http_url, api_key_env=None)},
        enabled_entities=["EMAIL_ADDRESS"],
    )
    return TestClient(create_app(cfg, pipeline=FakePipeline()))  # type: ignore[arg-type]


def test_health_and_root_and_501() -> None:
    client = _app("https://example.invalid/v1")
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    root = client.get("/")
    assert root.status_code == 404
    assert "<html" not in root.text.lower()
    forbidden = client.post("/v1/images/generations", json={"prompt": "nope"})
    assert forbidden.status_code == 501


def test_passthrough_preserves_unknown_keys_and_restores(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        async def aread(self) -> bytes:
            import json

            return json.dumps(self._body).encode()

    class FakeHttp:
        def __init__(self, profile: Any) -> None:
            self.profile = profile

        async def request(self, method: str, path: str, body: dict[str, Any] | None, stream: bool, **kwargs: Any) -> FakeResponse:
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            captured["stream"] = stream
            content = (body or {}).get("messages", [{}])[0].get("content", "")
            return FakeResponse(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": f"got {content}"}}],
                }
            )

        async def close(self) -> None:
            return None

    import enigmatic.server as server_mod

    monkeypatch.setattr(server_mod, "HttpProvider", FakeHttp)
    client = _app("https://api.openai.com/v1")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "write ada@example.com"}],
            "foo_unknown": True,
        },
    )
    assert response.status_code == 200
    assert captured["body"]["foo_unknown"] is True
    assert captured["body"]["model"] == "gpt-4o"
    assert "<EMAIL_ADDRESS_1>" in captured["body"]["messages"][0]["content"]
    assert "ada@example.com" not in captured["body"]["messages"][0]["content"]
    assert response.json()["choices"][0]["message"]["content"] == "got write ada@example.com"


def test_embeddings_anonymize_without_restore(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b'{"data":[{"embedding":[0.1],"index":0}],"object":"list"}'

    class FakeHttp:
        def __init__(self, profile: Any) -> None:
            self.profile = profile

        async def request(self, method: str, path: str, body: dict[str, Any] | None, stream: bool, **kwargs: Any) -> FakeResponse:
            captured["body"] = body
            return FakeResponse()

        async def close(self) -> None:
            return None

    import enigmatic.server as server_mod

    monkeypatch.setattr(server_mod, "HttpProvider", FakeHttp)
    client = _app("https://api.openai.com/v1")
    response = client.post(
        "/v1/embeddings",
        json={"model": "openai/text-embedding-3-small", "input": "secret ada@example.com"},
    )
    assert response.status_code == 200
    assert "<EMAIL_ADDRESS_1>" in captured["body"]["input"]
    assert response.json()["data"][0]["embedding"] == [0.1]


def test_agent_cli_embeddings_are_501() -> None:
    from enigmatic.config import AcpProfile, JsonlProfile

    cfg = EnigmaticConfig(
        default_profile="copilot",
        api_key_env=None,
        acp={"copilot": AcpProfile(command="copilot")},
        jsonl={"copilot": JsonlProfile(command="copilot")},
    )
    client = TestClient(create_app(cfg, pipeline=FakePipeline()))  # type: ignore[arg-type]
    response = client.post("/v1/embeddings", json={"model": "copilot/gpt-5", "input": "hi"})
    assert response.status_code == 501
