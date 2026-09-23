from __future__ import annotations

import base64
import os

from fastapi.testclient import TestClient

from enigmatic.config import EnigmaticConfig, HttpProfile, load_config
from enigmatic.doctor import format_status
from enigmatic.providers.http import build_headers
from enigmatic.server import create_app


def _write_config(path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _hide_env(monkeypatch, *names: str) -> None:
    """Drop names for this test and restore whatever the process had."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_load_config_reads_access_token_and_upstream_key_from_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _hide_env(monkeypatch, "ENIGMATIC_API_KEY", "OPENAI_API_KEY")
    (tmp_path / ".env").write_text(
        "ENIGMATIC_API_KEY=sk-from-dotenv\nOPENAI_API_KEY=sk-upstream\n",
        encoding="utf-8",
    )
    _write_config(
        tmp_path / "providers.yaml",
        """
api_key: null
api_key_env: ENIGMATIC_API_KEY
http:
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
""",
    )

    cfg = load_config(tmp_path / "providers.yaml")

    assert cfg.resolved_api_key == "sk-from-dotenv"
    assert os.environ["OPENAI_API_KEY"] == "sk-upstream"
    assert build_headers(cfg.http["openai"])["authorization"] == "Bearer sk-upstream"
    status = format_status(cfg)
    assert "enabled" in status.lower()
    assert "sk-from-dotenv" not in status
    assert "sk-upstream" not in status


def test_process_env_overrides_env_file_and_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENIGMATIC_API_KEY", "sk-process")
    (tmp_path / ".env").write_text("ENIGMATIC_API_KEY=sk-file\n", encoding="utf-8")
    _write_config(
        tmp_path / "providers.yaml",
        "api_key: sk-yaml\napi_key_env: ENIGMATIC_API_KEY\n",
    )

    cfg = load_config(tmp_path / "providers.yaml")

    assert cfg.resolved_api_key == "sk-process"


def test_yaml_api_key_used_when_env_is_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _hide_env(monkeypatch, "ENIGMATIC_API_KEY")
    _write_config(tmp_path / "providers.yaml", "api_key: sk-yaml\n")

    cfg = load_config(tmp_path / "providers.yaml")

    assert cfg.resolved_api_key == "sk-yaml"


def test_env_local_overrides_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _hide_env(monkeypatch, "PROXY_TOKEN")
    (tmp_path / ".env").write_text("PROXY_TOKEN=from-env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("PROXY_TOKEN=from-local\n", encoding="utf-8")
    _write_config(
        tmp_path / "providers.yaml",
        "api_key: null\napi_key_env: PROXY_TOKEN\n",
    )

    cfg = load_config(tmp_path / "providers.yaml")

    assert cfg.resolved_api_key == "from-local"


def _gated_client(token: str = "sk-test") -> TestClient:
    cfg = EnigmaticConfig(
        api_key=token,
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url="https://example.invalid/v1")},
    )
    return TestClient(create_app(cfg))


def test_openai_bearer_gate_rejects_missing_and_wrong_tokens() -> None:
    client = _gated_client()

    missing = client.post("/v1/images/generations", json={"prompt": "nope"})
    wrong = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong"},
        json={"model": "openai/gpt-4o", "messages": []},
    )
    health = client.get("/health")

    assert missing.status_code == 401
    assert missing.json()["error"]["type"] == "invalid_request_error"
    assert missing.json()["error"]["code"] is None
    assert missing.headers["www-authenticate"].lower().startswith("bearer")
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_api_key"
    assert health.status_code == 200


def test_openai_bearer_and_basic_tokens_are_accepted() -> None:
    client = _gated_client("sk-test")
    basic = base64.b64encode(b":sk-test").decode("ascii")

    bearer = client.post(
        "/v1/images/generations",
        headers={"Authorization": "bearer sk-test"},
        json={"prompt": "nope"},
    )
    password = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Basic {basic}"},
        json={"input": "nope"},
    )

    assert bearer.status_code == 501
    assert password.status_code == 501
