"""Security and performance guards: sessions, limits, secrets, agent isolation."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from enigmatic.auth import bind_requires_api_key, is_loopback_host
from enigmatic.config import AcpProfile, EnigmaticConfig, HttpProfile, JsonlProfile
from enigmatic.envfile import load_env_files
from enigmatic.presidio_ops.images import redact_data_url
from enigmatic.presidio_ops.mapping import MappingStore, SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline
from enigmatic.presidio_ops.secrets import secret_recognizers
from enigmatic.providers.http import DEFAULT_TIMEOUT
from enigmatic.providers.jsonl import jsonl_denies_tools, run_jsonl_prompt
from enigmatic.providers.router import Router
from enigmatic.server import create_app, session_store_key


def test_loopback_bind_without_key_is_allowed_and_public_bind_is_not() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not bind_requires_api_key("127.0.0.1", None)
    assert bind_requires_api_key("0.0.0.0", None)
    assert not bind_requires_api_key("0.0.0.0", "sk-local")


def test_serve_refuses_public_bind_without_api_key(monkeypatch: Any) -> None:
    from typer.testing import CliRunner

    from enigmatic.cli import app

    monkeypatch.setattr(
        "enigmatic.cli.load_config",
        lambda _path=None: EnigmaticConfig(api_key=None, api_key_env=None),
    )
    monkeypatch.setattr("enigmatic.cli.build_pipeline", lambda _cfg: object())
    ran: dict[str, bool] = {}

    def _run(*_args: object, **_kwargs: object) -> None:
        ran["yes"] = True

    monkeypatch.setattr("enigmatic.cli.uvicorn.run", _run)
    monkeypatch.setattr("enigmatic.cli.create_app", lambda cfg: object())
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert "yes" not in ran
    assert "ENIGMATIC_API_KEY" in result.output


def test_serve_warns_when_loopback_gate_is_off(monkeypatch: Any) -> None:
    from typer.testing import CliRunner

    from enigmatic.cli import app

    monkeypatch.setattr(
        "enigmatic.cli.load_config",
        lambda _path=None: EnigmaticConfig(api_key=None, api_key_env=None),
    )
    monkeypatch.setattr("enigmatic.cli.create_app", lambda cfg: object())
    monkeypatch.setattr("enigmatic.cli.uvicorn.run", lambda *_a, **_k: None)
    result = CliRunner().invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert "auth gate is off" in result.output.lower()


def test_env_file_does_not_apply_proxy_or_ca_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    (tmp_path / ".env").write_text(
        "HTTPS_PROXY=http://evil.example\nSSL_CERT_FILE=/tmp/evil.pem\nOPENAI_API_KEY=sk-ok\n",
        encoding="utf-8",
    )
    applied = load_env_files([tmp_path])
    assert applied["OPENAI_API_KEY"] == "sk-ok"
    assert "HTTPS_PROXY" not in applied
    assert "SSL_CERT_FILE" not in applied
    assert "HTTPS_PROXY" not in os.environ
    assert "SSL_CERT_FILE" not in os.environ


def test_session_key_is_ephemeral_without_header_and_namespaced_by_token() -> None:
    assert session_store_key(None, None, gate_enabled=False) is None
    first = session_store_key(None, "desk", gate_enabled=False)
    second = session_store_key(None, "desk", gate_enabled=False)
    assert first == second
    assert first != "default"
    assert first != "desk"
    alice = session_store_key("Bearer sk-alice", "desk", gate_enabled=True)
    bob = session_store_key("Bearer sk-bob", "desk", gate_enabled=True)
    assert alice != bob
    assert alice == session_store_key("Bearer sk-alice", "desk", gate_enabled=True)


def test_mapping_store_evicts_expired_and_overflow_sessions() -> None:
    store = MappingStore(max_sessions=1, ttl_seconds=10, max_entries=4)
    store.get("old")
    store._touched["old"] -= 100  # noqa: SLF001
    store.get("new")
    assert "old" not in store._sessions  # noqa: SLF001
    store.get("newer")
    assert list(store._sessions) == ["newer"]  # noqa: SLF001


def test_mapping_stops_storing_past_the_entry_cap() -> None:
    mapping = SessionMapping(max_entries=1)
    first = mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com")
    second = mapping.placeholder_for("EMAIL_ADDRESS", "bob@example.com")
    assert first == "<EMAIL_ADDRESS_1>"
    assert second == "<REDACTED>"
    assert mapping.restore_complete(second) == "<REDACTED>"
    assert mapping.restore_complete(first) == "ada@example.com"


def test_open_proxy_does_not_restore_another_requests_placeholders(monkeypatch: Any) -> None:
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
            content = (body or {}).get("messages", [{}])[0].get("content", "")
            captured["content"] = content
            return FakeResponse(
                {
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                }
            )

        async def close(self) -> None:
            return None

    class FakePipeline:
        def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
            needle = "ada@example.com"
            if needle in text:
                return text.replace(needle, mapping.placeholder_for("EMAIL_ADDRESS", needle))
            return text

    import enigmatic.server as server_mod

    monkeypatch.setattr(server_mod, "HttpProvider", FakeHttp)
    cfg = EnigmaticConfig(
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url="https://example.invalid/v1", api_key_env=None)},
    )
    client = TestClient(create_app(cfg, pipeline=FakePipeline()))  # type: ignore[arg-type]
    first = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "ada@example.com"}]},
    )
    assert first.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "ada@example.com"
    token = captured["content"]
    assert token.startswith("<EMAIL_ADDRESS_")
    second = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": token}]},
    )
    assert second.json()["choices"][0]["message"]["content"] == token


def test_internal_error_hides_exception_text(monkeypatch: Any) -> None:
    class Boom:
        def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
            raise RuntimeError("secret path /tmp/vault.pem")

    cfg = EnigmaticConfig(
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url="https://example.invalid/v1", api_key_env=None)},
    )
    client = TestClient(create_app(cfg, pipeline=Boom()), raise_server_exceptions=False)  # type: ignore[arg-type]
    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal error"
    assert "vault.pem" not in response.text


def test_rejects_oversized_json_body(monkeypatch: Any) -> None:
    import enigmatic.server as server_mod

    monkeypatch.setattr(server_mod, "MAX_BODY_BYTES", 64)
    cfg = EnigmaticConfig(
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url="https://example.invalid/v1", api_key_env=None)},
    )
    client = TestClient(create_app(cfg, pipeline=object()), raise_server_exceptions=False)  # type: ignore[arg-type]
    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "x" * 200}]},
    )
    assert response.status_code == 413


def test_pem_pattern_covers_the_key_body() -> None:
    pem = next(rec for rec in secret_recognizers() if "PEM_KEY" in rec.supported_entities)
    text = "-----BEGIN PRIVATE KEY-----\nSECRETKEYMATERIAL\n-----END PRIVATE KEY-----"
    matched = None
    for pattern in pem.patterns:
        found = re.search(pattern.regex, text)
        if found:
            matched = found.group(0)
            break
    assert matched is not None
    assert "SECRETKEYMATERIAL" in matched


def test_pem_recognizer_covers_the_key_body_with_presidio_flags() -> None:
    pem = next(rec for rec in secret_recognizers() if "PEM_KEY" in rec.supported_entities)
    key = "-----BEGIN PRIVATE KEY-----\nSECRETKEYMATERIAL\n-----END PRIVATE KEY-----"
    for text in (f"key:\n{key}\nthanks", "key:\n-----BEGIN PRIVATE KEY-----\nSECRETKEYMATERIAL\nmore"):
        results = pem.analyze(text, ["PEM_KEY"])
        assert results, text
        assert "SECRETKEYMATERIAL" in text[results[0].start : results[0].end]


def test_extra_secret_patterns_match() -> None:
    by_entity = {rec.supported_entities[0]: rec for rec in secret_recognizers()}
    samples = {
        "JWT": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.signature",
        "SLACK_TOKEN": "xoxb-1234567890-abcdefghij",
        "STRIPE_KEY": "sk_live_" + ("a" * 24),
        "CONNECTION_STRING": "postgres://user:secret@db.internal:5432/app",
    }
    for entity, sample in samples.items():
        recognizer = by_entity[entity]
        assert any(re.search(pattern.regex, sample) for pattern in recognizer.patterns), entity


def test_repeated_text_hits_the_analyzer_once() -> None:
    class CountingAnalyzer:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, **_kwargs: object) -> list[object]:
            self.calls += 1
            return []

    analyzer = CountingAnalyzer()
    pipeline = Pipeline(
        analyzer=analyzer,  # type: ignore[arg-type]
        anonymizer=None,  # type: ignore[arg-type]
        entities=["EMAIL_ADDRESS"],
        language="en",
        score_threshold=0.5,
    )
    mapping = SessionMapping()
    assert pipeline.anonymize_text("hello there", mapping) == "hello there"
    assert pipeline.anonymize_text("hello there", mapping) == "hello there"
    assert analyzer.calls == 1


def test_http_timeout_keeps_a_long_read_window() -> None:
    assert DEFAULT_TIMEOUT.connect == 10.0
    assert DEFAULT_TIMEOUT.read is not None
    assert DEFAULT_TIMEOUT.read >= 300.0


def test_router_refuses_jsonl_profile_without_tool_deny() -> None:
    cfg = EnigmaticConfig(
        default_profile="claude",
        jsonl={"claude": JsonlProfile(command="claude", extra_args=["--output-format", "json"])},
    )
    with pytest.raises(Exception, match="deny"):
        Router(cfg).resolve("claude/opus", "openai")


def test_bundled_jsonl_profiles_deny_tools() -> None:
    from enigmatic.config import load_config

    cfg = load_config()
    for name, profile in cfg.jsonl.items():
        assert jsonl_denies_tools(profile), name


def test_image_data_url_over_the_byte_cap_is_dropped(monkeypatch: Any) -> None:
    import enigmatic.presidio_ops.images as images

    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 8)
    payload = base64.b64encode(b"not-an-image-but-too-big").decode("ascii")
    out = redact_data_url(f"data:image/png;base64,{payload}", SessionMapping())
    assert "omitted" in out.lower()


@pytest.mark.asyncio
async def test_jsonl_prompt_is_stdin_not_argv(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeProc:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            captured["stdin"] = input
            line = b'{"type":"assistant.message","data":{"content":"ok"}}\n'
            return line, b""

    async def fake_exec(*argv: str, **kwargs: object) -> FakeProc:
        captured["argv"] = argv
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("enigmatic.providers.jsonl.command_on_path", lambda _cmd: True)
    profile = JsonlProfile(command="copilot", extra_args=["--deny-tool=shell", "--deny-tool=write"])
    text = await run_jsonl_prompt(profile, "secret prompt", model="gpt-5")
    assert text == "ok"
    assert "secret prompt" not in captured["argv"]
    assert captured["stdin"] == b"secret prompt"


@pytest.mark.asyncio
async def test_acp_runs_outside_the_process_cwd(monkeypatch: Any, tmp_path) -> None:
    from enigmatic.providers.acp import run_acp_prompt

    captured: dict[str, Any] = {}

    async def fake_sdk(*_args: object, **_kwargs: object) -> None:
        return None

    async def fake_exec(*argv: str, cwd: str | None = None, **kwargs: object) -> None:
        captured["cwd"] = cwd
        captured["argv"] = argv
        raise OSError("stop")

    monkeypatch.setattr("enigmatic.providers.acp._run_with_sdk", fake_sdk)
    monkeypatch.setattr("enigmatic.providers.acp.command_on_path", lambda _cmd: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    profile = AcpProfile(command="copilot", args=["--acp"], deny_tools=["shell"])
    with pytest.raises(OSError):
        await run_acp_prompt(profile, "hello")
    assert captured["cwd"]
    assert os.path.abspath(captured["cwd"]) != os.path.abspath(os.getcwd())
    joined = " ".join(captured["argv"])
    assert "--deny-tool=shell" in joined
