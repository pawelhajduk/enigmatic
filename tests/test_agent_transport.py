"""The proxy is a layer over installed agent CLIs and ACP processes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enigmatic.config import AcpProfile, EnigmaticConfig, HttpProfile, JsonlProfile, load_config
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.providers.acp import AcpError, run_acp_prompt
from enigmatic.providers.invoke import run_layered_prompt
from enigmatic.providers.jsonl import build_jsonl_argv, parse_codex_jsonl, run_jsonl_prompt
from enigmatic.providers.router import Router
from enigmatic.server import create_app

FAKE_AGENT = r"""
import json
import sys

mode = sys.argv[1]
log_path = sys.argv[2]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log(line):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


while True:
    msg = read()
    method = msg.get("method")
    mid = msg.get("id")
    if method:
        log("in:" + method)
    if method == "initialize":
        caps = (msg.get("params") or {}).get("clientCapabilities") or {}
        log("caps:" + json.dumps(caps, sort_keys=True))
        send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": 1,
                    "authMethods": [{"id": "cursor_login"}],
                },
            }
        )
    elif method == "authenticate":
        log("auth:" + json.dumps(msg.get("params")))
        if mode == "auth-error":
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": "already"}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "sess-1"}})
    elif method == "session/set_model":
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "unsupported"}})
    elif method == "session/prompt":
        text = msg["params"]["prompt"][0]["text"]
        log("prompt:" + text)
        if mode == "v2":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "sess-1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "m1",
                            "content": {"type": "text", "text": "hello "},
                        },
                    },
                }
            )
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "sess-1",
                        "update": {
                            "sessionUpdate": "agent_message",
                            "messageId": "m1",
                            "content": [{"type": "text", "text": "hello restored"}],
                        },
                    },
                }
            )
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "sess-1",
                        "update": {
                            "sessionUpdate": "state_update",
                            "state": "idle",
                            "stopReason": "end_turn",
                        },
                    },
                }
            )
        elif mode == "question":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 77,
                    "method": "cursor/ask_question",
                    "params": {"toolCallId": "q", "questions": []},
                }
            )
            reply = read()
            log("reply:" + json.dumps(reply.get("result")))
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "after question"},
                        }
                    },
                }
            )
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 50,
                    "method": "session/request_permission",
                    "params": {"toolCall": {"toolCallId": "1", "title": "rm"}},
                }
            )
            reply = read()
            log("perm:" + json.dumps(reply.get("result")))
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "seen " + text},
                        }
                    },
                }
            )
            send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    elif mid is not None and method:
        send({"jsonrpc": "2.0", "id": mid, "result": {"outcome": {"outcome": "cancelled"}}})
"""

FAKE_CODEX = r"""
import json
import sys

log_path = sys.argv[1]
prompt = sys.stdin.read()
with open(log_path, "w", encoding="utf-8") as handle:
    handle.write(prompt)
print(
    json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "reply <EMAIL_ADDRESS_1>"},
        }
    ),
    flush=True,
)
"""


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


def _agent_profile(tmp_path: Path, mode: str, *, auth_method: str | None) -> tuple[AcpProfile, Path]:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    log_path = tmp_path / "agent.log"
    profile = AcpProfile(
        command=sys.executable,
        args=[str(script), mode, str(log_path)],
        auth_method=auth_method,
    )
    return profile, log_path


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


def test_acp_v1_denies_tools_and_returns_agent_text(tmp_path: Path) -> None:
    profile, log_path = _agent_profile(tmp_path, "v1", auth_method="cursor_login")
    text = _run(run_acp_prompt(profile, "hello ada@example.com", model="default"))
    log = log_path.read_text(encoding="utf-8")
    assert text == "seen hello ada@example.com"
    assert "in:authenticate" in log
    assert '"methodId": "cursor_login"' in log
    assert '"outcome": "cancelled"' in log
    assert '"readTextFile": false' in log
    assert '"terminal": false' in log


def test_acp_v2_replaces_chunks_after_prompt_ack(tmp_path: Path) -> None:
    profile, _log_path = _agent_profile(tmp_path, "v2", auth_method=None)
    text = _run(run_acp_prompt(profile, "ping"))
    assert text == "hello restored"


def test_acp_continues_when_cli_login_is_already_present(tmp_path: Path) -> None:
    profile, log_path = _agent_profile(tmp_path, "auth-error", auth_method="cursor_login")
    text = _run(run_acp_prompt(profile, "ping"))
    log = log_path.read_text(encoding="utf-8")
    assert "in:authenticate" in log
    assert "prompt:ping" in log
    assert text == "seen ping"


def test_acp_answers_blocking_client_questions(tmp_path: Path) -> None:
    profile, log_path = _agent_profile(tmp_path, "question", auth_method=None)
    text = _run(run_acp_prompt(profile, "ping"))
    log = log_path.read_text(encoding="utf-8")
    assert text == "after question"
    assert '"outcome": "cancelled"' in log


def test_acp_missing_binary_names_the_cli() -> None:
    profile = AcpProfile(command="enigmatic-no-such-agent")
    with pytest.raises(AcpError, match="enigmatic-no-such-agent"):
        _run(run_acp_prompt(profile, "hi"))


def test_parse_codex_item_completed() -> None:
    line = '{"type":"item.completed","item":{"type":"agent_message","text":"from codex"}}'
    assert parse_codex_jsonl(line) == "from codex"
    assert parse_codex_jsonl('{"type":"thread.started","thread_id":"t"}') is None


def test_jsonl_stdin_calls_existing_binary(tmp_path: Path) -> None:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    log_path = tmp_path / "prompt.txt"
    profile = JsonlProfile(
        command=sys.executable,
        extra_args=[str(script), str(log_path), "--deny-tool=shell"],
        prompt_flag="",
        prompt_stdin=True,
        parser="codex",
        model_flag=None,
    )
    text = _run(run_jsonl_prompt(profile, "secret <EMAIL_ADDRESS_1>", model="gpt-5"))
    assert text == "reply <EMAIL_ADDRESS_1>"
    assert log_path.read_text(encoding="utf-8") == "secret <EMAIL_ADDRESS_1>"


def test_layered_prompt_redacts_before_the_cli_and_restores_the_reply(tmp_path: Path) -> None:
    profile, log_path = _agent_profile(tmp_path, "v1", auth_method=None)
    cfg = EnigmaticConfig(
        default_profile="cursor",
        api_key_env=None,
        acp={"cursor": profile},
    )
    text = _run(
        run_layered_prompt(
            cfg,
            FakePipeline(),  # type: ignore[arg-type]
            "cursor/default",
            "email ada@example.com",
            session_key="layer-test",
        )
    )
    sent = log_path.read_text(encoding="utf-8")
    assert "<EMAIL_ADDRESS_1>" in sent
    assert "ada@example.com" not in sent
    assert text == "seen email ada@example.com"


def test_layered_prompt_refuses_api_key_profiles() -> None:
    cfg = EnigmaticConfig(
        default_profile="openai",
        api_key_env=None,
        http={"openai": HttpProfile(type="openai", base_url="https://api.openai.com/v1")},
    )
    with pytest.raises(RuntimeError, match="cursor/"):
        _run(run_layered_prompt(cfg, FakePipeline(), "openai/gpt-4o", "hi"))  # type: ignore[arg-type]


def test_http_proxy_uses_agent_cli_without_an_upstream_key(tmp_path: Path) -> None:
    profile, log_path = _agent_profile(tmp_path, "v2", auth_method=None)
    cfg = EnigmaticConfig(
        default_profile="cursor",
        api_key_env=None,
        acp={"cursor": profile},
        enabled_entities=["EMAIL_ADDRESS"],
    )
    client = TestClient(create_app(cfg, pipeline=FakePipeline()))  # type: ignore[arg-type]
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Enigmatic-Session": "agent-transport"},
        json={
            "model": "cursor/default",
            "messages": [{"role": "user", "content": "write ada@example.com"}],
        },
    )
    assert response.status_code == 200
    body = response.json()["choices"][0]["message"]["content"]
    sent = log_path.read_text(encoding="utf-8")
    assert "ada@example.com" not in sent
    assert "<EMAIL_ADDRESS_1>" in sent
    assert body == "hello restored"


def test_codex_argv_is_the_existing_exec_cli() -> None:
    profile = load_config().jsonl["codex"]
    argv = build_jsonl_argv(profile, "do not put this on argv", "gpt-5.4")
    assert argv[0] == "codex"
    assert argv[1:3] == ["exec", "--json"]
    assert "--sandbox" in argv
    assert "read-only" in argv
    assert "--skip-git-repo-check" in argv
    assert argv[-3:] == ["-m", "gpt-5.4", "-"]
    assert "do not put this on argv" not in argv


def test_bundled_registry_calls_installed_clis() -> None:
    cfg = load_config()
    assert cfg.acp["cursor"].command == "agent"
    assert cfg.acp["cursor"].args == ["acp"]
    assert cfg.acp["cursor"].auth_method == "cursor_login"
    assert cfg.jsonl["codex"].command == "codex"
    assert cfg.jsonl["codex"].parser == "codex"
    assert cfg.jsonl["codex"].prompt_stdin is True
    assert cfg.jsonl["claude"].parser == "claude"
    cursor = Router(cfg).resolve("cursor/default", "openai")
    assert cursor.kind == "acp"
    assert cursor.model == "default"
    codex = Router(cfg).resolve("codex/gpt-5.4", "openai")
    assert codex.kind == "jsonl"
    assert codex.model == "gpt-5.4"


def test_prompt_command_describes_the_agent_layer() -> None:
    from typer.testing import CliRunner

    from enigmatic.cli import app

    result = CliRunner().invoke(app, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "installed" in result.stdout.lower()
    assert "ACP" in result.stdout or "acp" in result.stdout.lower()
