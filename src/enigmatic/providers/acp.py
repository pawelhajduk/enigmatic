"""ACP stdio client. Deny every tool permission. Prefer the official SDK when present."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from enigmatic.config import AcpProfile

logger = logging.getLogger("enigmatic.acp")

DenyHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def command_on_path(command: str) -> bool:
    return shutil.which(command) is not None


def deny_permission(_request: dict[str, Any]) -> dict[str, Any]:
    return {"outcome": {"outcome": "cancelled"}}


class AcpError(RuntimeError):
    pass


async def run_acp_prompt(
    profile: AcpProfile,
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
) -> str:
    if not command_on_path(profile.command):
        raise AcpError(f"{profile.command} is not on PATH")
    argv = [profile.command, *profile.args]
    for tool in profile.deny_tools:
        argv.append(f"--deny-tool={tool}")
        argv.append(f"--excluded-tools={tool}")
    try:
        text = await _run_with_sdk(argv, prompt, model, cwd)
        if text is not None:
            return text
    except Exception as exc:
        logger.info("ACP SDK path failed, using raw JSON-RPC: %s", exc)
    return await _run_raw_jsonrpc(argv, prompt, model, cwd)


async def _run_with_sdk(
    argv: list[str],
    prompt: str,
    model: str | None,
    cwd: str | None,
) -> str | None:
    try:
        from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
        from acp.interfaces import Client
    except Exception:
        return None

    chunks: list[str] = []

    class DenyClient(Client):  # type: ignore[misc]
        async def request_permission(self, params: Any) -> Any:
            return deny_permission(params if isinstance(params, dict) else {})

        async def session_update(self, params: Any) -> None:
            update = params.get("update") if isinstance(params, dict) else getattr(params, "update", None)
            if update is None:
                return
            kind = update.get("sessionUpdate") if isinstance(update, dict) else getattr(update, "session_update", None)
            content = update.get("content") if isinstance(update, dict) else getattr(update, "content", None)
            if kind in {"agent_message_chunk", "agent_message"} and content is not None:
                text = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)

    async with spawn_agent_process(DenyClient(), argv[0], *argv[1:], cwd=cwd) as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities={})
        session = await conn.new_session(cwd=cwd or ".", mcp_servers=[])
        session_id = getattr(session, "session_id", None) or (
            session.get("sessionId") if isinstance(session, dict) else None
        )
        if model:
            setter = getattr(conn, "set_session_model", None) or getattr(conn, "set_model", None)
            if setter is not None:
                try:
                    await setter(session_id=session_id, model_id=model)
                except Exception as exc:
                    logger.info("ACP set_model skipped: %s", exc)
        await conn.prompt(session_id=session_id, prompt=[text_block(prompt)])
    return "".join(chunks)


async def _run_raw_jsonrpc(
    argv: list[str],
    prompt: str,
    model: str | None,
    cwd: str | None,
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    if proc.stdin is None or proc.stdout is None:
        raise AcpError("ACP process missing stdio")

    next_id = 0
    chunks: list[str] = []

    async def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        msg_id = next_id
        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise AcpError("ACP process closed stdout")
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if message.get("method") == "session/update":
                _collect_update(message.get("params") or {}, chunks)
                continue
            if message.get("method") == "session/request_permission":
                reply = {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": deny_permission(message.get("params") or {}),
                }
                proc.stdin.write((json.dumps(reply) + "\n").encode("utf-8"))
                await proc.stdin.drain()
                continue
            if message.get("id") == msg_id:
                if "error" in message:
                    raise AcpError(str(message["error"]))
                result = message.get("result") or {}
                if not isinstance(result, dict):
                    return {}
                return result

    try:
        await rpc("initialize", {"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "enigmatic"}})
        session = await rpc("session/new", {"cwd": cwd or ".", "mcpServers": []})
        session_id = session.get("sessionId") or session.get("session_id")
        if model:
            try:
                await rpc("session/set_model", {"sessionId": session_id, "modelId": model})
            except AcpError as exc:
                logger.info("ACP session/set_model skipped: %s", exc)
        await rpc(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]},
        )
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
    return "".join(chunks)


def _collect_update(params: dict[str, Any], chunks: list[str]) -> None:
    update = params.get("update") or params
    kind = update.get("sessionUpdate") or update.get("session_update")
    content = update.get("content") or {}
    if kind in {"agent_message_chunk", "agent_message"}:
        text = content.get("text") if isinstance(content, dict) else None
        if isinstance(text, str):
            chunks.append(text)
