"""ACP stdio client for an existing agent binary.

Enigmatic is the client. The coding agent remains the installed CLI
(`agent acp`, `copilot --acp --stdio`, `gemini --acp`, and the other
registry rows). Tool permission, filesystem, and terminal requests are
denied so a chat completion cannot write the disk.

The turn loop is implemented here, not via the Python SDK's `prompt()`
helper. ACP v1 returns assistant text before `session/prompt` resolves.
ACP v2 resolves that RPC as soon as the prompt is accepted and streams
the reply afterwards. This client waits for either shape.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from typing import Any

from enigmatic import __version__
from enigmatic.config import AcpProfile

logger = logging.getLogger("enigmatic.acp")

AGENT_TIMEOUT_SECONDS = 120.0
_MESSAGE_KINDS = frozenset({"agent_message_chunk", "agent_message"})
DEFAULT_DENY_TOOLS = (
    "shell",
    "bash",
    "write",
    "read",
    "edit",
    "url",
    "memory",
    "fetch",
)
DEFAULT_DENY_FLAGS = ("--deny-tool={tool}", "--excluded-tools={tool}")


def command_on_path(command: str) -> bool:
    return shutil.which(command) is not None


def deny_permission(_request: dict[str, Any]) -> dict[str, Any]:
    return {"outcome": {"outcome": "cancelled"}}


def acp_argv(profile: AcpProfile) -> list[str]:
    """CLI argv with tool-deny flags. Permission requests are still cancelled in-process."""
    templates = DEFAULT_DENY_FLAGS if profile.deny_flags is None else profile.deny_flags
    tools = list(dict.fromkeys([*profile.deny_tools, *DEFAULT_DENY_TOOLS]))
    argv = [profile.command, *profile.args]
    for tool in tools:
        argv.extend(template.replace("{tool}", tool) for template in templates)
    return argv


class AcpError(RuntimeError):
    pass


class AgentTranscript:
    """Assemble assistant text from chunk and whole-message updates."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._parts: dict[str, str] = {}
        self._seq = 0

    def add(self, params: dict[str, Any]) -> None:
        update = params.get("update") or params
        if not isinstance(update, dict):
            return
        kind = update.get("sessionUpdate") or update.get("session_update")
        if kind not in _MESSAGE_KINDS:
            return
        text = _content_text(update.get("content"))
        raw_id = update.get("messageId") or update.get("message_id")
        message_id = raw_id if isinstance(raw_id, str) and raw_id else ""
        replace = kind == "agent_message"
        if not message_id:
            self._seq += 1
            message_id = f"anon-{self._seq}"
        if message_id not in self._parts:
            self._order.append(message_id)
            self._parts[message_id] = ""
        if replace:
            self._parts[message_id] = text
        else:
            self._parts[message_id] += text

    def text(self) -> str:
        return "".join(self._parts[message_id] for message_id in self._order)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _turn_finished(update: dict[str, Any]) -> bool:
    kind = update.get("sessionUpdate") or update.get("session_update")
    if kind != "state_update":
        return False
    state = str(update.get("state") or update.get("status") or "")
    if state in {"idle", "completed", "complete"}:
        return True
    return bool(update.get("stopReason") or update.get("stop_reason"))


def _inbound_result(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "session/request_permission":
        return deny_permission(params)
    # Blocking extension requests (cursor/ask_question, cursor/create_plan)
    # must be answered or the agent waits forever. Cancel them.
    return {"outcome": {"outcome": "cancelled"}}


def _usable_model(model: str | None) -> str | None:
    if model is None or model == "" or model == "default":
        return None
    return model


async def _run_with_sdk(*_args: object, **_kwargs: object) -> str | None:
    """Optional SDK hook. The raw client owns the turn, including ACP v2."""
    return None


async def run_acp_prompt(
    profile: AcpProfile,
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
) -> str:
    if not command_on_path(profile.command):
        raise AcpError(
            f"{profile.command} is not on PATH. Install that coding-agent CLI and log in; "
            "Enigmatic only forwards the anonymized prompt to it."
        )
    argv = acp_argv(profile)
    usable_model = _usable_model(model)
    # Agents start in an empty directory so a tool that ignores the deny list
    # is not already sitting in the user's project.
    owned_dir: tempfile.TemporaryDirectory[str] | None = None
    if cwd is None:
        owned_dir = tempfile.TemporaryDirectory(prefix="enigmatic-agent-")
        cwd = owned_dir.name
    try:
        try:
            text = await asyncio.wait_for(
                _run_with_sdk(argv, prompt, usable_model, cwd),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            if text:
                return text
        except TimeoutError as exc:
            raise AcpError(f"{profile.command} timed out") from exc
        except Exception as exc:
            logger.info("ACP SDK path failed, using raw JSON-RPC: %s", exc)
        try:
            return await asyncio.wait_for(
                _run_raw_jsonrpc(profile, prompt, usable_model, cwd, argv),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise AcpError(f"{profile.command} timed out") from exc
    finally:
        if owned_dir is not None:
            owned_dir.cleanup()


async def _run_raw_jsonrpc(
    profile: AcpProfile,
    prompt: str,
    model: str | None,
    cwd: str | None,
    argv: list[str] | None = None,
) -> str:
    command = argv if argv is not None else acp_argv(profile)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        raise AcpError("ACP process missing stdio")

    loop = asyncio.get_running_loop()
    pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
    transcript = AgentTranscript()
    turn_done = asyncio.Event()
    stderr_lines: list[str] = []
    next_id = 0

    async def write(payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def read_stdout() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                for fut in pending.values():
                    if not fut.done():
                        fut.set_exception(AcpError("ACP process closed stdout"))
                pending.clear()
                turn_done.set()
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            method = message.get("method")
            if isinstance(method, str):
                params = message.get("params") if isinstance(message.get("params"), dict) else {}
                if method == "session/update":
                    transcript.add(params)
                    update = params.get("update")
                    if isinstance(update, dict) and _turn_finished(update):
                        turn_done.set()
                    continue
                msg_id = message.get("id")
                if msg_id is not None:
                    await write(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": _inbound_result(method, params),
                        }
                    )
                continue
            msg_id = message.get("id")
            fut = pending.get(msg_id) if isinstance(msg_id, int) else None
            if fut is None or fut.done():
                continue
            if "error" in message:
                fut.set_exception(AcpError(str(message["error"])))
            else:
                result = message.get("result") or {}
                fut.set_result(result if isinstance(result, dict) else {})

    async def read_stderr() -> None:
        assert proc.stderr is not None
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            stderr_lines.append(line.decode("utf-8", errors="replace"))

    async def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        msg_id = next_id
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        pending[msg_id] = fut
        await write({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, AGENT_TIMEOUT_SECONDS)
        finally:
            pending.pop(msg_id, None)

    stdout_task = asyncio.create_task(read_stdout())
    stderr_task = asyncio.create_task(read_stderr())
    try:
        await rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "enigmatic", "version": __version__},
            },
        )
        if profile.auth_method:
            try:
                await rpc("authenticate", {"methodId": profile.auth_method})
            except AcpError as exc:
                logger.info("ACP authenticate continued with the CLI's existing login: %s", exc)
        session = await rpc(
            "session/new",
            {"cwd": cwd or ".", "mcpServers": []},
        )
        session_id = session.get("sessionId") or session.get("session_id")
        if model:
            try:
                await rpc("session/set_model", {"sessionId": session_id, "modelId": model})
            except AcpError as exc:
                logger.info("ACP session/set_model skipped: %s", exc)
        result = await rpc(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]},
        )
        stop = result.get("stopReason") or result.get("stop_reason")
        if not stop:
            try:
                await asyncio.wait_for(turn_done.wait(), AGENT_TIMEOUT_SECONDS)
            except TimeoutError:
                if not transcript.text():
                    err = "".join(stderr_lines).strip()
                    raise AcpError(err or "ACP agent produced no assistant text") from None
        text = transcript.text()
        if not text:
            err = "".join(stderr_lines).strip()
            raise AcpError(err or "ACP agent produced no assistant text")
        return text
    finally:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                logger.debug("ACP stdin close failed", exc_info=True)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except TimeoutError:
            proc.kill()
            await proc.wait()
