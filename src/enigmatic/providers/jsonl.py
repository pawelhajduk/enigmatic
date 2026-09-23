"""Prompt-mode JSONL backends (Copilot -p, Claude stream-json)."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from enigmatic.config import JsonlProfile

logger = logging.getLogger("enigmatic.jsonl")


class JsonlError(RuntimeError):
    pass


def command_on_path(command: str) -> bool:
    return shutil.which(command) is not None


def parse_copilot_jsonl(line: str) -> str | None:
    """Extract assistant text from a Copilot --output-format json event."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    kind = str(event.get("type") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else event
    if kind in {"assistant.message", "assistant"}:
        content = data.get("content") or data.get("text") or event.get("content")
        if isinstance(content, str):
            return content
    if kind.endswith("assistant.message") or kind == "message":
        content = data.get("content") if isinstance(data, dict) else None
        if isinstance(content, str):
            return content
    delta = data.get("delta") if isinstance(data, dict) else None
    if isinstance(delta, str) and "assistant" in kind:
        return delta
    return None


def parse_codex_jsonl(line: str) -> str | None:
    """Extract assistant text from a `codex exec --json` event."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") in {"agent_message", "message"}:
        text = item.get("text")
        if isinstance(text, str):
            return text
    if event.get("type") in {"agent_message", "message"} and isinstance(event.get("text"), str):
        return event["text"]
    content = event.get("content")
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    return None


def parse_claude_jsonl(line: str) -> str | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts) or None
        if isinstance(content, str):
            return content
    if event.get("type") == "result" and isinstance(event.get("result"), str):
        return event["result"]
    return None


def build_jsonl_argv(profile: JsonlProfile, prompt: str, model: str | None) -> list[str]:
    """Argv for an existing CLI. The prompt is not an API request."""
    if profile.prompt_stdin:
        args = list(profile.extra_args)
        if model and profile.model_flag:
            if args and args[-1] == "-":
                args = [*args[:-1], profile.model_flag, model, "-"]
            else:
                args = [*args, profile.model_flag, model]
        argv = [profile.command, *args]
        if profile.prompt_flag:
            argv.append(profile.prompt_flag)
        return argv
    argv = [profile.command, *profile.extra_args]
    if profile.prompt_flag:
        argv.append(profile.prompt_flag)
    argv.append(prompt)
    if model and profile.model_flag:
        argv.extend([profile.model_flag, model])
    return argv


def _parse_jsonl_line(parser: str, line: str) -> str | None:
    if parser == "copilot":
        return parse_copilot_jsonl(line)
    if parser == "claude":
        return parse_claude_jsonl(line)
    if parser == "codex":
        return parse_codex_jsonl(line)
    if parser == "text":
        return line
    raise JsonlError(f"Unknown JSONL parser {parser}")


async def run_jsonl_prompt(
    profile: JsonlProfile,
    prompt: str,
    model: str | None = None,
    parser: str | None = None,
) -> str:
    if not command_on_path(profile.command):
        raise JsonlError(
            f"{profile.command} is not on PATH. Install that coding-agent CLI and log in; "
            "Enigmatic only forwards the anonymized prompt to it."
        )
    chosen = parser or profile.parser
    argv = build_jsonl_argv(profile, prompt, model if model and model != "default" else None)
    logger.info("jsonl spawn %s", profile.command)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if profile.prompt_stdin else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin_bytes = prompt.encode("utf-8") if profile.prompt_stdin else None
    stdout, stderr = await proc.communicate(stdin_bytes)
    if proc.returncode not in (0, None) and not stdout:
        err = stderr.decode("utf-8", errors="replace")
        raise JsonlError(err.strip() or f"{profile.command} exited {proc.returncode}")
    texts: list[str] = []
    last_complete = ""
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        piece = _parse_jsonl_line(chosen, line)
        if piece:
            last_complete = piece
            texts.append(piece)
    if last_complete:
        return last_complete
    stripped = stdout.decode("utf-8", errors="replace").strip()
    if stripped and not stripped.startswith("{"):
        return stripped
    if texts:
        return texts[-1]
    err = stderr.decode("utf-8", errors="replace").strip()
    raise JsonlError(err or f"{profile.command} produced no assistant text")


def extract_assistant_text(events: list[dict[str, Any]]) -> str:
    last = ""
    for event in events:
        kind = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else event
        if "assistant" in kind or kind in {"assistant", "assistant.message"}:
            content = data.get("content") or data.get("text")
            if isinstance(content, str):
                last = content
    return last
