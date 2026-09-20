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


async def run_jsonl_prompt(
    profile: JsonlProfile,
    prompt: str,
    model: str | None = None,
    parser: str = "copilot",
) -> str:
    if not command_on_path(profile.command):
        raise JsonlError(f"{profile.command} is not on PATH")
    argv = [profile.command, *profile.extra_args, profile.prompt_flag, prompt]
    if model:
        argv.extend(["--model", model])
    logger.info("jsonl spawn %s", profile.command)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode not in (0, None) and not stdout:
        err = stderr.decode("utf-8", errors="replace")
        raise JsonlError(err.strip() or f"{profile.command} exited {proc.returncode}")
    texts: list[str] = []
    last_complete = ""
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        piece = parse_claude_jsonl(line) if parser == "claude" else parse_copilot_jsonl(line)
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
