"""Diagnostics for spaCy, Tesseract, and agent CLIs. Never prints secrets or prompt text."""

from __future__ import annotations

import shutil
from typing import Any

from enigmatic.config import EnigmaticConfig
from enigmatic.presidio_ops.images import tesseract_status
from enigmatic.presidio_ops.pipeline import spacy_status
from enigmatic.providers.acp import command_on_path


def collect_doctor(config: EnigmaticConfig) -> dict[str, Any]:
    clis: dict[str, Any] = {}
    seen: set[str] = set()
    for name, profile in config.acp.items():
        path = shutil.which(profile.command)
        clis[name] = {
            "transport": "acp",
            "command": profile.command,
            "path": path,
            "ok": path is not None,
        }
        seen.add(name)
    for name, profile in config.jsonl.items():
        path = shutil.which(profile.command)
        entry = clis.get(name, {})
        entry.update(
            {
                "jsonl_command": profile.command,
                "jsonl_path": path,
                "jsonl_ok": path is not None,
            }
        )
        if name not in seen:
            entry.setdefault("transport", "jsonl")
            entry.setdefault("command", profile.command)
            entry.setdefault("ok", path is not None)
        clis[name] = entry
    http = {
        name: {"type": profile.type, "base_url": profile.base_url, "api_key_env": profile.api_key_env}
        for name, profile in config.http.items()
    }
    return {
        "listen": f"{config.listen_host}:{config.listen_port}",
        "spacy": spacy_status(),
        "tesseract": tesseract_status(),
        "http_profiles": http,
        "agent_clis": clis,
        "copilot": command_on_path("copilot"),
    }
