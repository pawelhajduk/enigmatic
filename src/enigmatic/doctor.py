"""Diagnostics for spaCy, Tesseract, and agent CLIs. Never prints secrets or prompt text."""

from __future__ import annotations

import shutil
from typing import Any

from enigmatic import __version__
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
        "version": __version__,
        "listen": f"{config.listen_host}:{config.listen_port}",
        "default_profile": config.default_profile,
        "auth": "enabled" if config.resolved_api_key else "off",
        "entities": list(config.enabled_entities),
        "base_url": f"http://{config.listen_host}:{config.listen_port}/v1",
        "spacy": spacy_status(),
        "tesseract": tesseract_status(),
        "http_profiles": http,
        "agent_clis": clis,
        "copilot": command_on_path("copilot"),
    }


def format_status(config: EnigmaticConfig) -> str:
    report = collect_doctor(config)
    spacy_ok = bool(report["spacy"].get("ok"))
    tess = report["tesseract"]
    tess_line = tess.get("path") if tess.get("ok") and tess.get("path") else "missing (vision fail-closed)"
    spacy_line = "en_core_web_sm ready" if spacy_ok else "en_core_web_sm missing"
    entities = ", ".join(report["entities"]) or "(none)"
    lines = [
        f"Enigmatic {report['version']}",
        "",
        "Listener",
        f"  bind             {report['listen']}",
        f"  default profile  {report['default_profile']}",
        f"  auth gate        {report['auth']}",
        f"  client base_url  {report['base_url']}",
        "",
        "Pipeline",
        f"  spaCy            {spacy_line}",
        f"  Tesseract        {tess_line}",
        f"  entities         {entities}",
        "",
        "HTTP profiles",
    ]
    http = report["http_profiles"]
    if not http:
        lines.append("  (none)")
    else:
        width = max(len(name) for name in http)
        for name, profile in http.items():
            lines.append(f"  {name:<{width}}  {profile['type']}  {profile['base_url']}")
    lines.append("")
    lines.append("Agent CLIs")
    clis = report["agent_clis"]
    if not clis:
        lines.append("  (none)")
    else:
        width = max(len(name) for name in clis)
        for name, profile in clis.items():
            flag = "on PATH" if profile.get("ok") else "not installed"
            transport = str(profile.get("transport") or "cli")
            if profile.get("jsonl_command") and transport == "acp":
                transport = "acp+jsonl"
            command = profile.get("command") or ""
            lines.append(f"  {name:<{width}}  {transport}  {command}  ({flag})")
    lines.append("")
    return "\n".join(lines)
