"""Enigmatic command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from enigmatic.config import load_config
from enigmatic.doctor import collect_doctor
from enigmatic.server import create_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Enigmatic anonymizes prompts locally before they reach an LLM or agent CLI.",
)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address. Defaults to config listen_host."),
    port: int | None = typer.Option(None, help="Bind port. Defaults to 47821."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to providers.yaml."),
) -> None:
    """Start the local proxy. Point OpenAI/Anthropic clients at http://HOST:PORT/v1."""
    cfg = load_config(config)
    bind_host = host or cfg.listen_host
    bind_port = port or cfg.listen_port
    cfg.listen_host = bind_host
    cfg.listen_port = bind_port
    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=bind_host, port=bind_port, log_level="info")


@app.command()
def status(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Print listener and pipeline diagnostics as JSON (no secrets, no prompt text)."""
    cfg = load_config(config)
    typer.echo(json.dumps(collect_doctor(cfg), indent=2))


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Check spaCy, Tesseract, HTTP profiles, and whether copilot/agent CLIs are on PATH."""
    cfg = load_config(config)
    report = collect_doctor(cfg)
    spacy_ok = bool(report["spacy"].get("ok"))
    tess_ok = bool(report["tesseract"].get("ok"))
    missing_spacy = (
        "MISSING en_core_web_sm — run: python -m spacy download en_core_web_sm"
    )
    typer.echo(f"listen:     {report['listen']}")
    typer.echo(f"spaCy:      {'ok' if spacy_ok else missing_spacy}")
    tess_line = (
        f"ok {report['tesseract'].get('path')}"
        if tess_ok
        else "missing (vision requests fail closed)"
    )
    typer.echo(f"Tesseract:  {tess_line}")
    typer.echo(f"copilot:    {'on PATH' if report['copilot'] else 'not installed'}")
    typer.echo("HTTP profiles:")
    for name, profile in report["http_profiles"].items():
        typer.echo(f"  - {name}: {profile['type']} {profile['base_url']}")
    typer.echo("Agent CLIs:")
    for name, profile in report["agent_clis"].items():
        flag = "ok" if profile.get("ok") else "missing"
        typer.echo(f"  - {name}: {profile.get('command')} ({flag})")
    if not spacy_ok:
        raise typer.Exit(code=1)
