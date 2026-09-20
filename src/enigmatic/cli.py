"""Enigmatic command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from enigmatic.config import load_config
from enigmatic.doctor import collect_doctor, format_status
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
    typer.echo(format_status(cfg))
    uvicorn.run(fastapi_app, host=bind_host, port=bind_port, log_level="warning")


@app.command()
def status(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Print listener and pipeline diagnostics (no secrets, no prompt text)."""
    cfg = load_config(config)
    typer.echo(format_status(cfg))


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Same dump as status; exits 1 if spaCy is missing."""
    cfg = load_config(config)
    typer.echo(format_status(cfg))
    if not collect_doctor(cfg)["spacy"].get("ok"):
        typer.echo("run: python -m spacy download en_core_web_sm", err=True)
        raise typer.Exit(code=1)
