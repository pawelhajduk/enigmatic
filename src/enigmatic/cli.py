"""Enigmatic command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import uvicorn

from enigmatic.auth import bind_requires_api_key
from enigmatic.config import load_config
from enigmatic.doctor import collect_doctor, format_status
from enigmatic.dry_run import anonymize_json_preview, anonymize_preview, format_preview
from enigmatic.presidio_ops.pipeline import build_pipeline
from enigmatic.server import create_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Enigmatic anonymizes prompts locally before they reach an LLM or agent CLI.",
)

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
    if bind_requires_api_key(bind_host, cfg.resolved_api_key):
        typer.echo(
            f"Refusing to listen on {bind_host} without ENIGMATIC_API_KEY.",
            err=True,
        )
        raise typer.Exit(code=1)
    cfg.listen_host = bind_host
    cfg.listen_port = bind_port
    fastapi_app = create_app(cfg)
    if not cfg.resolved_api_key:
        typer.echo(
            "warning: auth gate is off; any local account can use this proxy",
            err=True,
        )
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


def _read_prompt(prompt: list[str] | None, file: Path | None) -> str:
    if file is not None and prompt:
        typer.echo("pass either a prompt or --file, not both", err=True)
        raise typer.Exit(code=2)
    if file is not None:
        return file.read_text(encoding="utf-8")
    if prompt:
        return " ".join(prompt)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    typer.echo("pass a prompt, --file, or pipe stdin", err=True)
    raise typer.Exit(code=2)


@app.command("dry-run")
def dry_run(
    prompt: list[str] | None = typer.Argument(
        default=None,
        help="Prompt text. Omit to read --file or stdin.",
    ),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read the prompt from a file."),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Treat input as an OpenAI/Anthropic JSON body and walk it like the proxy.",
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Show a prompt before and after local anonymization. Does not call an LLM."""
    raw = _read_prompt(prompt, file)
    if raw == "":
        typer.echo("prompt is empty", err=True)
        raise typer.Exit(code=2)
    cfg = load_config(config)
    pipeline = build_pipeline(cfg)
    if as_json:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            typer.echo(f"invalid JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(payload, dict):
            typer.echo("JSON payload must be an object", err=True)
            raise typer.Exit(code=2)
        result = anonymize_json_preview(pipeline, payload)
    else:
        result = anonymize_preview(pipeline, raw)
    typer.echo(format_preview(result), nl=False)

