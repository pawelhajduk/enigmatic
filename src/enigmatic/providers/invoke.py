"""Run an anonymized prompt on an installed coding-agent CLI.

This module does not implement an agent. It selects the ACP process or the
CLI prompt mode already configured for the model prefix, then returns the
agent's text.
"""

from __future__ import annotations

import logging

from enigmatic.config import EnigmaticConfig
from enigmatic.presidio_ops.mapping import STORE, SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline
from enigmatic.providers.acp import run_acp_prompt
from enigmatic.providers.jsonl import run_jsonl_prompt
from enigmatic.providers.router import Route, Router

logger = logging.getLogger("enigmatic.invoke")


class AgentRouteError(RuntimeError):
    pass


def _model_arg(route: Route) -> str | None:
    if not route.model or route.model == "default":
        return None
    return route.model


async def invoke_agent(route: Route, prompt: str) -> str:
    """Send `prompt` to the route's existing CLI. `prompt` is already anonymized."""
    model = _model_arg(route)
    if route.kind == "acp" and route.acp is not None:
        try:
            return await run_acp_prompt(route.acp, prompt, model=model)
        except Exception as exc:
            logger.info("ACP failed, JSONL fallback: %s", exc)
            if route.jsonl is None:
                raise
    if route.jsonl is None:
        raise AgentRouteError(f"No agent CLI configured for {route.profile_id}")
    return await run_jsonl_prompt(
        route.jsonl,
        prompt,
        model=model,
        parser=route.jsonl.parser,
    )


async def run_layered_prompt(
    config: EnigmaticConfig,
    pipeline: Pipeline,
    model: str,
    text: str,
    session_key: str = "cli",
) -> str:
    """Anonymize `text`, call the installed agent, and restore placeholders."""
    mapping: SessionMapping = STORE.get(session_key)
    anonymized = pipeline.anonymize_text(text, mapping)
    route = Router(config).resolve(model, "openai")
    if route.kind not in {"acp", "jsonl"}:
        raise AgentRouteError(
            f"{route.profile_id} is an HTTP API profile. "
            "This command calls an installed agent CLI via ACP or its prompt mode. "
            "Use a model such as cursor/, copilot/, claude/, or codex/."
        )
    reply = await invoke_agent(route, anonymized)
    return mapping.restore_complete(reply)
