"""Route a request to HTTP, ACP, or JSONL based on model prefix / default profile."""

from __future__ import annotations

from dataclasses import dataclass

from enigmatic.config import AcpProfile, EnigmaticConfig, HttpProfile, JsonlProfile
from enigmatic.protocols.translate import strip_model_prefix
from enigmatic.providers.jsonl import jsonl_denies_tools


class RouteError(Exception):
    """The request mapped to a profile that is not safe to run."""


@dataclass
class Route:
    kind: str  # openai, anthropic, acp, jsonl
    profile_id: str
    model: str
    http: HttpProfile | None = None
    acp: AcpProfile | None = None
    jsonl: JsonlProfile | None = None


class Router:
    def __init__(self, config: EnigmaticConfig) -> None:
        self.config = config

    def resolve(self, model: str | None, inbound: str) -> Route:
        """inbound is `openai` (chat/responses/embeddings) or `anthropic`."""
        raw = model or ""
        prefix, bare = strip_model_prefix(raw)
        hint = prefix or self.config.default_profile

        if hint in self.config.http:
            profile = self.config.http[hint]
            kind = "anthropic" if profile.type == "anthropic" else "openai"
            return Route(kind=kind, profile_id=hint, model=bare or raw, http=profile)

        if hint in self.config.acp:
            return Route(
                kind="acp",
                profile_id=hint,
                model=bare or raw,
                acp=self.config.acp[hint],
                jsonl=self.config.jsonl.get(hint),
            )

        if hint in self.config.jsonl:
            return self._jsonl_route(hint, bare or raw, self.config.jsonl[hint])

        if inbound == "anthropic" and "anthropic" in self.config.http:
            profile = self.config.http["anthropic"]
            return Route(kind="anthropic", profile_id="anthropic", model=bare or raw, http=profile)

        default = self.config.default_profile
        if default in self.config.http:
            profile = self.config.http[default]
            kind = "anthropic" if profile.type == "anthropic" else "openai"
            return Route(kind=kind, profile_id=default, model=bare or raw or "", http=profile)

        if default in self.config.acp:
            return Route(
                kind="acp",
                profile_id=default,
                model=bare or raw,
                acp=self.config.acp[default],
                jsonl=self.config.jsonl.get(default),
            )
        if default in self.config.jsonl:
            return self._jsonl_route(default, bare or raw, self.config.jsonl[default])

        raise KeyError(f"No provider profile for model {model!r} (hint={hint!r})")

    def _jsonl_route(self, profile_id: str, model: str, profile: JsonlProfile) -> Route:
        if not jsonl_denies_tools(profile):
            raise RouteError(f"Agent profile {profile_id} does not deny tools")
        return Route(kind="jsonl", profile_id=profile_id, model=model, jsonl=profile)
