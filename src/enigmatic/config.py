"""Load Enigmatic YAML config and named provider profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from enigmatic.envfile import load_env_files

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_CONF_DIR = PACKAGE_DIR.parents[1] / "conf"
DEFAULT_PROVIDERS_PATH = REPO_CONF_DIR / "providers.yaml"
DEFAULT_RECOGNIZERS_PATH = REPO_CONF_DIR / "default_recognizers.yaml"


class HttpProfile(BaseModel):
    type: str = "openai"
    base_url: str
    api_key_env: str | None = None
    auth_header: str = "Authorization"
    api_version: str | None = None


class AcpProfile(BaseModel):
    """Spawn an existing ACP agent (`agent acp`, `copilot --acp`)."""

    command: str
    args: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)
    # Uses the CLI's own login (for example cursor_login). Not an API key.
    auth_method: str | None = None


class JsonlProfile(BaseModel):
    """One-shot prompt mode of an existing CLI (`claude -p`, `codex exec`)."""

    command: str
    prompt_flag: str = "-p"
    extra_args: list[str] = Field(default_factory=list)
    parser: str = "copilot"
    prompt_stdin: bool = False
    model_flag: str | None = "--model"


class EnigmaticConfig(BaseModel):
    listen_host: str = "127.0.0.1"
    listen_port: int = 47821
    api_key: str | None = None
    api_key_env: str | None = "ENIGMATIC_API_KEY"
    default_profile: str = "openai"
    analyzer_conf: str | None = None
    score_threshold: float = 0.5
    language: str = "en"
    http: dict[str, HttpProfile] = Field(default_factory=dict)
    acp: dict[str, AcpProfile] = Field(default_factory=dict)
    jsonl: dict[str, JsonlProfile] = Field(default_factory=dict)
    enabled_entities: list[str] = Field(default_factory=list)

    @property
    def resolved_api_key(self) -> str | None:
        """Access token clients must send. Environment wins over the YAML literal."""
        if self.api_key_env:
            from_env = os.environ.get(self.api_key_env, "").strip()
            if from_env:
                return from_env
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root in {path} must be a mapping")
    return loaded


def _env_directories(providers_path: Path) -> list[Path]:
    """Directories searched for `.env` and `.env.local`, most specific first."""
    resolved = providers_path.expanduser().resolve()
    directories = [Path.cwd(), resolved.parent]
    if resolved.parent.name == "conf":
        directories.append(resolved.parent.parent)
    return directories


def load_config(path: Path | None = None) -> EnigmaticConfig:
    """Load providers.yaml plus optional recognizer allowlist.

    `.env` and `.env.local` are applied first so `ENIGMATIC_API_KEY` and
    upstream `*_API_KEY` variables are available. Existing process variables
    are not overwritten.
    """
    providers_path = path or DEFAULT_PROVIDERS_PATH
    if not providers_path.is_file():
        bundled = PACKAGE_DIR / "data" / "providers.yaml"
        providers_path = bundled if bundled.is_file() else providers_path
    if providers_path.is_file():
        load_env_files(_env_directories(providers_path))
    data = _read_yaml(providers_path)

    recognizers_path = DEFAULT_RECOGNIZERS_PATH
    analyzer_conf = data.get("analyzer_conf")
    if isinstance(analyzer_conf, str) and analyzer_conf.strip():
        recognizers_path = Path(analyzer_conf)
    elif not recognizers_path.is_file():
        bundled = PACKAGE_DIR / "data" / "default_recognizers.yaml"
        if bundled.is_file():
            recognizers_path = bundled
    rec = _read_yaml(recognizers_path)
    entities = rec.get("enabled_entities")
    if isinstance(entities, list):
        data["enabled_entities"] = [str(item) for item in entities]
    return EnigmaticConfig.model_validate(data)
