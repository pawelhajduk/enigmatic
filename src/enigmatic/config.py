"""Load Enigmatic YAML config and named provider profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

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
    command: str
    args: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)


class JsonlProfile(BaseModel):
    command: str
    prompt_flag: str = "-p"
    extra_args: list[str] = Field(default_factory=list)


class EnigmaticConfig(BaseModel):
    listen_host: str = "127.0.0.1"
    listen_port: int = 47821
    api_key: str | None = None
    default_profile: str = "openai"
    analyzer_conf: str | None = None
    score_threshold: float = 0.5
    language: str = "en"
    http: dict[str, HttpProfile] = Field(default_factory=dict)
    acp: dict[str, AcpProfile] = Field(default_factory=dict)
    jsonl: dict[str, JsonlProfile] = Field(default_factory=dict)
    enabled_entities: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root in {path} must be a mapping")
    return loaded


def load_config(path: Path | None = None) -> EnigmaticConfig:
    """Load providers.yaml plus optional recognizer allowlist."""
    providers_path = path or DEFAULT_PROVIDERS_PATH
    if not providers_path.is_file():
        bundled = PACKAGE_DIR / "data" / "providers.yaml"
        providers_path = bundled if bundled.is_file() else providers_path
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
