"""Load KEY=VALUE files into the process environment.

Values are kept literal: `$` is not expanded, so API tokens survive unchanged.
Process variables already set win over files. Within one directory, `.env.local`
wins over `.env`. Earlier directories win over later ones.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}


def parse_env(text: str) -> dict[str, str]:
    """Parse a dotenv-style document. Later keys in the same text win."""
    if text.startswith("\ufeff"):
        text = text[1:]
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _KEY.match(key):
            continue
        values[key] = _parse_value(raw_value.strip())
    return values


def _parse_value(value: str) -> str:
    if value.startswith(("'", '"')):
        quote = value[0]
        chars: list[str] = []
        index = 1
        while index < len(value):
            char = value[index]
            if quote == '"' and char == "\\" and index + 1 < len(value):
                chars.append(_ESCAPES.get(value[index + 1], value[index + 1]))
                index += 2
                continue
            if char == quote:
                return "".join(chars)
            chars.append(char)
            index += 1
        return "".join(chars)
    comment = value.find(" #")
    if comment != -1:
        value = value[:comment].rstrip()
    return value


def load_env_files(
    directories: list[Path],
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply `.env` and `.env.local` from each directory.

    Returns the keys this call inserted. Keys already present in the target
    environment are left untouched.
    """
    target = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    seen: set[Path] = set()
    for directory in directories:
        resolved = directory.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        values: dict[str, str] = {}
        env_path = resolved / ".env"
        local_path = resolved / ".env.local"
        if env_path.is_file():
            values.update(parse_env(env_path.read_text(encoding="utf-8")))
        if local_path.is_file():
            values.update(parse_env(local_path.read_text(encoding="utf-8")))
        for key, value in values.items():
            if key in target or key in applied:
                continue
            applied[key] = value
            target[key] = value
    return applied
