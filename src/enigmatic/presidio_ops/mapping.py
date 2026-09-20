"""Session-stable placeholder mapping. Never persisted, never logged."""

from __future__ import annotations

import re
import threading
from collections import defaultdict

PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")


def format_placeholder(entity_type: str, index: int) -> str:
    """Build `<EMAIL_ADDRESS_1>`-style tokens from a Presidio entity type."""
    token = entity_type.strip().upper().replace(" ", "_")
    if not token:
        token = "ENTITY"
    return f"<{token}_{index}>"


class SessionMapping:
    """Bidirectional map: original string <-> placeholder, per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def placeholder_for(self, entity_type: str, text: str) -> str:
        key = (entity_type.upper(), text)
        with self._lock:
            existing = self._forward.get(key)
            if existing is not None:
                return existing
            self._counts[entity_type.upper()] += 1
            token = format_placeholder(entity_type, self._counts[entity_type.upper()])
            self._forward[key] = token
            self._reverse[token] = text
            return token

    def original_for(self, placeholder: str) -> str | None:
        with self._lock:
            return self._reverse.get(placeholder)

    def reverse_items(self) -> list[tuple[str, str]]:
        with self._lock:
            return sorted(self._reverse.items(), key=lambda item: len(item[0]), reverse=True)

    def restore_complete(self, text: str) -> str:
        """Replace placeholders in a finished string. Longest token first."""
        result = text
        for token, original in self.reverse_items():
            result = result.replace(token, original)
        return result

    def restore_complete_any(self, value: object) -> object:
        if isinstance(value, str):
            return self.restore_complete(value)
        if isinstance(value, list):
            return [self.restore_complete_any(item) for item in value]
        if isinstance(value, dict):
            return {key: self.restore_complete_any(item) for key, item in value.items()}
        return value


class MappingStore:
    """In-memory session maps. Keys are opaque session ids, never entity text."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMapping] = {}

    def get(self, session_id: str) -> SessionMapping:
        with self._lock:
            mapping = self._sessions.get(session_id)
            if mapping is None:
                mapping = SessionMapping()
                self._sessions[session_id] = mapping
            return mapping


STORE = MappingStore()
