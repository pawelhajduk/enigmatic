"""Session-stable placeholder mapping. Never persisted, never logged."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import defaultdict

PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")
REDACTED = "<REDACTED>"
DEFAULT_MAX_SESSIONS = 128
DEFAULT_SESSION_TTL_SECONDS = 60 * 60
DEFAULT_MAX_ENTRIES = 4096


def format_placeholder(entity_type: str, index: int) -> str:
    """Build `<EMAIL_ADDRESS_1>`-style tokens from a Presidio entity type."""
    token = entity_type.strip().upper().replace(" ", "_")
    if not token:
        token = "ENTITY"
    return f"<{token}_{index}>"


class SessionMapping:
    """Bidirectional map: original string <-> placeholder, per session."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._cache: dict[str, str] = {}
        self._sorted: list[tuple[str, str]] | None = None
        self._version = 0
        self._max_entries = max_entries

    def placeholder_for(self, entity_type: str, text: str) -> str:
        key = (entity_type.upper(), text)
        with self._lock:
            existing = self._forward.get(key)
            if existing is not None:
                return existing
            if len(self._forward) >= self._max_entries:
                return REDACTED
            self._counts[entity_type.upper()] += 1
            token = format_placeholder(entity_type, self._counts[entity_type.upper()])
            self._forward[key] = token
            self._reverse[token] = text
            self._sorted = None
            self._version += 1
            return token

    def original_for(self, placeholder: str) -> str | None:
        with self._lock:
            return self._reverse.get(placeholder)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def reverse_items(self) -> list[tuple[str, str]]:
        with self._lock:
            if self._sorted is None:
                self._sorted = sorted(self._reverse.items(), key=lambda item: len(item[0]), reverse=True)
            return list(self._sorted)

    def cached_anonymized(self, text: str) -> str | None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            return self._cache.get(digest)

    def remember_anonymized(self, text: str, anonymized: str) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if len(self._cache) >= self._max_entries and digest not in self._cache:
                return
            self._cache[digest] = anonymized

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

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMapping] = {}
        self._touched: dict[str, float] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    def ephemeral(self) -> SessionMapping:
        """A map that is not stored and cannot be reused by another request."""
        return SessionMapping(max_entries=self._max_entries)

    def get(self, session_id: str) -> SessionMapping:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            mapping = self._sessions.get(session_id)
            if mapping is None:
                mapping = SessionMapping(max_entries=self._max_entries)
                self._sessions[session_id] = mapping
            self._touched[session_id] = now
            self._evict(now)
            return mapping

    def _evict(self, now: float) -> None:
        expired = [key for key, touched in self._touched.items() if now - touched > self._ttl]
        for key in expired:
            self._sessions.pop(key, None)
            self._touched.pop(key, None)
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._touched, key=self._touched.get)
            self._sessions.pop(oldest, None)
            self._touched.pop(oldest, None)


STORE = MappingStore()
