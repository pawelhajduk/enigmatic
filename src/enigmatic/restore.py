"""Streaming placeholder restore with a split-token buffer."""

from __future__ import annotations

import re

from enigmatic.presidio_ops.mapping import SessionMapping

_INCOMPLETE_RE = re.compile(r"<[A-Z0-9_]*$")


class StreamRestorer:
    """Hold back partial `<EMAIL_1>` tokens that were split across SSE chunks."""

    def __init__(self, mapping: SessionMapping) -> None:
        self._mapping = mapping
        self._buf = ""
        self._version = -1
        self._pattern: re.Pattern[str] | None = None
        self._lookup: dict[str, str] = {}

    def _sync(self) -> None:
        version = self._mapping.version
        if version == self._version:
            return
        items = self._mapping.reverse_items()
        self._lookup = dict(items)
        if items:
            # Longest token is first, and Python uses the leftmost alternative.
            self._pattern = re.compile("|".join(re.escape(token) for token, _value in items))
        else:
            self._pattern = None
        self._version = version

    def _replace(self, text: str) -> str:
        if not text:
            return ""
        self._sync()
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda match: self._lookup.get(match.group(0), match.group(0)), text)

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        hold = ""
        incomplete = _INCOMPLETE_RE.search(self._buf)
        if incomplete is not None and ">" not in incomplete.group(0):
            hold = incomplete.group(0)
            ready = self._buf[: incomplete.start()]
        else:
            ready = self._buf
        self._buf = hold
        return self._replace(ready)

    def flush(self) -> str:
        leftover = self._buf
        self._buf = ""
        return self._mapping.restore_complete(leftover)
