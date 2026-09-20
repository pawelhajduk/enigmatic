"""Streaming placeholder restore with a split-token buffer."""

from __future__ import annotations

import re

from enigmatic.presidio_ops.mapping import PLACEHOLDER_RE, SessionMapping

_INCOMPLETE_RE = re.compile(r"<[A-Z0-9_]*$")


class StreamRestorer:
    """Hold back partial `<EMAIL_1>` tokens that were split across SSE chunks."""

    def __init__(self, mapping: SessionMapping) -> None:
        self._mapping = mapping
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        longest = self._mapping.reverse_items()
        out: list[str] = []
        i = 0
        text = self._buf
        while i < len(text):
            if text[i] != "<":
                out.append(text[i])
                i += 1
                continue
            matched: str | None = None
            original: str | None = None
            for token, value in longest:
                if text.startswith(token, i):
                    matched = token
                    original = value
                    break
            if matched is None:
                generic = PLACEHOLDER_RE.match(text, i)
                if generic:
                    matched = generic.group(0)
                    original = self._mapping.original_for(matched) or matched
            if matched is not None and original is not None:
                out.append(original)
                i += len(matched)
                continue
            rest = text[i:]
            if _INCOMPLETE_RE.match(rest) and ">" not in rest:
                break
            out.append(text[i])
            i += 1
        self._buf = text[i:]
        return "".join(out)

    def flush(self) -> str:
        leftover = self._buf
        self._buf = ""
        return self._mapping.restore_complete(leftover)
