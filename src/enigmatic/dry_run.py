"""Local before/after anonymization preview. Never forwards to an LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass

from enigmatic.openai_walk import walk
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.pipeline import Pipeline


@dataclass(frozen=True)
class DryRunResult:
    before: str
    after: str
    replacements: list[tuple[str, str]]


def anonymize_preview(pipeline: Pipeline, text: str) -> DryRunResult:
    mapping = SessionMapping()
    after = pipeline.anonymize_text(text, mapping)
    return DryRunResult(before=text, after=after, replacements=_replacements(mapping))


def anonymize_json_preview(pipeline: Pipeline, payload: dict[str, object]) -> DryRunResult:
    mapping = SessionMapping()
    walked = walk(payload, pipeline, mapping)
    return DryRunResult(
        before=json.dumps(payload, indent=2, ensure_ascii=False),
        after=json.dumps(walked, indent=2, ensure_ascii=False),
        replacements=_replacements(mapping),
    )


def format_preview(result: DryRunResult) -> str:
    lines = [
        "before",
        "------",
        result.before,
        "",
        "after",
        "-----",
        result.after,
        "",
    ]
    if not result.replacements:
        lines.append("no entities replaced")
        lines.append("")
        return "\n".join(lines)
    lines.append("placeholders")
    lines.append("------------")
    width = max(len(token) for token, _original in result.replacements)
    for token, original in result.replacements:
        lines.append(f"  {token:<{width}}  {original}")
    lines.append("")
    return "\n".join(lines)


def _replacements(mapping: SessionMapping) -> list[tuple[str, str]]:
    return sorted(mapping.reverse_items(), key=lambda item: item[0])
