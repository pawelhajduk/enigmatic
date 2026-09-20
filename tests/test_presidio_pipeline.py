"""Presidio-backed tests. Skipped if spaCy model is not installed."""

from __future__ import annotations

import pytest

from enigmatic.config import EnigmaticConfig
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.pipeline import build_pipeline, spacy_status
from enigmatic.presidio_ops.secrets import secret_recognizers

pytestmark = pytest.mark.skipif(not spacy_status().get("ok"), reason="en_core_web_sm not installed")


def test_pipeline_emails_and_github_tokens() -> None:
    cfg = EnigmaticConfig(enabled_entities=["EMAIL_ADDRESS", "GITHUB_TOKEN"])
    pipeline = build_pipeline(cfg)
    mapping = SessionMapping()
    text = "ping ada@example.com with ghp_" + ("A" * 36)
    out = pipeline.anonymize_text(text, mapping)
    assert "ada@example.com" not in out
    assert "<EMAIL_ADDRESS_1>" in out
    assert mapping.restore_complete(out).startswith("ping ada@example.com")


def test_secret_recognizers_exist() -> None:
    names = {rec.supported_entities[0] for rec in secret_recognizers()}
    assert names >= {"GITHUB_TOKEN", "OPENAI_KEY", "AWS_ACCESS_KEY", "PEM_KEY"}
