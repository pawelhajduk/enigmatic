from __future__ import annotations

import json

from enigmatic.config import EnigmaticConfig
from enigmatic.dry_run import anonymize_json_preview, anonymize_preview, format_preview
from enigmatic.presidio_ops.mapping import STORE, SessionMapping


class FakePipeline:
    entities = ["EMAIL_ADDRESS"]
    language = "en"
    score_threshold = 0.5

    def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
        needle = "ada@example.com"
        if needle in text:
            token = mapping.placeholder_for("EMAIL_ADDRESS", needle)
            return text.replace(needle, token)
        return text


def test_preview_shows_before_after_and_placeholders() -> None:
    result = anonymize_preview(FakePipeline(), "write ada@example.com")  # type: ignore[arg-type]
    text = format_preview(result)
    assert "write ada@example.com" in text
    assert "<EMAIL_ADDRESS_1>" in text
    assert "ada@example.com" in text
    before_idx = text.index("before")
    after_idx = text.index("after")
    orig_idx = text.index("write ada@example.com")
    anon_idx = text.index("write <EMAIL_ADDRESS_1>")
    assert before_idx < orig_idx < after_idx < anon_idx
    assert "placeholders" in text


def test_preview_with_no_entities() -> None:
    result = anonymize_preview(FakePipeline(), "hello there")  # type: ignore[arg-type]
    text = format_preview(result)
    assert "hello there" in text
    assert "no entities replaced" in text
    assert "<EMAIL_ADDRESS_" not in text


def test_json_preview_walks_messages_not_model() -> None:
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "write ada@example.com"}],
    }
    result = anonymize_json_preview(FakePipeline(), payload)  # type: ignore[arg-type]
    after = json.loads(result.after)
    assert after["model"] == "openai/gpt-4o"
    assert after["messages"][0]["content"] == "write <EMAIL_ADDRESS_1>"
    assert "ada@example.com" in result.before
    assert "<EMAIL_ADDRESS_1>" in result.after


def test_preview_does_not_use_global_session_store() -> None:
    sessions_before = dict(STORE._sessions)
    anonymize_preview(FakePipeline(), "write ada@example.com")  # type: ignore[arg-type]
    assert STORE._sessions == sessions_before


def test_dry_run_command_prints_preview(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner

    from enigmatic.cli import app

    monkeypatch.setattr("enigmatic.cli.build_pipeline", lambda _cfg: FakePipeline())
    monkeypatch.setattr("enigmatic.cli.load_config", lambda _path=None: EnigmaticConfig())

    runner = CliRunner()
    result = runner.invoke(app, ["dry-run", "write", "ada@example.com"])
    assert result.exit_code == 0, result.output
    assert "write ada@example.com" in result.output
    assert "write <EMAIL_ADDRESS_1>" in result.output

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello there", encoding="utf-8")
    file_result = runner.invoke(app, ["dry-run", "--file", str(prompt_file)])
    assert file_result.exit_code == 0, file_result.output
    assert "no entities replaced" in file_result.output
