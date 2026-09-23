from __future__ import annotations

from enigmatic.envfile import load_env_files, parse_env


def test_parse_env_reads_export_quotes_and_comments() -> None:
    parsed = parse_env(
        """
        # local access token
        export ENIGMATIC_API_KEY="sk-local token"
        OPENAI_API_KEY='sk-upstream'
        EMPTY=
        QUOTED_HASH="keep # this"
        UNQUOTED=value # trailing comment
        """
    )

    assert parsed["ENIGMATIC_API_KEY"] == "sk-local token"
    assert parsed["OPENAI_API_KEY"] == "sk-upstream"
    assert parsed["EMPTY"] == ""
    assert parsed["QUOTED_HASH"] == "keep # this"
    assert parsed["UNQUOTED"] == "value"


def test_load_env_files_prefers_local_file_without_overriding_process(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "ENIGMATIC_API_KEY=from-env\nOPENAI_API_KEY=from-env\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("ENIGMATIC_API_KEY=from-local\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")
    monkeypatch.setenv("ENIGMATIC_API_KEY", "")
    monkeypatch.delenv("ENIGMATIC_API_KEY")

    applied = load_env_files([tmp_path])

    assert applied["ENIGMATIC_API_KEY"] == "from-local"
    assert "OPENAI_API_KEY" not in applied
