from __future__ import annotations

from enigmatic.config import AcpProfile, EnigmaticConfig, HttpProfile, JsonlProfile
from enigmatic.doctor import format_status


def test_format_status_prints_listener_pipeline_and_profiles(monkeypatch) -> None:
    monkeypatch.setattr(
        "enigmatic.doctor.spacy_status",
        lambda: {"ok": True, "model": "en_core_web_sm"},
    )
    monkeypatch.setattr(
        "enigmatic.doctor.tesseract_status",
        lambda: {"ok": True, "path": "/usr/bin/tesseract"},
    )
    monkeypatch.setattr("enigmatic.doctor.shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    cfg = EnigmaticConfig(
        listen_host="127.0.0.1",
        listen_port=47821,
        api_key=None,
        default_profile="openai",
        enabled_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        http={
            "openai": HttpProfile(type="openai", base_url="https://api.openai.com/v1"),
            "anthropic": HttpProfile(type="anthropic", base_url="https://api.anthropic.com"),
        },
        acp={"copilot": AcpProfile(command="copilot")},
        jsonl={"copilot": JsonlProfile(command="copilot")},
    )

    text = format_status(cfg)

    assert "Enigmatic 0.1.0" in text
    assert "127.0.0.1:47821" in text
    assert "openai" in text
    assert "auth" in text.lower()
    assert "off" in text
    assert "http://127.0.0.1:47821/v1" in text
    assert "en_core_web_sm" in text
    assert "/usr/bin/tesseract" in text
    assert "EMAIL_ADDRESS" in text
    assert "PHONE_NUMBER" in text
    assert "https://api.openai.com/v1" in text
    assert "https://api.anthropic.com" in text
    assert "copilot" in text
    assert "<" not in text
    assert "{" not in text


def test_format_status_marks_missing_spacy_and_tesseract(monkeypatch) -> None:
    monkeypatch.setattr("enigmatic.doctor.spacy_status", lambda: {"ok": False})
    monkeypatch.setattr("enigmatic.doctor.tesseract_status", lambda: {"ok": False, "path": None})
    monkeypatch.setattr("enigmatic.doctor.shutil.which", lambda _cmd: None)

    cfg = EnigmaticConfig(
        listen_host="0.0.0.0",
        listen_port=9,
        api_key="secret-local-key",
        default_profile="copilot",
        enabled_entities=["CREDIT_CARD"],
        acp={"copilot": AcpProfile(command="copilot")},
    )

    text = format_status(cfg)

    assert "missing" in text.lower()
    assert "fail-closed" in text.lower() or "vision" in text.lower()
    assert "enabled" in text.lower()
    assert "secret-local-key" not in text
    assert "CREDIT_CARD" in text
    assert "not installed" in text.lower() or "missing" in text.lower()
