from enigmatic.config import AcpProfile, EnigmaticConfig, HttpProfile, JsonlProfile, load_config
from enigmatic.providers.jsonl import extract_assistant_text, parse_copilot_jsonl
from enigmatic.providers.router import Router


def test_load_bundled_config() -> None:
    cfg = load_config()
    assert cfg.listen_port == 47821
    assert "openai" in cfg.http
    assert "anthropic" in cfg.http
    assert "copilot" in cfg.acp
    assert "EMAIL_ADDRESS" in cfg.enabled_entities
    assert "PERSON" not in cfg.enabled_entities


def test_router_model_prefix() -> None:
    cfg = EnigmaticConfig(
        default_profile="openai",
        http={
            "openai": HttpProfile(type="openai", base_url="https://api.openai.com/v1"),
            "anthropic": HttpProfile(type="anthropic", base_url="https://api.anthropic.com"),
            "groq": HttpProfile(type="openai", base_url="https://api.groq.com/openai/v1"),
        },
        acp={"copilot": AcpProfile(command="copilot", args=["--acp", "--stdio"])},
        jsonl={"copilot": JsonlProfile(command="copilot")},
    )
    router = Router(cfg)
    openai = router.resolve("openai/gpt-4o", "openai")
    assert openai.kind == "openai" and openai.model == "gpt-4o"
    groq = router.resolve("groq/llama-3.1-70b", "openai")
    assert groq.kind == "openai" and groq.profile_id == "groq"
    anthropic = router.resolve("anthropic/claude-sonnet-4-5", "openai")
    assert anthropic.kind == "anthropic" and anthropic.model == "claude-sonnet-4-5"
    copilot = router.resolve("copilot/gpt-5", "openai")
    assert copilot.kind == "acp" and copilot.jsonl is not None


def test_parse_copilot_jsonl_assistant_message() -> None:
    line = '{"type":"assistant.message","data":{"content":"hello from copilot"}}'
    assert parse_copilot_jsonl(line) == "hello from copilot"
    events = [
        {"type": "assistant.message", "data": {"content": "first"}},
        {"type": "assistant.message", "data": {"content": "final answer"}},
    ]
    assert extract_assistant_text(events) == "final answer"
