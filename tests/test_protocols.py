from enigmatic.protocols.packing import pack_openai_chat, pack_responses
from enigmatic.protocols.translate import (
    openai_to_anthropic,
    responses_input_to_messages,
    strip_model_prefix,
)


def test_strip_model_prefix() -> None:
    assert strip_model_prefix("openai/gpt-4o") == ("openai", "gpt-4o")
    assert strip_model_prefix("gpt-4o") == ("", "gpt-4o")


def test_openai_to_anthropic_moves_system_and_tools() -> None:
    body = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "hi"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "max_tokens": 32,
    }
    out = openai_to_anthropic(body)
    assert out["model"] == "claude-sonnet-4-5"
    assert out["system"] == "Be brief"
    assert out["messages"][0]["role"] == "user"
    assert out["tools"][0]["name"] == "lookup"
    assert out["tools"][0]["input_schema"]["type"] == "object"


def test_pack_openai_chat_includes_roles() -> None:
    text = pack_openai_chat(
        {
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hello"},
            ]
        }
    )
    assert "SYSTEM: rules" in text
    assert "USER: hello" in text


def test_responses_input_to_messages() -> None:
    chat = responses_input_to_messages(
        {"model": "gpt-4o", "instructions": "sys", "input": "hello"}
    )
    assert chat["messages"][0] == {"role": "system", "content": "sys"}
    assert chat["messages"][1] == {"role": "user", "content": "hello"}
    assert pack_responses({"input": "hello"}) == "hello"
