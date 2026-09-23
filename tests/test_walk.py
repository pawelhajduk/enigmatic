import json
from typing import Any

from enigmatic.openai_walk import collect_strings, walk
from enigmatic.presidio_ops.mapping import SessionMapping


class _MarkEverything:
    """Pipeline stand-in that rewrites every string it is given."""

    def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
        return f"<SCANNED>{text}"


def _walk(body: dict[str, Any]) -> Any:
    return walk(body, _MarkEverything(), SessionMapping())  # type: ignore[arg-type]


def test_codex_responses_opaque_fields_are_byte_identical() -> None:
    grammar = 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF'
    body = {
        "model": "openai/gpt-5-codex",
        "instructions": "be careful",
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "0199-abc",
        "reasoning": {"effort": "high", "summary": "auto"},
        "tool_choice": "auto",
        "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Edit files", "format": {"type": "grammar", "syntax": "lark", "definition": grammar}},
            {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
        ],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev note"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAAB" + "x" * 64},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell", "arguments": '{"cmd": ["ls"]}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"},
            {"type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch", "input": "*** Begin Patch"},
            {"type": "compaction", "encrypted_content": "gAAAAC" + "y" * 64},
            {"type": "message", "role": "user", "content": [{"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0x"}]},
        ],
    }
    out = _walk(body)
    for key in ("model", "store", "include", "prompt_cache_key", "reasoning", "tool_choice"):
        assert out[key] == body[key], key
    assert out["tools"][0]["format"] == body["tools"][0]["format"]
    assert out["tools"][0]["name"] == "apply_patch"
    assert out["tools"][1]["parameters"] == body["tools"][1]["parameters"]
    assert out["input"][1] == body["input"][1]
    assert out["input"][5] == body["input"][5]
    assert out["input"][2]["call_id"] == "call_1"
    assert out["input"][2]["name"] == "shell"
    assert out["input"][0]["role"] == "developer"
    assert out["input"][6]["content"][0]["file_data"] == body["input"][6]["content"][0]["file_data"]
    # User-visible text is still scanned.
    assert out["instructions"] == "<SCANNED>be careful"
    assert out["input"][3]["output"] == "<SCANNED>file.txt"
    assert out["input"][4]["input"] == "<SCANNED>*** Begin Patch"
    assert json.loads(out["input"][2]["arguments"]) == {"cmd": ["<SCANNED>ls"]}


def test_schema_properties_named_type_and_name_do_not_break_the_walk() -> None:
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["file", "dir"]},
            "name": {"type": "string", "description": "for ada@example.com"},
        },
    }
    body = {
        "tools": [{"type": "function", "function": {"name": "search", "parameters": schema}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
    }
    out = _walk(body)
    assert out["tools"][0]["function"]["parameters"]["properties"]["type"] == schema["properties"]["type"]
    assert out["response_format"]["json_schema"]["schema"]["properties"]["type"]["enum"] == ["file", "dir"]
    assert "for ada@example.com" in collect_strings(body)


def test_anthropic_tool_use_input_is_scanned_as_data() -> None:
    body = {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "send", "input": {"type": "email", "id": "ada@example.com"}}]}]}
    block = _walk(body)["messages"][0]["content"][0]
    assert block["id"] == "toolu_1" and block["name"] == "send"
    assert block["input"] == {"type": "<SCANNED>email", "id": "<SCANNED>ada@example.com"}


def test_same_history_anonymizes_identically_in_fresh_vaults() -> None:
    """Placeholder numbering follows first appearance, so resent history keeps its prefix cache."""

    class Emails:
        def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
            for email in ("bob@example.com", "ada@example.com"):
                text = text.replace(email, mapping.placeholder_for("EMAIL_ADDRESS", email))
            return text

    turn1 = {"input": [{"role": "user", "content": "ada@example.com then bob@example.com"}]}
    turn2 = {"input": [*turn1["input"], {"role": "user", "content": "and bob@example.com again"}]}
    first = walk(turn1, Emails(), SessionMapping())  # type: ignore[arg-type]
    second = walk(turn2, Emails(), SessionMapping())  # type: ignore[arg-type]
    assert isinstance(first, dict) and isinstance(second, dict)
    assert second["input"][: len(first["input"])] == first["input"]
    assert "ada@example.com" not in json.dumps(second)


def test_unchanged_arguments_keep_original_formatting() -> None:
    class Noop:
        def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
            return text

    raw = '{ "path": "a.py",  "n": 1 }'
    out = walk({"input": [{"type": "function_call", "arguments": raw}]}, Noop(), SessionMapping())  # type: ignore[arg-type]
    assert isinstance(out, dict)
    assert out["input"][0]["arguments"] == raw


def test_dropped_image_becomes_a_schema_valid_text_part(monkeypatch: Any) -> None:
    import enigmatic.presidio_ops.images as images

    monkeypatch.setattr(images, "tesseract_available", lambda: False)
    url = "data:image/png;base64,iVBORw0KGgo="
    body = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url, "detail": "low"}}]}],
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": url}]}],
        "anthropic": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}],
        "remote": [{"type": "image_url", "image_url": {"url": "https://example.com/ada@example.com.png"}}],
    }
    out = _walk(body)
    chat_part = out["messages"][0]["content"][0]
    assert chat_part["type"] == "text" and "omitted" in chat_part["text"]
    responses_part = out["input"][0]["content"][0]
    assert responses_part["type"] == "input_text" and "omitted" in responses_part["text"]
    assert out["anthropic"][0]["type"] == "text"
    assert out["remote"] == body["remote"]


def test_walk_anonymizes_message_and_tool_arguments_not_schemas() -> None:
    body = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "user", "content": "email ada@example.com"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"email\":\"ada@example.com\"}",
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "ada@example.com in schema"}
                        },
                    },
                },
            }
        ],
    }
    strings = collect_strings(body)
    assert "email ada@example.com" in strings
    # Arguments are scanned as decoded JSON values, so restores can re-escape exactly.
    assert "ada@example.com" in strings
    assert '{"email":"ada@example.com"}' not in strings
    # Schema structure stays intact. Description and example text is still scanned.
    assert "ada@example.com in schema" in strings
    assert "object" not in strings
    assert "openai/gpt-4o" not in strings
    # Tool names must keep matching between the tool list and calls.
    assert "lookup" not in strings
    assert "call_1" not in strings
