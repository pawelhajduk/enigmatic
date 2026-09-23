from enigmatic.openai_walk import collect_strings


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
    assert '{"email":"ada@example.com"}' in strings
    # Schema structure stays intact. Description and example text is still scanned.
    assert "ada@example.com in schema" in strings
    assert "object" not in strings
    assert "openai/gpt-4o" not in strings
    assert "lookup" in strings  # tool name in the call is not under parameters
