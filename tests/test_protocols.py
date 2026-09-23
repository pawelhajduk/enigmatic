from enigmatic.protocols.packing import pack_openai_chat, pack_responses
from enigmatic.protocols.translate import (
    anthropic_to_openai_chat,
    openai_to_anthropic,
    responses_from_chat_completion,
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


def test_openai_to_anthropic_merges_tool_results_and_maps_developer() -> None:
    out = openai_to_anthropic(
        {
            "model": "claude",
            "messages": [
                {"role": "developer", "content": "dev rules"},
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "calling"}],
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": '{"x":1}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "one"},
                {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "two"}]},
            ],
            "metadata": {"trace": "x"},
            "user": "u-1",
        }
    )
    assert out["system"] == "dev rules"
    assistant = out["messages"][1]
    assert [block["type"] for block in assistant["content"]] == ["text", "tool_use", "tool_use"]
    results = out["messages"][2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["call_1", "call_2"]
    assert out["metadata"] == {"user_id": "u-1"}


def test_anthropic_to_openai_chat_maps_tool_choice_and_stream_usage() -> None:
    out = anthropic_to_openai_chat(
        {
            "model": "gpt-5",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "t", "disable_parallel_tool_use": True},
            "stop_sequences": ["END"],
        }
    )
    assert out["tool_choice"] == {"type": "function", "function": {"name": "t"}}
    assert out["parallel_tool_calls"] is False
    assert out["stream_options"] == {"include_usage": True}
    assert out["stop"] == ["END"]


def test_responses_from_chat_completion_restores_custom_tool_calls() -> None:
    completion = {
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [{"id": "call_1", "function": {"name": "apply_patch", "arguments": '{"input":"*** Begin"}'}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
    out = responses_from_chat_completion(completion, "m", {"tools": [{"type": "custom", "name": "apply_patch"}]})
    assert [item["type"] for item in out["output"]] == ["message", "custom_tool_call"]
    assert out["output"][1]["input"] == "*** Begin"
    assert out["usage"]["total_tokens"] == 5


def test_responses_tool_choice_objects_map_or_drop_without_crashing() -> None:
    tools = [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "shell"}]
    custom = responses_input_to_messages({"input": "x", "tools": tools, "tool_choice": {"type": "custom", "name": "apply_patch"}})
    assert custom["tool_choice"] == {"type": "function", "function": {"name": "apply_patch"}}
    allowed = responses_input_to_messages(
        {"input": "x", "tools": tools, "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": []}}
    )
    assert "tool_choice" not in allowed


def test_responses_stream_output_follows_output_index() -> None:
    from enigmatic.protocols.sse import ResponsesStreamBuilder

    builder = ResponsesStreamBuilder("m")
    builder.feed({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "a", "arguments": "{}"}}]}}]})
    builder.feed({"choices": [{"index": 0, "delta": {"content": "after the call"}}]})
    builder.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    events = builder.finish()
    output = events[-1][1]["response"]["output"]
    assert [item["type"] for item in output] == ["function_call", "message"]
    added = [payload for _event, payload in events if payload["type"] == "response.output_item.done"]
    assert {payload["item"]["type"]: payload["output_index"] for payload in added} == {"message": 1, "function_call": 0}


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
