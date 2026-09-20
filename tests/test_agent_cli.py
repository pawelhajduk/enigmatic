from enigmatic.providers.acp import deny_permission
from enigmatic.providers.jsonl import parse_claude_jsonl, parse_copilot_jsonl


def test_deny_permission_cancels_tools() -> None:
    result = deny_permission({"toolCall": {"toolCallId": "1", "title": "rm -rf"}})
    assert result["outcome"]["outcome"] == "cancelled"


def test_parse_claude_stream_json() -> None:
    line = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hi there"}]}}'
    )
    assert parse_claude_jsonl(line) == "hi there"


def test_parse_copilot_ignores_non_assistant() -> None:
    assert parse_copilot_jsonl('{"type":"system","data":{"content":"nope"}}') is None
