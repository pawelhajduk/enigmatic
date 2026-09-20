"""Pack OpenAI / Anthropic / Responses payloads into one agent-CLI transcript."""

from __future__ import annotations

from typing import Any


def _text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return ""


def pack_openai_chat(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        chunks.append(f"SYSTEM: {system}")
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).upper()
            text = _text_from_content(message.get("content"))
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict):
                        text = (
                            text
                            + f"\n[tool_call {fn.get('name', '')} {call.get('id', '')}] "
                            + str(fn.get("arguments", ""))
                        )
            if message.get("role") == "tool":
                text = f"[tool_result {message.get('tool_call_id', '')}] {text}"
            if text:
                chunks.append(f"{role}: {text}")
    prompt = body.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        chunks.append(f"USER: {prompt}")
    if isinstance(prompt, list):
        chunks.append("USER: " + "\n".join(str(item) for item in prompt))
    return "\n\n".join(chunks).strip() or " "


def pack_anthropic(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        chunks.append(f"SYSTEM: {system}")
    elif isinstance(system, list):
        text = _text_from_content(system)
        if text:
            chunks.append(f"SYSTEM: {text}")
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).upper()
            chunks.append(f"{role}: {_text_from_content(message.get('content'))}")
    return "\n\n".join(chunks).strip() or " "


def pack_responses(body: dict[str, Any]) -> str:
    incoming = body.get("input")
    if isinstance(incoming, str):
        return incoming
    if isinstance(incoming, list):
        chunks: list[str] = []
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            chunks.append(f"SYSTEM: {instructions}")
        for item in incoming:
            if isinstance(item, str):
                chunks.append(f"USER: {item}")
                continue
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", item.get("type", "user"))).upper()
            chunks.append(f"{role}: {_text_from_content(item.get('content') or item.get('text'))}")
        return "\n\n".join(chunks).strip() or " "
    return pack_openai_chat(body)
