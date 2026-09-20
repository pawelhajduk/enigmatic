"""OpenAI Chat Completions <-> Anthropic Messages field mapping (Meridian-shaped)."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def strip_model_prefix(model: str) -> tuple[str, str]:
    """Return (profile_hint, bare_model). `openai/gpt-4o` -> (`openai`, `gpt-4o`)."""
    if "/" in model:
        prefix, rest = model.split("/", 1)
        return prefix, rest
    return "", model


def openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    _, model = strip_model_prefix(str(body.get("model", "")))
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = content if isinstance(content, str) else _blocks_to_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed = {"_raw": raw_args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id", f"tool_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""),
                        "input": parsed if isinstance(parsed, dict) else {"value": parsed},
                    }
                )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        messages.append({"role": "user", "content": _openai_content_to_anthropic(content)})

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            continue
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )

    out: dict[str, Any] = {
        "model": model or str(body.get("model", "")),
        "messages": messages,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    if tools:
        out["tools"] = tools
    for key in ("temperature", "top_p", "stop_sequences", "metadata"):
        if key in body:
            out[key] = body[key]
    if "stop" in body and body["stop"] is not None:
        stop = body["stop"]
        out["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    return out


def anthropic_to_openai_chat(body: dict[str, Any]) -> dict[str, Any]:
    _, model = strip_model_prefix(str(body.get("model", "")))
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        messages.append({"role": "system", "content": _blocks_to_text(system)})
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        content = message.get("content")
        if role == "assistant":
            text, tool_calls = _anthropic_assistant_to_openai(content)
            item: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
            continue
        if isinstance(content, list):
            tool_results = [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]
            texts = [block for block in content if not (isinstance(block, dict) and block.get("type") == "tool_result")]
            if tool_results:
                for block in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", "")
                            if isinstance(block.get("content"), str)
                            else json.dumps(block.get("content")),
                        }
                    )
            if texts:
                messages.append({"role": "user", "content": _anthropic_content_to_openai(texts)})
            continue
        messages.append({"role": "user", "content": content})

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    out: dict[str, Any] = {
        "model": model or str(body.get("model", "")),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if tools:
        out["tools"] = tools
    for key in ("temperature", "top_p", "stop"):
        if key in body:
            out[key] = body[key]
    return out


def openai_completion_from_anthropic(response: dict[str, Any], model: str) -> dict[str, Any]:
    text, tool_calls = _anthropic_assistant_to_openai(response.get("content"))
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"
    return {
        "id": f"chatcmpl-{response.get('id', uuid.uuid4().hex[:12])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": int((response.get("usage") or {}).get("input_tokens") or 0),
            "completion_tokens": int((response.get("usage") or {}).get("output_tokens") or 0),
            "total_tokens": int((response.get("usage") or {}).get("input_tokens") or 0)
            + int((response.get("usage") or {}).get("output_tokens") or 0),
        },
    }


def anthropic_message_from_openai(response: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        raw_args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed = {"_raw": raw_args}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": parsed if isinstance(parsed, dict) else {"value": parsed},
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    stop = "tool_use" if message.get("tool_calls") else "end_turn"
    usage = response.get("usage") or {}
    return {
        "id": str(response.get("id", f"msg_{uuid.uuid4().hex[:12]}")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def openai_completion_from_text(text: str, model: str, stream: bool = False) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def anthropic_message_from_text(text: str, model: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def responses_output_from_text(text: str, model: str) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "output_text": text,
    }


def responses_input_to_messages(body: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Responses body into a chat.completions-shaped dict."""
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    incoming = body.get("input")
    if isinstance(incoming, str):
        messages.append({"role": "user", "content": incoming})
    elif isinstance(incoming, list):
        for item in incoming:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = item.get("content")
            if item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": str(item.get("output", "")),
                    }
                )
                continue
            messages.append({"role": role, "content": content if content is not None else item.get("text", "")})
    chat = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    if "tools" in body:
        chat["tools"] = body["tools"]
    if "temperature" in body:
        chat["temperature"] = body["temperature"]
    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    return chat


def _openai_content_to_anthropic(content: object) -> object:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                header, _, b64 = url.partition(",")
                media = "image/png"
                if "jpeg" in header or "jpg" in header:
                    media = "image/jpeg"
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
                )
            else:
                blocks.append({"type": "text", "text": str(url)})
        elif kind in {"text", "input_text"}:
            blocks.append({"type": "text", "text": str(item.get("text", ""))})
        else:
            if "text" in item:
                blocks.append({"type": "text", "text": str(item["text"])})
    return blocks or ""


def _anthropic_content_to_openai(content: object) -> object:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block.get("source") or {}
            data = source.get("data", "")
            media = source.get("media_type", "image/png")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"},
                }
            )
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text", "")
    return parts


def _anthropic_assistant_to_openai(content: object) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
    return "".join(texts), tool_calls


def _blocks_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts)
