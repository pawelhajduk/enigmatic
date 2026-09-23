"""OpenAI Chat Completions / Responses <-> Anthropic Messages field mapping."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

DEFAULT_ANTHROPIC_MAX_TOKENS = 8192

_ANTHROPIC_TO_OPENAI_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def strip_model_prefix(model: str) -> tuple[str, str]:
    """Return (profile_hint, bare_model). `openai/gpt-4o` -> (`openai`, `gpt-4o`)."""
    if "/" in model:
        prefix, rest = model.split("/", 1)
        return prefix, rest
    return "", model


def openai_finish_reason(stop_reason: str | None) -> str:
    return _ANTHROPIC_TO_OPENAI_FINISH.get(stop_reason or "", "stop")


def anthropic_stop_reason(finish_reason: str | None) -> str:
    return _OPENAI_TO_ANTHROPIC_STOP.get(finish_reason or "", "end_turn")


def openai_usage_from_anthropic(usage: dict[str, Any]) -> dict[str, Any]:
    cached = int(usage.get("cache_read_input_tokens") or 0)
    created = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + cached + created
    completion = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def responses_usage_from_chat(usage: dict[str, Any]) -> dict[str, Any]:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": prompt + completion,
    }


def _parse_arguments(raw_args: object) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        parsed = {"_raw": raw_args}
    if parsed is None:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _anthropic_tool_choice(choice: object, parallel: object) -> dict[str, Any] | None:
    out: dict[str, Any] | None = None
    if choice == "auto":
        out = {"type": "auto"}
    elif choice == "none":
        out = {"type": "none"}
    elif choice == "required":
        out = {"type": "any"}
    elif isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        if isinstance(fn, dict) and fn.get("name"):
            out = {"type": "tool", "name": str(fn["name"])}
    if parallel is False:
        out = out or {"type": "auto"}
        if out["type"] != "none":
            out["disable_parallel_tool_use"] = True
    return out


def _openai_tool_choice(choice: object) -> object:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "any":
        return "required"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": str(choice["name"])}}
    return None


def openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    _, model = strip_model_prefix(str(body.get("model", "")))
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            text = content if isinstance(content, str) else _blocks_to_text(content)
            if text:
                system_parts.append(text)
            continue
        if role in {"tool", "function"}:
            tool_content = _openai_content_to_anthropic(content)
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(message.get("tool_call_id") or message.get("name") or ""),
                        "content": tool_content if tool_content else "",
                    }
                ],
            )
            continue
        if role == "assistant":
            blocks = _as_blocks(_openai_content_to_anthropic(content))
            calls = list(message.get("tool_calls") or [])
            if isinstance(message.get("function_call"), dict):
                calls.append({"id": message["function_call"].get("name"), "function": message["function_call"]})
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": str(fn.get("name", "")),
                        "input": _parse_arguments(fn.get("arguments", "{}")),
                    }
                )
            append("assistant", [block for block in blocks if block.get("type") != "text" or block.get("text")])
            continue
        append("user", _as_blocks(_openai_content_to_anthropic(content)))

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" and "function" in tool else tool
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    out: dict[str, Any] = {
        "model": model or str(body.get("model", "")),
        "messages": messages,
        "max_tokens": int(max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS),
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    if tools:
        out["tools"] = tools
        tool_choice = _anthropic_tool_choice(body.get("tool_choice"), body.get("parallel_tool_calls"))
        if tool_choice is not None:
            out["tool_choice"] = tool_choice
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    if body.get("stop") is not None:
        stop = body["stop"]
        out["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    user = body.get("user") or body.get("safety_identifier")
    if isinstance(user, str) and user:
        out["metadata"] = {"user_id": user}
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
            item: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
            continue
        if isinstance(content, list):
            tool_results = [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]
            others = [block for block in content if not (isinstance(block, dict) and block.get("type") == "tool_result")]
            for block in tool_results:
                result = block.get("content", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result if isinstance(result, str) else _anthropic_content_to_openai(result),
                    }
                )
            if others:
                messages.append({"role": "user", "content": _anthropic_content_to_openai(others)})
            continue
        messages.append({"role": "user", "content": content})

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
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
    stream = bool(body.get("stream", False))
    out: dict[str, Any] = {
        "model": model or str(body.get("model", "")),
        "messages": messages,
        "stream": stream,
    }
    if stream:
        out["stream_options"] = {"include_usage": True}
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if tools:
        out["tools"] = tools
        tool_choice = body.get("tool_choice")
        mapped = _openai_tool_choice(tool_choice)
        if mapped is not None:
            out["tool_choice"] = mapped
        if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    for key in ("temperature", "top_p"):
        if key in body:
            out[key] = body[key]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
        out["user"] = metadata["user_id"]
    return out


def openai_completion_from_anthropic(response: dict[str, Any], model: str) -> dict[str, Any]:
    text, tool_calls = _anthropic_assistant_to_openai(response.get("content"))
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = openai_finish_reason(response.get("stop_reason"))
    if tool_calls and finish == "stop":
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{response.get('id', uuid.uuid4().hex[:12])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": openai_usage_from_anthropic(response.get("usage") or {}),
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
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": _parse_arguments(fn.get("arguments", "{}")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    stop = anthropic_stop_reason(choice.get("finish_reason"))
    if message.get("tool_calls"):
        stop = "tool_use"
    usage = response.get("usage") or {}
    return {
        "id": str(response.get("id", f"msg_{uuid.uuid4().hex[:12]}")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
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
                "logprobs": None,
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
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def responses_output_from_text(text: str, model: str) -> dict[str, Any]:
    return responses_from_chat_completion(openai_completion_from_text(text, model), model)


def responses_from_chat_completion(
    completion: dict[str, Any],
    model: str,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Responses API object from a Chat Completion (for non-OpenAI upstreams)."""
    request = request or {}
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    custom = custom_tool_names(request)
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
        arguments = str(fn.get("arguments") or "{}")
        if name in custom:
            output.append(
                {
                    "id": f"ctc_{uuid.uuid4().hex}",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": custom_tool_input(arguments),
                }
            )
            continue
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    incomplete = choice.get("finish_reason") == "length"
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "error": None,
        "model": model,
        "output": output,
        "output_text": text if isinstance(text, str) else "",
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools") or [],
        "usage": responses_usage_from_chat(completion.get("usage") or {}),
    }


def _responses_content_to_chat(content: object) -> object:
    if not isinstance(content, list):
        return content if content is not None else ""
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text", "summary_text", "refusal"}:
            parts.append({"type": "text", "text": str(part.get("text") or part.get("refusal") or "")})
        elif kind == "input_image":
            url = part.get("image_url")
            if isinstance(url, str) and url:
                image: dict[str, Any] = {"url": url}
                if part.get("detail"):
                    image["detail"] = part["detail"]
                parts.append({"type": "image_url", "image_url": image})
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(str(part.get("text", "")) for part in parts)
    return parts


def custom_tool_names(request: dict[str, Any] | None) -> set[str]:
    """Names of Responses `custom` (free-form input) tools in a request."""
    tools = (request or {}).get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        str(tool["name"])
        for tool in tools
        if isinstance(tool, dict) and tool.get("type") == "custom" and tool.get("name")
    }


def custom_tool_input(arguments: str) -> str:
    parsed = _parse_arguments(arguments)
    value = parsed.get("input", parsed.get("_raw", ""))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _responses_tools_to_chat(tools: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "custom" and tool.get("name"):
            description = str(tool.get("description") or "")
            fmt = tool.get("format")
            if isinstance(fmt, dict) and fmt.get("type") == "grammar" and fmt.get("definition"):
                description += f"\n\nThe input must match this {fmt.get('syntax', '')} grammar:\n{fmt['definition']}"
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": description.strip(),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string", "description": "Raw tool input"}},
                            "required": ["input"],
                        },
                    },
                }
            )
        elif isinstance(tool.get("function"), dict):
            out.append(tool)
        elif tool.get("type") == "function" and tool.get("name"):
            fn: dict[str, Any] = {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            if tool.get("strict") is not None:
                fn["strict"] = tool["strict"]
            out.append({"type": "function", "function": fn})
    return out


def _responses_tool_choice_to_chat(choice: object) -> object:
    if isinstance(choice, dict) and choice.get("type") == "function" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    if choice in {"auto", "none", "required"}:
        return choice
    return None


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
            kind = item.get("type")
            if kind in {"function_call", "custom_tool_call"}:
                arguments = item.get("arguments")
                if kind == "custom_tool_call":
                    arguments = json.dumps({"input": item.get("input", "")})
                call = {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {"name": str(item.get("name") or ""), "arguments": str(arguments or "{}")},
                }
                last = messages[-1] if messages else None
                if last is not None and last.get("role") == "assistant":
                    last.setdefault("tool_calls", []).append(call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                continue
            if kind in {"function_call_output", "custom_tool_call_output"}:
                output = item.get("output", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item.get("call_id", "")),
                        "content": output if isinstance(output, str) else _responses_content_to_chat(output),
                    }
                )
                continue
            if kind in {"reasoning", "compaction", "item_reference"}:
                continue
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            content = item.get("content")
            messages.append(
                {
                    "role": role,
                    "content": _responses_content_to_chat(content if content is not None else item.get("text", "")),
                }
            )
    chat: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    tools = _responses_tools_to_chat(body.get("tools"))
    if tools:
        chat["tools"] = tools
        choice = _responses_tool_choice_to_chat(body.get("tool_choice"))
        if choice is not None:
            chat["tool_choice"] = choice
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    for key in ("temperature", "top_p", "user", "safety_identifier"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    return chat


def _as_blocks(content: object) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return content
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


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
        if kind in {"image_url", "input_image"}:
            raw = item.get("image_url")
            url = raw.get("url", "") if isinstance(raw, dict) else raw
            if isinstance(url, str) and url.startswith("data:"):
                header, _, b64 = url.partition(",")
                media = header[len("data:") :].split(";", 1)[0] or "image/png"
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
            elif isinstance(url, str) and url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        elif kind in {"text", "input_text", "output_text"}:
            blocks.append({"type": "text", "text": str(item.get("text", ""))})
        elif kind == "refusal":
            blocks.append({"type": "text", "text": str(item.get("refusal", ""))})
        elif "text" in item:
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
            if source.get("type") == "url":
                parts.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
                continue
            data = source.get("data", "")
            media = source.get("media_type", "image/png")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(str(part.get("text", "")) for part in parts)
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
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
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
