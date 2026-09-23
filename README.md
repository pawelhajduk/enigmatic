# Enigmatic

Local anonymizing proxy for LLM APIs and coding-agent CLIs.

Clients (OpenAI SDKs, Anthropic SDKs, OpenCode, Cline, Codex CLI) point `base_url` at Enigmatic. Enigmatic runs [Presidio](https://github.com/microsoft/presidio) in-process, replaces PII and secrets with stable placeholders such as `<EMAIL_ADDRESS_1>`, forwards the request, then restores the originals in the streamed response.

It is not encryption. Placeholders keep entity type visible to the model so it can still reason about “this is an email” vs “this is a card number.”

## What talks to what

**Inbound:** OpenAI-compatible HTTP and Anthropic Messages (`/v1/messages`). Every route is also served without the `/v1` prefix for clients whose base URL omits it.

| OpenAI endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings` | Streaming, tool calls, `stream_options.include_usage` |
| `POST /v1/responses` | Streaming, function and custom tools (Codex `apply_patch`), reasoning items, `previous_response_id`, background mode |
| `POST /v1/responses/compact`, `POST /v1/responses/input_tokens` | OpenAI-compatible upstreams only |
| `GET`/`DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, `GET /v1/responses/{id}/input_items` | Routed to the profile that created the response |
| `GET /v1/models`, `GET /v1/models/{model}` | |

Responses sent to an Anthropic upstream are translated through Chat Completions and streamed back as the full Responses event sequence. That path cannot use `previous_response_id`; clients must resend the full input (Codex does this with `store: false`).

Encrypted reasoning and compaction content, call ids, tool names, enums, custom tool grammars, and non-image data URLs (`file_data`, audio) are forwarded untouched. Tool-call arguments are scanned as decoded JSON values, and restored originals are JSON-escaped so arguments always parse.

Client headers `OpenAI-Beta`, `OpenAI-Organization`, `OpenAI-Project`, `Idempotency-Key`, `X-Client-Request-Id`, `session_id`, `conversation_id`, `originator`, and `anthropic-beta` are forwarded. Upstream `retry-after`, `x-request-id`, and rate-limit headers are returned to the client. Errors use OpenAI's `{"error": {"message", "type", "param", "code"}}` shape (Anthropic's shape on `/messages`), and a stream that fails mid-way ends with an SSE error event.

**Outbound HTTP:** one generic OpenAI-compatible client plus native Anthropic. Named profiles cover OpenAI, Azure, Groq, OpenRouter, Together, Fireworks, DeepSeek, Mistral, Google’s OpenAI-compat endpoint, Ollama, LM Studio, and vLLM. Unknown JSON keys are preserved.

**Outbound agent CLIs:** Enigmatic does not implement a coding agent. It anonymizes the prompt, then calls a CLI that is already installed and logged in, then restores placeholders in the reply. Two transports:

- [ACP](https://agentclientprotocol.com/) over stdio (JSON-RPC). Cursor is `agent --mode ask --sandbox enabled acp` with `cursor_login`. `agent` rejects unknown flags, so the profile sets `deny_flags: []` and relies on Cursor's read-only ask mode instead of `--deny-tool`. Copilot is `copilot --acp --stdio`, with a `copilot -p` fallback. Gemini, Kimi, Hermes, Kiro, Qoder, Trae, QwenPaw, and Grok are further registry rows in `conf/providers.yaml`.
- Prompt mode of the same binaries: `claude -p`, `codex exec --json` (stdin prompt, read-only sandbox), and Copilot's JSON output.

Tool, filesystem, and terminal requests are denied, so a chat completion cannot write your disk. These profiles do not need an upstream API key; they use the CLI's own login. VS Code Copilot and Copilot CLI are **not** Enigmatic clients; do not set `COPILOT_PROVIDER_BASE_URL` to this proxy.

Audio, image generation, files, batches, fine-tuning, and conversations return **501**. Embeddings on agent-CLI upstreams also return 501. Vision data URLs are OCR-redacted when Tesseract is installed. Otherwise the image part is replaced with a text part saying it was omitted (fail-closed). Remote image URLs are forwarded as-is.

## Install

Python 3.11+. [uv](https://docs.astral.sh/uv/) is the intended installer.

```bash
uv sync
uv run python -m spacy download en_core_web_sm
# optional, for vision:
# sudo apt install tesseract-ocr
```

Or: `pip install -e .` then the same spaCy download.

Copy `.env.example` to `.env` and set the upstream `*_API_KEY` variables. Set `ENIGMATIC_API_KEY` in that file when clients should authenticate to the proxy. Copy `conf/providers.yaml` if you need to change profiles. You can point `analyzer_conf` at an existing Presidio recognizer YAML.

## Run

```bash
uv run enigmatic serve
```

Listens on `127.0.0.1:47821` by default. Override with `--host` / `--port` / `--config`. On start, the proxy prints listener, pipeline, HTTP profiles, and agent CLIs to the terminal (same dump as `status`). There is no web status page; `/health` remains a JSON liveness check for scripts.

```bash
uv run enigmatic doctor
uv run enigmatic status
```

`status` and `doctor` print the same diagnostics. `doctor` exits 1 if spaCy is missing. Neither command prints prompt text or API keys.

Preview what the proxy would send upstream without calling an LLM:

```bash
uv run enigmatic dry-run "email me at ada@example.com"
uv run enigmatic dry-run --file prompt.txt
uv run enigmatic dry-run --json '{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi ada@example.com"}]}'
```

The command prints the original text, the anonymized text, and the placeholder map.

## Point a client at Enigmatic

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:47821/v1", api_key="not-used")
print(client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Reply with a short greeting."}],
).choices[0].message.content)
```

`api_key` can be any string while the local gate is off. When `ENIGMATIC_API_KEY` is set, pass that token instead. See [Access token](#access-token). Listening on anything other than loopback requires that token; `enigmatic serve --host 0.0.0.0` exits if it is unset.

Anthropic: same host, model `anthropic/claude-sonnet-4-5`, path `/v1/messages`.

OpenCode / Cline / Codex: set the provider `base_url` to `http://127.0.0.1:47821/v1`. Model prefixes select a profile (`openai/…`, `groq/…`, `anthropic/…`, `copilot/…`).

Optional session key: `X-Enigmatic-Session` so placeholders stay stable across turns. The header is a capability for that vault: with the gate on it is namespaced by the bearer token, and with the gate off a missing header does not share a global map. When that header is absent, a client `session_id` / `conversation_id` header or the request's `prompt_cache_key` serves the same role. Every `/v1/responses` call gets a stored vault, and a later `previous_response_id` that points at it reuses it, so placeholders in upstream-stored history still restore. Sessions expire after an hour and the process keeps a bounded number of them.

## Access token

Inbound auth matches the OpenAI API. Clients send `Authorization: Bearer <token>` (the OpenAI SDK does this from `api_key`). HTTP Basic is also accepted, with a blank username and the token as the password.

Enigmatic reads that token from the environment. On startup it loads `.env` and `.env.local` from the working directory, from the directory that contains the config file, and from the repo root when the config lives in `conf/`. Existing process variables win. `.env.local` wins over `.env`. Values are not shell-expanded.

```bash
# .env
ENIGMATIC_API_KEY=sk-local-change-me
OPENAI_API_KEY=sk-...
```

```python
client = OpenAI(base_url="http://127.0.0.1:47821/v1", api_key="sk-local-change-me")
```

`api_key_env` in `conf/providers.yaml` selects the variable (default `ENIGMATIC_API_KEY`). A literal `api_key` in that file is used only when the variable is empty. Leave both empty to keep the proxy open on localhost; `serve` prints a warning in that case, and refuses a non-loopback bind. A missing token returns OpenAI's `invalid_request_error` shape with HTTP 401. `status` reports the gate as enabled or off and never prints the token. `/health` stays unauthenticated.

`.env` files do not apply `HTTP(S)_PROXY`, `ALL_PROXY`, or CA bundle variables. Set those in the process environment if you need them.

The same env files supply upstream keys (`OPENAI_API_KEY` and the other `api_key_env` names).

## Installed coding agents

Log in with the agent's own CLI. There is no `enigmatic login`.

```bash
# Cursor
agent login
# Copilot
npm i -g @github/copilot && copilot
# Claude Code and Codex use their own login as well
claude
codex login
```

From an OpenAI-compatible client, set the model prefix to the profile: `cursor/default`, `copilot/gpt-5`, `claude/default`, or `codex/gpt-5.4`. Enigmatic packs the anonymized transcript into ACP or the CLI's prompt mode. The prompt is written to the CLI's stdin, not its argv. Agent profiles that cannot deny tools are rejected. ACP sessions and prompt-mode CLIs start in an empty temp directory, and at most two agent CLIs run at once.

Codex's `--sandbox read-only` blocks writes but still lets the model run read-only shell commands. A read-only shell could read local files and send them upstream without anonymization, so the Codex profile only counts as tool-denying when `shell_tool` and `unified_exec` are also disabled. The bundled profile also disables image viewing, browser, computer-use, apps, plugins, image generation, and sub-agents. It passes `--ignore-user-config`, so MCP servers from `~/.codex/config.toml` are not loaded; `codex login` credentials still apply.

Or call the layer directly. This anonymizes the text, runs the installed CLI, and prints the restored reply:

```bash
uv run enigmatic prompt cursor/default "email me at ada@example.com"
uv run enigmatic prompt codex/gpt-5.4 --file prompt.txt
```

`prompt` refuses HTTP profiles such as `openai/gpt-4o`. Those still go through `enigmatic serve` and the provider API key.

Request bodies are capped at 32 MiB (`max_body_bytes` in `conf/providers.yaml`). Upstream reads time out after 600 s without data, matching the OpenAI SDK; set `read_timeout` on an HTTP profile to change it. Vision data URLs over 5 MiB are dropped. Presidio runs off the request event loop, and repeated strings in a session are analyzed once. Names and locations stay off unless you add `PERSON` or `LOCATION` to `enabled_entities`.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

HTTP tests mock the upstream. Presidio engine tests skip if `en_core_web_sm` is missing. Agent-CLI tests mock subprocess parsers; they do not require Copilot installed.

## Why not Presidio AES encrypt/decrypt?

AES-CBC ciphertext is opaque, has a random IV, and carries no entity type. The model cannot tell a name from a card number and cannot use the value as stable context. Numbered placeholders are the supported AnonymizerEngine extension point for this product.
