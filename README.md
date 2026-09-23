# Enigmatic

Local anonymizing proxy for LLM APIs and coding-agent CLIs.

Clients (OpenAI SDKs, Anthropic SDKs, OpenCode, Cline, Codex CLI) point `base_url` at Enigmatic. Enigmatic runs [Presidio](https://github.com/microsoft/presidio) in-process, replaces PII and secrets with stable placeholders such as `<EMAIL_ADDRESS_1>`, forwards the request, then restores the originals in the streamed response.

It is not encryption. Placeholders keep entity type visible to the model so it can still reason about “this is an email” vs “this is a card number.”

## What talks to what

**Inbound:** OpenAI-compatible HTTP (`/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/completions`, `/v1/models`) and Anthropic Messages (`/v1/messages`).

**Outbound HTTP:** one generic OpenAI-compatible client plus native Anthropic. Named profiles cover OpenAI, Azure, Groq, OpenRouter, Together, Fireworks, DeepSeek, Mistral, Google’s OpenAI-compat endpoint, Ollama, LM Studio, and vLLM. Unknown JSON keys are preserved.

**Outbound agent CLIs:** Enigmatic does not implement a coding agent. It anonymizes the prompt, then calls a CLI that is already installed and logged in, then restores placeholders in the reply. Two transports:

- [ACP](https://agentclientprotocol.com/) over stdio (JSON-RPC). Cursor is `agent acp` with `cursor_login`. Copilot is `copilot --acp --stdio`, with a `copilot -p` fallback. Gemini, Kimi, Hermes, Kiro, Qoder, Trae, QwenPaw, and Grok are further registry rows in `conf/providers.yaml`.
- Prompt mode of the same binaries: `claude -p`, `codex exec --json` (stdin prompt, read-only sandbox), and Copilot's JSON output.

Tool, filesystem, and terminal requests are denied, so a chat completion cannot write your disk. These profiles do not need an upstream API key; they use the CLI's own login. VS Code Copilot and Copilot CLI are **not** Enigmatic clients; do not set `COPILOT_PROVIDER_BASE_URL` to this proxy.

Audio, image generation, files, batches, and fine-tuning return **501**. Embeddings on agent-CLI upstreams also return 501. Vision data URLs are OCR-redacted when Tesseract is installed; otherwise the image is dropped (fail-closed).

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

`api_key` can be any string while the local gate is off. When `ENIGMATIC_API_KEY` is set, pass that token instead. See [Access token](#access-token).

Anthropic: same host, model `anthropic/claude-sonnet-4-5`, path `/v1/messages`.

OpenCode / Cline / Codex: set the provider `base_url` to `http://127.0.0.1:47821/v1`. Model prefixes select a profile (`openai/…`, `groq/…`, `anthropic/…`, `copilot/…`).

Optional session key: `X-Enigmatic-Session` so placeholders stay stable across turns.

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

`api_key_env` in `conf/providers.yaml` selects the variable (default `ENIGMATIC_API_KEY`). A literal `api_key` in that file is used only when the variable is empty. Leave both empty to keep the proxy open on localhost. A missing token returns OpenAI's `invalid_request_error` shape with HTTP 401. `status` reports the gate as enabled or off and never prints the token. `/health` stays unauthenticated.

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

From an OpenAI-compatible client, set the model prefix to the profile: `cursor/default`, `copilot/gpt-5`, `claude/default`, or `codex/gpt-5.4`.

Or call the layer directly. This anonymizes the text, runs the installed CLI, and prints the restored reply:

```bash
uv run enigmatic prompt cursor/default "email me at ada@example.com"
uv run enigmatic prompt codex/gpt-5.4 --file prompt.txt
```

`prompt` refuses HTTP profiles such as `openai/gpt-4o`. Those still go through `enigmatic serve` and the provider API key.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

HTTP tests mock the upstream. Presidio engine tests skip if `en_core_web_sm` is missing. Agent-CLI tests mock subprocess parsers; they do not require Copilot installed.

## Why not Presidio AES encrypt/decrypt?

AES-CBC ciphertext is opaque, has a random IV, and carries no entity type. The model cannot tell a name from a card number and cannot use the value as stable context. Numbered placeholders are the supported AnonymizerEngine extension point for this product.
