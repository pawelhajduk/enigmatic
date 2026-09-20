# Enigmatic

Local anonymizing proxy for LLM APIs and coding-agent CLIs.

Clients (OpenAI SDKs, Anthropic SDKs, OpenCode, Cline, Codex CLI) point `base_url` at Enigmatic. Enigmatic runs [Presidio](https://github.com/microsoft/presidio) in-process, replaces PII and secrets with stable placeholders such as `<EMAIL_ADDRESS_1>`, forwards the request, then restores the originals in the streamed response.

It is not encryption. Placeholders keep entity type visible to the model so it can still reason about “this is an email” vs “this is a card number.”

## What talks to what

**Inbound:** OpenAI-compatible HTTP (`/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/completions`, `/v1/models`) and Anthropic Messages (`/v1/messages`).

**Outbound HTTP:** one generic OpenAI-compatible client plus native Anthropic. Named profiles cover OpenAI, Azure, Groq, OpenRouter, Together, Fireworks, DeepSeek, Mistral, Google’s OpenAI-compat endpoint, Ollama, LM Studio, and vLLM. Unknown JSON keys are preserved.

**Outbound agent CLIs:** a shared [ACP](https://agentclientprotocol.com/) client (JSON-RPC over stdio) plus prompt-mode JSONL. Copilot uses `copilot --acp --stdio` with a `copilot -p --output-format json` fallback. Other ACP binaries (Gemini, Kimi, Hermes, Kiro, Qoder, Trae, QwenPaw, Grok) are registry rows in `conf/providers.yaml`. Enigmatic denies CLI tools so a chat completion cannot write your disk. VS Code Copilot and Copilot CLI are **not** Enigmatic clients; do not set `COPILOT_PROVIDER_BASE_URL` to this proxy.

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

Copy `conf/providers.yaml` and set the matching `*_API_KEY` environment variables. You can point `analyzer_conf` at an existing Presidio recognizer YAML.

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

Anthropic: same host, model `anthropic/claude-sonnet-4-5`, path `/v1/messages`.

OpenCode / Cline / Codex: set the provider `base_url` to `http://127.0.0.1:47821/v1`. Model prefixes select a profile (`openai/…`, `groq/…`, `anthropic/…`, `copilot/…`).

Optional local gate: set `api_key` in `conf/providers.yaml` and send `Authorization: Bearer <key>`. Optional session key: `X-Enigmatic-Session` so placeholders stay stable across turns.

## Copilot CLI

Install and log in yourself:

```bash
npm i -g @github/copilot
copilot   # complete login in the CLI
```

Then call Enigmatic with `model: "copilot/gpt-5"` (or another model the CLI accepts). Enigmatic packs the anonymized transcript into ACP or `copilot -p`. There is no `enigmatic login copilot`.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

HTTP tests mock the upstream. Presidio engine tests skip if `en_core_web_sm` is missing. Agent-CLI tests mock subprocess parsers; they do not require Copilot installed.

## Why not Presidio AES encrypt/decrypt?

AES-CBC ciphertext is opaque, has a random IV, and carries no entity type. The model cannot tell a name from a card number and cannot use the value as stable context. Numbered placeholders are the supported AnonymizerEngine extension point for this product.
