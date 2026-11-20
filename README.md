# Phantasm

Phantasm is a local agent runtime for a desktop AI companion on macOS. It runs as a background
service on the loopback interface; client apps connect to it over WebSocket. Phantasm runs the
agent loop and owns everything that matters: sessions, memory, tools, permissions and approvals.
Clients only render what it streams and carry out the actions they declare.

## Features

- **Agent loop with streaming** — text, tool calls and approval requests reach every connected
  client as they happen; a running turn can be cancelled or steered.
- **Tools with approvals** — files, shell, code search, web pages, GitHub and the user's Chrome,
  each behind an allow / ask / deny policy and a path sandbox.
- **Client actions** — a client declares what it can do (take a screenshot, open an app) and the
  model calls it like any other tool.
- **Memory** — long-term and per-conversation memory, a journal across conversations, summaries
  and full-text search over past sessions.
- **Long-running work** — background agents with optional review, scheduled tasks and reminders,
  recovery of turns interrupted by a crash.
- **Extensible** — skills, MCP servers, Python plugins and tools loaded on demand.
- **Any model** — OpenAI-compatible Chat Completions APIs (presets for Google AI Studio,
  OpenRouter, Mistral and Ollama) or a local Claude Code CLI; a fake provider for development.

## Architecture

```
desktop app ──┐                         ┌── model provider (HTTPS)
              ├── WebSocket ── Phantasm ┼── tools: files, shell, web, GitHub
Chrome ext. ──┘   127.0.0.1             └── SQLite: sessions, memory, schedules
```

Python 3.12, FastAPI + uvicorn, pydantic v2, SQLite via aiosqlite, httpx. Fully asynchronous;
blocking work runs in threads or subprocesses.

## Security

- Listens on `127.0.0.1` only; every connection needs a local token, browser origins are refused.
- File tools work inside allowed folders; `deny` rules can never be bypassed, not even by an
  "approve everything" mode.
- API keys live only in a git-ignored `.env` file: never in the config, the database or the logs,
  and they are masked in tool output.

## Getting started

Requirements: macOS and [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 itself).

```bash
git clone https://github.com/mvrhaa/phantasm && cd phantasm
uv sync
uv run phantasm doctor   # check folders, config, database, token and provider
uv run phantasm serve    # start the service on 127.0.0.1:47400
```

Without configuration Phantasm runs on the fake provider, which echoes messages back. To use a
real model, set the provider in `config.json` (`uv run phantasm config path` shows where it is):

```json
"provider": { "name": "google", "model": "gemini-3.5-flash-lite" }
```

and put the key in `.env` at the project root, e.g. `GEMINI_API_KEY=…`
(`OPENROUTER_API_KEY` and `MISTRAL_API_KEY` for the other presets; Ollama needs none).

## Command line

| Command | What it does |
|---|---|
| `phantasm serve` | start the service |
| `phantasm token` | print the token clients connect with |
| `phantasm doctor` | check the installation without changing anything |
| `phantasm usage` | model requests and tokens used today and in the last 30 days |
| `phantasm config path` | show where the config file is |
| `phantasm eval` | run the evaluation tasks against the configured model |

Run them with `uv run`. To start Phantasm at login, install a LaunchAgent with
`uv run python scripts/install_launchagent.py`.

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/gen_protocol_docs.py   # regenerate docs/protocol.md from the models
```

Tests never touch the network or the real data folder.

## Documentation

- [docs/protocol.md](docs/protocol.md) — WebSocket protocol reference for client authors.
- [extension/README.md](extension/README.md) — installing the Chrome extension.
- [PHANTASM_SPEC.md](PHANTASM_SPEC.md) — full design specification (in Russian).
