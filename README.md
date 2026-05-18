# tinycode

A minimal AI coding assistant in a single Python file, powered by [deepagents](https://docs.langchain.com/oss/python/deepagents/overview) and [OpenRouter](https://openrouter.ai).

## Install

```bash
pip install -e .
# or
uv pip install -e .
```

## Setup

```bash
tinycode keys set openrouter sk-or-...
# or
export OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
# Single shot
tinycode "explain this codebase"
cat main.py | tinycode "what does this do?"
tinycode -f src/app.py "add error handling"

# Pick a model
tinycode -m anthropic/claude-opus-4 "refactor this"

# Interactive chat
tinycode chat
tinycode chat -m google/gemini-2.5-flash

# Browse models
tinycode models
tinycode models --search claude

# Manage keys
tinycode keys set openrouter sk-or-...
tinycode keys get openrouter
```

## Project context

Drop an `AGENTS.md` (or `TINYCODE.md` / `CLAUDE.md`) in your project root and tinycode will automatically include it in every prompt — useful for repo-specific rules, conventions, or context.

## What deepagents provides

tinycode uses `create_deep_agent()` which gives the agent built-in tools:

- `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` — file system
- `write_todos` — task planning
- `task` — delegate to sub-agents with isolated context

## Default model

`google/gemini-2.5-flash` — fast, large context, cheap. Override with `-m` or set `default_model` in `~/.config/tinycode/config.json`.
