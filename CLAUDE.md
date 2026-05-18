# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .
# or
uv pip install -e .

# Run
tinycode "prompt"
tinycode chat
tinycode models

# Set API key
tinycode keys set openrouter sk-or-...
# or
export OPENROUTER_API_KEY=sk-or-...
```

There are no tests or linting configured yet.

## Architecture

The entire application lives in a single file: `tinycode.py`.

**Entry point**: `cli` — a custom `click.Group` subclass (`_CLI`) that overrides `parse_args` to support bare prompts (e.g. `tinycode "explain this"`) by detecting when the first positional arg is not a known subcommand and routing it to `run_single_shot`.

**Agent layer**: Built on `deepagents.create_deep_agent()` (wraps LangGraph), which provides built-in file-system tools (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`) and task delegation. The LLM backend is `ChatOpenAI` pointed at OpenRouter's API endpoint.

**Two execution modes**:
- Single-shot (`run_single_shot`): stateless, no checkpointer
- Chat (`run_chat_loop`): uses `MemorySaver` as LangGraph checkpointer, keyed by PID for per-session memory

**System prompt**: `build_system_prompt()` auto-loads `AGENTS.md`, `TINYCODE.md`, or `CLAUDE.md` from the CWD and appends it to the base prompt — so tinycode is self-context-aware when run from this repo.

**Config**: stored at `~/.config/tinycode/config.json` (chmod 600). API key resolution order: `$OPENROUTER_API_KEY` env var → stored config. Model resolution: `$TINYCODE_MODEL` env var → stored `default_model` → hardcoded `google/gemini-2.5-flash`.

**Dependencies**: `deepagents` (agent framework), `langchain-openai` + `langgraph` (LLM + graph runtime), `click` (CLI), `rich` (terminal output), `httpx` (models listing API call).
