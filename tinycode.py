#!/usr/bin/env python3
"""tinycode — minimal AI coding assistant powered by deepagents + OpenRouter."""

import json
import os
import sys
from pathlib import Path

import click
import httpx
import openai
from rich.console import Console
from rich.table import Table
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".config" / "tinycode"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

BASE_SYSTEM_PROMPT = """\
You are a concise, expert coding assistant.
- Prefer working code over explanation
- When showing changes, use unified diffs or full rewrites — never partial snippets without context
- Be direct. Omit pleasantries.
- If you need clarification, ask one focused question
"""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        click.echo(f"warning: malformed config at {CONFIG_FILE}, ignoring", err=True)
        return {}


def save_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Restrict to owner — the file contains an API key.
    os.chmod(CONFIG_FILE, 0o600)


def get_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or load_config().get("openrouter_api_key")


def get_default_model() -> str:
    return (
        os.environ.get("TINYCODE_MODEL")
        or load_config().get("default_model")
        or DEFAULT_MODEL
    )


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return key[:8] + "..." + key[-4:]


def require_api_key() -> str:
    key = get_api_key()
    if not key:
        raise click.ClickException(
            "No API key found.\n"
            "Set OPENROUTER_API_KEY env var or run:\n"
            "  tinycode keys set openrouter sk-or-..."
        )
    return key

# ---------------------------------------------------------------------------
# Context / system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(cwd: Path) -> str:
    for name in ("AGENTS.md", "TINYCODE.md", "CLAUDE.md"):
        ctx = cwd / name
        if ctx.exists():
            return BASE_SYSTEM_PROMPT + f"\n\n## Project Context ({name})\n{ctx.read_text()}"
    return BASE_SYSTEM_PROMPT


def build_user_content(prompt: str, files: tuple[str, ...], stdin_text: str | None) -> str:
    parts: list[str] = []
    if stdin_text:
        parts.append(f"<stdin>\n{stdin_text}\n</stdin>")
    for f in files:
        try:
            content = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise click.ClickException(f"could not read {f}: {e}")
        parts.append(f'<file path="{f}">\n{content}\n</file>')
    parts.append(prompt)
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_model(api_key: str, model_id: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        timeout=120,
        default_headers={
            "HTTP-Referer": "https://github.com/rajgupt/tinycode",
            "X-Title": "tinycode",
        },
    )


def build_agent(model: ChatOpenAI, system_prompt: str, checkpointer=None):
    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        backend=FilesystemBackend(root_dir=Path.cwd(), virtual_mode=True),
    )

# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def stream_to_terminal(agent, messages: list[dict], config: dict | None = None) -> None:
    console = Console()
    kwargs = {"stream_mode": "messages"}
    if config:
        kwargs["config"] = config

    try:
        for chunk, _meta in agent.stream({"messages": messages}, **kwargs):
            if hasattr(chunk, "content") and chunk.content:
                console.print(chunk.content, end="")
    except openai.AuthenticationError:
        raise click.ClickException("Invalid API key. Check it with: tinycode keys get openrouter")
    except openai.PermissionDeniedError as e:
        raise click.ClickException(f"Permission denied: {e}")
    except openai.RateLimitError:
        raise click.ClickException("Rate limited — wait a moment and try again.")
    except openai.APIError as e:
        raise click.ClickException(f"API error: {e}")
    except httpx.HTTPError as e:
        raise click.ClickException(f"Network error: {e}")
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        return

    console.print()  # trailing newline


def run_single_shot(
    prompt: str,
    files: tuple[str, ...],
    model_id: str | None,
    system_override: str | None,
) -> None:
    api_key = require_api_key()
    model = build_model(api_key, model_id or get_default_model())
    system = system_override or build_system_prompt(Path.cwd())
    agent = build_agent(model, system)

    stdin_text = None
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()

    user_content = build_user_content(prompt, files, stdin_text)
    stream_to_terminal(agent, [{"role": "user", "content": user_content}])


def run_chat_loop(model_id: str | None, system_override: str | None) -> None:
    api_key = require_api_key()
    resolved_model = model_id or get_default_model()
    model = build_model(api_key, resolved_model)
    system = system_override or build_system_prompt(Path.cwd())
    checkpointer = MemorySaver()
    agent = build_agent(model, system, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"tinycode-{os.getpid()}"}}

    console = Console()
    console.print(f"[dim]tinycode chat | {resolved_model} | Ctrl-C or /exit to quit[/dim]\n")

    while True:
        try:
            user_input = click.prompt("you", prompt_suffix="> ")
        except (click.Abort, EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        cmd = user_input.strip().lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]bye[/dim]")
            break
        if not user_input.strip():
            continue

        console.print()
        stream_to_terminal(agent, [{"role": "user", "content": user_input}], config=config)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _CLI(click.Group):
    """Group that falls back to single-shot mode when the first arg isn't a known subcommand."""

    def _value_taking_opts(self) -> set[str]:
        # Introspect own options so this stays in sync if new ones are added.
        opts: set[str] = set()
        for p in self.params:
            if isinstance(p, click.Option) and not p.is_flag and not p.count:
                opts.update(p.opts)
                opts.update(p.secondary_opts)
        return opts

    def parse_args(self, ctx, args):
        # Split into flag tokens and positional tokens. If the first positional
        # isn't a known subcommand, treat all positionals as the prompt and only
        # let Click parse the flags.
        value_opts = self._value_taking_opts()
        flags, positionals = [], []
        i = 0
        while i < len(args):
            a = args[i]
            if a.startswith("-"):
                flags.append(a)
                # --opt=value is a single token already; only consume the next
                # token when the option is bare and takes a value.
                if "=" not in a and a in value_opts and i + 1 < len(args):
                    i += 1
                    flags.append(args[i])
            else:
                positionals.append(a)
            i += 1

        if positionals and positionals[0] not in self.commands:
            ctx.ensure_object(dict)
            ctx.obj["_prompt"] = " ".join(positionals)
            return super().parse_args(ctx, flags)
        return super().parse_args(ctx, args)


@click.group(
    cls=_CLI,
    invoke_without_command=True,
)
@click.option("-f", "--file", "files", multiple=True, type=click.Path(exists=True), help="Attach a file")
@click.option("-m", "--model", default=None, help="Model ID, e.g. anthropic/claude-opus-4")
@click.option("-s", "--system", default=None, help="Override system prompt")
@click.pass_context
def cli(ctx, files, model, system):
    """tinycode — minimal AI coding assistant.

    \b
    Pass a prompt directly, or use a subcommand:
      tinycode "explain this codebase"
      cat main.py | tinycode "what does this do?"
      tinycode -f src/app.py "add error handling"
      tinycode chat
    """
    if ctx.invoked_subcommand is not None:
        return
    prompt = (ctx.obj or {}).get("_prompt")
    if prompt is not None:
        run_single_shot(prompt, files, model, system)
    elif not sys.stdin.isatty():
        run_single_shot("", files, model, system)
    else:
        click.echo(ctx.get_help())


@cli.command()
@click.option("-m", "--model", default=None, help="Model ID")
@click.option("-s", "--system", default=None, help="Override system prompt")
def chat(model, system):
    """Start an interactive chat session."""
    run_chat_loop(model, system)


@cli.group()
def keys():
    """Manage API keys."""


@keys.command("set")
@click.argument("service")
@click.argument("key")
def keys_set(service, key):
    """Store an API key. SERVICE must be 'openrouter'."""
    if service != "openrouter":
        raise click.UsageError(f"Unknown service '{service}'. Only 'openrouter' is supported.")
    config = load_config()
    config["openrouter_api_key"] = key
    save_config(config)
    click.echo(f"Saved {service} key to {CONFIG_FILE}")


@keys.command("get")
@click.argument("service")
def keys_get(service):
    """Show a stored API key (masked)."""
    if service != "openrouter":
        raise click.UsageError(f"Unknown service '{service}'.")
    env_key = os.environ.get("OPENROUTER_API_KEY")
    stored_key = load_config().get("openrouter_api_key")
    if env_key:
        click.echo(f"{mask_key(env_key)} (from $OPENROUTER_API_KEY)")
    elif stored_key:
        click.echo(mask_key(stored_key))
    else:
        click.echo("No key set. Run: tinycode keys set openrouter sk-or-...")


@cli.command()
@click.option("--search", default=None, help="Filter models by name")
def models(search):
    """List available OpenRouter models."""
    api_key = require_api_key()
    try:
        resp = httpx.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise click.ClickException(f"API error {e.response.status_code}: {e.response.text[:200]}")

    data = resp.json()["data"]
    table = Table("Model ID", "Context", "$/1M in", "$/1M out")
    for m in sorted(data, key=lambda x: x["id"]):
        if search and search.lower() not in m["id"].lower():
            continue
        p = m.get("pricing", {})
        table.add_row(
            m["id"],
            str(m.get("context_length", "?")),
            p.get("prompt", "?"),
            p.get("completion", "?"),
        )
    Console().print(table)


if __name__ == "__main__":
    cli()
