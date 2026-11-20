"""Who thinks in the user's turns: the configured provider, or the user's Claude Code.

The choice is the user's (``provider.set``, the ``5`` of the body) and stays in
``config.json``. Claude Code thinks only in turns; titles, summaries and
reviews stay with the configured provider.
"""

import os
import shutil
from pathlib import Path

from phantasm.config import Config
from phantasm.paths import home_dir
from phantasm.protocol.messages import Thinker
from phantasm.providers.base import LLMProvider
from phantasm.providers.claude_code import ClaudeCodeProvider

FOLDER = "claude-code"
# Apps started from Finder get a short PATH; Claude Code installs into one of these.
INSTALL_FOLDERS = ("~/.local/bin", "~/.claude/local", "/usr/local/bin", "/opt/homebrew/bin")
NAMES = {"google": "Gemini", "openai": "OpenAI", "fake": "Fake"}


def _runnable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_claude(command: str) -> str | None:
    """The full path of the ``claude`` command, or None if it is not installed."""
    if os.sep in command:
        path = Path(command).expanduser()
        return str(path) if _runnable(path) else None
    found = shutil.which(command)
    if found:
        return found
    for folder in INSTALL_FOLDERS:
        path = Path(folder).expanduser() / command
        if _runnable(path):
            return str(path)
    return None


def make_claude(config: Config) -> ClaudeCodeProvider | None:
    """A Claude Code provider for the config's command, or None if it is not installed."""
    command = find_claude(config.claude_code.command)
    if command is None:
        return None
    return ClaudeCodeProvider(home_dir() / FOLDER, command=[command])


def thinker_info(config: Config) -> Thinker:
    """Who thinks in the user's turns, for clients."""
    if config.claude_code.active:
        model = config.claude_code.model
        return Thinker(
            claude_code=True, name=f"Claude {model.capitalize()}" if model.isalpha() else model
        )
    name = config.provider.name
    return Thinker(name=NAMES.get(name, name.capitalize()))


def turn_thinker(config: Config, claude: LLMProvider | None) -> tuple[LLMProvider, str] | None:
    """The provider and model for a turn's steps when Claude Code thinks, else None."""
    if not config.claude_code.active or claude is None:
        return None
    return claude, config.claude_code.model
