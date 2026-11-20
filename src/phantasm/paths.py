"""Filesystem locations used by Phantasm on macOS.

Every function reads the environment on each call, so tests can redirect
all paths by setting ``PHANTASM_HOME``.
"""

import os
from pathlib import Path

from phantasm import APP_NAME

HOME_ENV = "PHANTASM_HOME"


def _home_override() -> Path | None:
    """Return the directory from ``PHANTASM_HOME``, or None if it is unset or empty."""
    value = os.environ.get(HOME_ENV)
    if not value:
        return None
    return Path(value).expanduser().absolute()


def home_dir() -> Path:
    """Return ``PHANTASM_HOME`` or ``~/Library/Application Support/Phantasm``."""
    return _home_override() or Path.home() / "Library" / "Application Support" / APP_NAME


def config_path() -> Path:
    """Return the path of ``config.json``."""
    return home_dir() / "config.json"


def db_path() -> Path:
    """Return the path of the SQLite database."""
    return home_dir() / "phantasm.db"


def token_path() -> Path:
    """Return the path of the local access token file."""
    return home_dir() / "token"


def files_dir() -> Path:
    """Return the directory for attachments and files sent to the user."""
    return home_dir() / "files"


def outputs_dir() -> Path:
    """Return the directory for tool outputs too long to show the model whole."""
    return files_dir() / "outputs"


def logs_dir() -> Path:
    """Return the log directory: ``PHANTASM_HOME/logs`` or ``~/Library/Logs/Phantasm``."""
    override = _home_override()
    if override is not None:
        return override / "logs"
    return Path.home() / "Library" / "Logs" / APP_NAME


def ensure_dirs() -> None:
    """Create the data, files and log directories if they do not exist."""
    for directory in (home_dir(), files_dir(), logs_dir()):
        directory.mkdir(parents=True, exist_ok=True)
