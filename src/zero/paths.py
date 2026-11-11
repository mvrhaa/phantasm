"""Filesystem locations used by Zero on macOS.

Every function reads the environment on each call, so tests can redirect
all paths by setting ``ZERO_HOME``.
"""

import os
from pathlib import Path

from zero import APP_NAME

HOME_ENV = "ZERO_HOME"


def _home_override() -> Path | None:
    """Return the directory from ``ZERO_HOME``, or None if it is unset or empty."""
    value = os.environ.get(HOME_ENV)
    if not value:
        return None
    return Path(value).expanduser().absolute()


def home_dir() -> Path:
    """Return the data directory: ``ZERO_HOME`` or ``~/Library/Application Support/Zero``."""
    return _home_override() or Path.home() / "Library" / "Application Support" / APP_NAME


def config_path() -> Path:
    """Return the path of ``config.json``."""
    return home_dir() / "config.json"


def db_path() -> Path:
    """Return the path of the SQLite database."""
    return home_dir() / "zero.db"


def token_path() -> Path:
    """Return the path of the local access token file."""
    return home_dir() / "token"


def files_dir() -> Path:
    """Return the directory for attachments and files sent to the user."""
    return home_dir() / "files"


def logs_dir() -> Path:
    """Return the log directory: ``ZERO_HOME/logs`` or ``~/Library/Logs/Zero``."""
    override = _home_override()
    if override is not None:
        return override / "logs"
    return Path.home() / "Library" / "Logs" / APP_NAME


def ensure_dirs() -> None:
    """Create the data, files and log directories if they do not exist."""
    for directory in (home_dir(), files_dir(), logs_dir()):
        directory.mkdir(parents=True, exist_ok=True)
