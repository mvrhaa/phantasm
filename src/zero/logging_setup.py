"""Logging configuration: a rotating log file plus the console."""

import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from zero.paths import logs_dir

LOG_ENV = "ZERO_LOG"
LOG_FILE_NAME = "zero.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
DEFAULT_LEVEL = logging.INFO

_FILE_HANDLER = "zero-file"
_CONSOLE_HANDLER = "zero-console"
_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_TOKEN_IN_URL = re.compile(r"(token=)[^&\s\"']+", re.IGNORECASE)

logger = logging.getLogger(__name__)


class RedactTokenFilter(logging.Filter):
    """Hide the access token when a URL with ``?token=`` gets logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "token=" in message.lower():
            record.msg = _TOKEN_IN_URL.sub(r"\1***", message)
            record.args = None
        return True


def log_file_path() -> Path:
    """Return the path of the main log file."""
    return logs_dir() / LOG_FILE_NAME


def _formatter() -> logging.Formatter:
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime  # log timestamps are UTC, like everything else
    return formatter


def reset_logging() -> None:
    """Remove and close the handlers installed by ``setup_logging``."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in (_FILE_HANDLER, _CONSOLE_HANDLER):
            root.removeHandler(handler)
            handler.close()


def setup_logging() -> Path:
    """Send all logs to the rotating log file and the console.

    The level comes from the ``ZERO_LOG`` environment variable (DEBUG, INFO,
    WARNING, ERROR or CRITICAL; INFO if unset). Calling this again replaces
    the handlers instead of adding duplicates. Returns the log file path.
    """
    path = log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    requested = os.environ.get(LOG_ENV, "").strip()
    level = _LEVELS.get(requested.upper(), DEFAULT_LEVEL)

    file_handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.set_name(_FILE_HANDLER)
    console_handler = logging.StreamHandler()
    console_handler.set_name(_CONSOLE_HANDLER)

    reset_logging()
    root = logging.getLogger()
    for handler in (file_handler, console_handler):
        handler.setFormatter(_formatter())
        handler.addFilter(RedactTokenFilter())
        root.addHandler(handler)
    root.setLevel(level)

    if requested and requested.upper() not in _LEVELS:
        logger.warning("Unknown %s level %r, using INFO", LOG_ENV, requested)
    return path
