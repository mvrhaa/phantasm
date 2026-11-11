"""Local access token: created on first run and readable only by the owner.

The token itself is never written to the logs.
"""

import hmac
import logging
import os
import secrets
import stat
from pathlib import Path

from zero.paths import token_path

TOKEN_BYTES = 32
TOKEN_FILE_MODE = 0o600

logger = logging.getLogger(__name__)


def _read_token(path: Path) -> str | None:
    """Return the stored token, or None if the file is missing or blank."""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return token or None


def _write_token(path: Path, token: str) -> None:
    """Write the token so that the file is never readable by other users."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, TOKEN_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        os.fchmod(fd, TOKEN_FILE_MODE)  # the mode given to os.open is subject to umask
        file.write(token)
    os.replace(tmp_path, path)


def get_or_create_token(path: Path | None = None) -> str:
    """Return the local access token, generating and saving it on first use.

    The token file always ends up with mode 600; looser permissions on an
    existing file are tightened.
    """
    path = path or token_path()
    token = _read_token(path)
    if token is None:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        _write_token(path, token)
        logger.info("Created a new access token file at %s", path)
        return token

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != TOKEN_FILE_MODE:
        path.chmod(TOKEN_FILE_MODE)
        logger.warning("Tightened permissions of %s from %o to 600", path, mode)
    return token


def verify(candidate: str | None, expected: str) -> bool:
    """Check a token presented by a client against the real one in constant time."""
    if not candidate or not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))
