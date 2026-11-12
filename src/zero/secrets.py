"""API keys. They live in one place only: the ``.env`` file at the root of the project.

A key is read from that file when it is needed and is never written anywhere
else: not into the config, the database, the logs, error messages or command
lines. Its value is masked as ``***`` wherever text leaves Zero's hands, and
the file itself is off limits to the file tools, so the model cannot read it.
"""

import os
import stat
from pathlib import Path

from zero.paths import home_dir
from zero.providers.presets import PRESETS

ENV_FILE_VAR = "ZERO_ENV_FILE"  # where tests point Zero at their own .env
KEY_NAMES: dict[str, str] = {p.name: p.key_name for p in PRESETS.values() if p.key_name}
MASK = "***"
MIN_SECRET_CHARS = 8  # shorter values are not masked: they would hide ordinary words

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The values last read from the .env file, with the file's state when they
# were read. A cache of a file, re-read whenever the file changes.
_cache: tuple[tuple[str, float, int], frozenset[str]] | None = None


def env_file() -> Path:
    """The .env file: ``$ZERO_ENV_FILE`` if set, else the root of the project.

    A copy of Zero installed outside a checkout of the project looks in the
    data folder instead.
    """
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        return Path(override).expanduser().absolute()
    if (_PROJECT_ROOT / "pyproject.toml").exists():
        return _PROJECT_ROOT / ".env"
    return home_dir() / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """The ``KEY=value`` lines of a .env file; a missing file gives an empty dict.

    Blank lines, ``#`` comments, an ``export`` in front and quotes around the
    value are allowed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            values[name] = value
    return values


def api_key(provider: str) -> str | None:
    """The API key of ``provider`` from the .env file, or None if it is not there.

    Only the file counts: a variable of the same name in the environment is
    ignored, so the key has exactly one home.
    """
    name = KEY_NAMES.get(provider)
    if name is None:
        return None
    return read_env_file(env_file()).get(name) or None


def open_to_others(path: Path) -> bool:
    """True if users other than the owner may read or write the file."""
    try:
        return bool(stat.S_IMODE(path.stat().st_mode) & 0o077)
    except FileNotFoundError:
        return False


def _secret_values() -> frozenset[str]:
    global _cache
    path = env_file()
    try:
        info = path.stat()
    except FileNotFoundError:
        return frozenset()
    state = (str(path), info.st_mtime, info.st_size)
    if _cache is None or _cache[0] != state:
        values = read_env_file(path).values()
        _cache = (state, frozenset(v for v in values if len(v) >= MIN_SECRET_CHARS))
    return _cache[1]


def mask_secrets(text: str) -> str:
    """``text`` with every value from the .env file replaced by ``***``."""
    for secret in _secret_values():
        if secret in text:
            text = text.replace(secret, MASK)
    return text
