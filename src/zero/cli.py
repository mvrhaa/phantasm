"""The ``zero`` command: run the service and look after it.

zero serve          start the service (the same as ``python -m zero``)
zero token          show the local token clients connect with
zero doctor         check folders, config, database, token, port, provider and API key
zero usage          how many model requests and tokens were used today and in 30 days
zero config path    show where the config file is
"""

import argparse
import os
import shutil
import sqlite3
import stat
import sys
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from zero import APP_NAME, __version__, clock
from zero.__main__ import PortState, probe_port
from zero.__main__ import main as serve
from zero.config import Config, ConfigError, load_config
from zero.paths import config_path, db_path, files_dir, home_dir, logs_dir, token_path
from zero.providers.registry import ProviderNotConfigured, get_provider, small_model
from zero.secrets import KEY_NAMES, api_key, env_file, open_to_others
from zero.server.auth import get_or_create_token
from zero.store.db import MIGRATIONS_DIR

Level = Literal["ok", "warn", "fail"]
_MARKS = {"ok": "✓", "warn": "!", "fail": "✗"}


@dataclass(frozen=True)
class Check:
    """One line of ``zero doctor``."""

    level: Level
    name: str
    detail: str


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


# --- doctor ------------------------------------------------------------------


def check_folders() -> list[Check]:
    """Every data folder exists and can be written, or does not exist yet."""
    checks = []
    for name, folder in (("data", home_dir()), ("logs", logs_dir()), ("files", files_dir())):
        if not folder.exists():
            checks.append(Check("ok", f"{name} folder", f"{folder} will be created on start"))
        elif not os.access(folder, os.W_OK):
            checks.append(Check("fail", f"{name} folder", f"{folder} cannot be written"))
        else:
            checks.append(Check("ok", f"{name} folder", str(folder)))
    return checks


def check_config() -> tuple[Check, Config | None]:
    """The config file loads (or there is none yet and the defaults apply)."""
    path = config_path()
    if not path.exists():
        return Check("ok", "config", f"{path} will be created with defaults"), Config()
    try:
        return Check("ok", "config", str(path)), load_config(path)
    except ConfigError as exc:
        return Check("fail", "config", str(exc)), None


def latest_schema() -> int:
    """The newest schema version this version of Zero brings."""
    return max(int(path.name.split("_", 1)[0]) for path in MIGRATIONS_DIR.glob("*.sql"))


def check_database() -> Check:
    """The database opens and its schema is one this version can use; nothing is changed."""
    path = db_path()
    if not path.exists():
        return Check("ok", "database", "will be created on start")
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.Error as exc:
        return Check("fail", "database", f"{path} cannot be read: {exc}")
    version, latest = (row[0] or 0), latest_schema()
    if version > latest:
        return Check("fail", "database", f"schema {version} is newer than this Zero ({latest})")
    if version < latest:
        return Check("ok", "database", f"schema {version}, upgraded to {latest} on start")
    return Check("ok", "database", f"schema {version}")


def check_token() -> Check:
    """The token file is readable by its owner only."""
    path = token_path()
    if not path.exists():
        return Check("ok", "token", "will be created on start")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        return Check("fail", "token", f"{path} can be read by others; run chmod 600 on it")
    return Check("ok", "token", str(path))


def check_port(config: Config, probe: Callable[[str, int], object]) -> Check:
    """The service's port is free or used by Zero itself."""
    host, port = config.server.host, config.server.port
    state = probe(host, port)
    if state is PortState.ZERO_RUNNING:
        return Check("ok", "port", f"{APP_NAME} is running on {host}:{port}")
    if state is PortState.FREE:
        return Check("ok", "port", f"{host}:{port} is free")
    return Check("fail", "port", f"{host}:{port} is used by another program")


def check_provider(config: Config) -> Check:
    """The configured provider exists."""
    try:
        provider = get_provider(config.provider)
    except ProviderNotConfigured as exc:
        return Check("fail", "provider", str(exc))
    model = config.provider.model
    small = small_model(config.provider)
    extra = f", small model {small}" if small != model else ""
    return Check("ok", "provider", f"{provider.name}, model {model}{extra}")


def check_api_key(config: Config) -> Check:
    """The provider's key is in the .env file, and only its owner can read the file.

    The key itself is never shown.
    """
    path = env_file()
    if open_to_others(path):
        return Check("fail", "key", f"{path} can be read by others; run chmod 600 on it")
    name = KEY_NAMES.get(config.provider.name)
    if name is None:
        return Check("ok", "key", f"none needed for the {config.provider.name} provider")
    if api_key(config.provider.name) is None:
        return Check("fail", "key", f"{name} is missing in {path}")
    return Check("ok", "key", f"{name} found in {path}")


def check_ripgrep() -> Check:
    """ripgrep makes search_files faster; without it a Python search is used."""
    found = shutil.which("rg")
    if found is None:
        return Check("warn", "ripgrep", "not installed; search_files uses a slower Python search")
    return Check("ok", "ripgrep", found)


def doctor(probe: Callable[[str, int], object] | None = None) -> list[Check]:
    """Run every check; nothing on disk is created or changed."""
    probe = probe or probe_port
    checks = check_folders()
    config_check, config = check_config()
    checks += [config_check, check_database(), check_token()]
    if config is not None:
        checks += [check_port(config, probe), check_provider(config), check_api_key(config)]
    checks.append(check_ripgrep())
    return checks


# --- usage -------------------------------------------------------------------

USAGE_SQL = (
    "SELECT provider, model, COUNT(*), SUM(1 - ok), SUM(input_tokens), SUM(output_tokens) "
    "FROM model_calls WHERE created_at >= ? GROUP BY provider, model ORDER BY provider, model"
)


def usage_since(since: datetime) -> list[tuple[str, str, int, int, int, int]] | None:
    """Requests, failures and tokens per provider and model; None if nothing is recorded."""
    path = db_path()
    if not path.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            return conn.execute(USAGE_SQL, (clock.to_iso(since),)).fetchall()
    except sqlite3.Error:
        return None  # an older database without the table


def _usage() -> int:
    now = clock.now()
    periods = (
        ("Today (since 00:00 UTC)", now.replace(hour=0, minute=0, second=0, microsecond=0)),
        ("Last 30 days", now - timedelta(days=30)),
    )
    for title, since in periods:
        rows = usage_since(since)
        _say(f"{title}:")
        if not rows:
            _say("  no model calls")
            continue
        for provider, model, requests, failed, tokens_in, tokens_out in rows:
            _say(
                f"  {provider} / {model}: {requests} requests ({failed} failed), "
                f"{tokens_in} tokens in, {tokens_out} out"
            )
    return 0


# --- The command -------------------------------------------------------------


def _doctor(probe: Callable[[str, int], object] | None) -> int:
    checks = doctor(probe)
    width = max(len(check.name) for check in checks)
    for check in checks:
        _say(f"{_MARKS[check.level]} {check.name.ljust(width)}  {check.detail}")
    failed = sum(1 for check in checks if check.level == "fail")
    _say("All good." if not failed else f"{failed} problem{'s' if failed > 1 else ''} found.")
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero", description=f"{APP_NAME}, the local brain.")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="start the service")
    commands.add_parser("token", help="show the token clients connect with")
    commands.add_parser("doctor", help="check that everything is in order")
    commands.add_parser("usage", help="model requests and tokens used lately")
    config = commands.add_parser("config", help="about the config file")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("path", help="show where the config file is")
    return parser


def main(
    argv: Sequence[str] | None = None, *, probe: Callable[[str, int], object] | None = None
) -> int:
    """Run the ``zero`` command and return its exit code."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "serve":
        return serve()
    if arguments.command == "token":
        _say(get_or_create_token())
        return 0
    if arguments.command == "doctor":
        return _doctor(probe)
    if arguments.command == "usage":
        return _usage()
    _say(str(config_path()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
