"""Tests of the ``zero`` command."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from zero import cli
from zero.__main__ import PortState
from zero.paths import config_path, db_path, token_path


def run(
    capsys: pytest.CaptureFixture[str], *argv: str, state: PortState = PortState.FREE
) -> tuple[int, str]:
    code = cli.main(list(argv), probe=lambda host, port: state)
    return code, capsys.readouterr().out


def test_config_path(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(capsys, "config", "path") == (0, f"{config_path()}\n")


def test_token_is_created_once_and_shown(capsys: pytest.CaptureFixture[str]) -> None:
    code, first = run(capsys, "token")
    _, second = run(capsys, "token")

    assert code == 0
    assert first == second == token_path().read_text().strip() + "\n"


def test_serve_starts_the_service(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "serve", lambda: calls.append("served") or 0)

    assert cli.main(["serve"]) == 0
    assert calls == ["served"]


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])

    assert capsys.readouterr().out.startswith("Zero ")


# --- doctor ------------------------------------------------------------------


def test_doctor_on_a_fresh_install_is_green_and_changes_nothing(
    capsys: pytest.CaptureFixture[str], zero_home: Path
) -> None:
    code, output = run(capsys, "doctor")

    assert code == 0
    assert output.splitlines()[-1] == "All good."
    for name in ("config", "database", "token", "port", "provider"):
        assert f"✓ {name}" in output
    assert list(zero_home.iterdir()) == []


def test_doctor_after_a_start_reports_the_files(capsys: pytest.CaptureFixture[str]) -> None:
    run(capsys, "token")
    with sqlite3.connect(db_path()) as conn:
        conn.executescript(
            "CREATE TABLE schema_version (version INTEGER NOT NULL);"
            f"INSERT INTO schema_version VALUES ({cli.latest_schema()});"
        )

    code, output = run(capsys, "doctor", state=PortState.ZERO_RUNNING)

    assert code == 0
    assert f"✓ database      schema {cli.latest_schema()}" in output
    assert f"✓ token         {token_path()}" in output
    assert "Zero is running on 127.0.0.1:47400" in output


@pytest.mark.parametrize(
    ("version", "expected"),
    [(1, "upgraded to"), (999, "newer than this Zero")],
)
def test_doctor_compares_the_schema(
    capsys: pytest.CaptureFixture[str], version: int, expected: str
) -> None:
    db_path().parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path()) as conn:
        conn.executescript(
            "CREATE TABLE schema_version (version INTEGER NOT NULL);"
            f"INSERT INTO schema_version VALUES ({version});"
        )

    code, output = run(capsys, "doctor")

    assert expected in output
    assert code == (1 if version == 999 else 0)


@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        ("bad_config", "✗ config"),
        ("open_token", "can be read by others; run chmod 600"),
        ("taken_port", "is used by another program"),
        ("unknown_provider", "✗ provider"),
        ("broken_db", "✗ database"),
    ],
)
def test_doctor_finds_problems(
    capsys: pytest.CaptureFixture[str], zero_home: Path, setup: str, problem: str
) -> None:
    state = PortState.FREE
    if setup == "bad_config":
        config_path().write_text("{not json")
    elif setup == "open_token":
        run(capsys, "token")
        token_path().chmod(0o644)
    elif setup == "taken_port":
        state = PortState.TAKEN
    elif setup == "unknown_provider":
        config_path().write_text(json.dumps({"provider": {"name": "nope"}}))
    elif setup == "broken_db":
        db_path().write_bytes(b"this is not a database")

    code, output = run(capsys, "doctor", state=state)

    assert code == 1
    assert problem in output
    assert output.splitlines()[-1] == "1 problem found."


def test_doctor_without_ripgrep_only_warns(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    code, output = run(capsys, "doctor")

    assert code == 0
    assert "! ripgrep" in output


def test_entry_point_is_declared() -> None:
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()

    assert '[project.scripts]\nzero = "zero.cli:main"' in pyproject


def test_doctor_lists_every_check(capsys: pytest.CaptureFixture[str]) -> None:
    _, output = run(capsys, "doctor")
    names: list[Any] = [line.split()[1] for line in output.splitlines()[:-1]]

    assert names == [
        "data",
        "logs",
        "files",
        "config",
        "database",
        "token",
        "port",
        "provider",
        "ripgrep",
    ]
