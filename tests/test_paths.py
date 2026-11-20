"""Tests for phantasm.paths."""

from pathlib import Path

import pytest

from phantasm import paths


def test_home_dir_uses_env_override(phantasm_home: Path) -> None:
    assert paths.home_dir() == phantasm_home


def test_home_dir_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PHANTASM_HOME", "~/custom")

    assert paths.home_dir() == tmp_path / "custom"


@pytest.mark.parametrize("value", [None, ""])
def test_defaults_without_override(
    value: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    if value is None:
        monkeypatch.delenv("PHANTASM_HOME")
    else:
        monkeypatch.setenv("PHANTASM_HOME", value)

    assert paths.home_dir() == tmp_path / "Library" / "Application Support" / "Phantasm"
    assert paths.logs_dir() == tmp_path / "Library" / "Logs" / "Phantasm"


def test_file_locations_inside_home(phantasm_home: Path) -> None:
    assert paths.config_path() == phantasm_home / "config.json"
    assert paths.db_path() == phantasm_home / "phantasm.db"
    assert paths.token_path() == phantasm_home / "token"
    assert paths.files_dir() == phantasm_home / "files"


def test_logs_dir_follows_override(phantasm_home: Path) -> None:
    assert paths.logs_dir() == phantasm_home / "logs"


def test_ensure_dirs_creates_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "nested" / "home"
    monkeypatch.setenv("PHANTASM_HOME", str(home))

    paths.ensure_dirs()

    assert home.is_dir()
    assert paths.files_dir().is_dir()
    assert paths.logs_dir().is_dir()


def test_ensure_dirs_is_idempotent(phantasm_home: Path) -> None:
    marker = phantasm_home / "config.json"
    marker.write_text("{}")

    paths.ensure_dirs()
    paths.ensure_dirs()

    assert marker.read_text() == "{}"
    assert paths.files_dir().is_dir()
