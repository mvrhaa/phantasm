"""Tests for zero.paths."""

from pathlib import Path

import pytest

from zero import paths


def test_home_dir_uses_env_override(zero_home: Path) -> None:
    assert paths.home_dir() == zero_home


def test_home_dir_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ZERO_HOME", "~/custom")

    assert paths.home_dir() == tmp_path / "custom"


@pytest.mark.parametrize("value", [None, ""])
def test_defaults_without_override(
    value: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    if value is None:
        monkeypatch.delenv("ZERO_HOME")
    else:
        monkeypatch.setenv("ZERO_HOME", value)

    assert paths.home_dir() == tmp_path / "Library" / "Application Support" / "Zero"
    assert paths.logs_dir() == tmp_path / "Library" / "Logs" / "Zero"


def test_file_locations_inside_home(zero_home: Path) -> None:
    assert paths.config_path() == zero_home / "config.json"
    assert paths.db_path() == zero_home / "zero.db"
    assert paths.token_path() == zero_home / "token"
    assert paths.files_dir() == zero_home / "files"


def test_logs_dir_follows_override(zero_home: Path) -> None:
    assert paths.logs_dir() == zero_home / "logs"


def test_ensure_dirs_creates_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "nested" / "home"
    monkeypatch.setenv("ZERO_HOME", str(home))

    paths.ensure_dirs()

    assert home.is_dir()
    assert paths.files_dir().is_dir()
    assert paths.logs_dir().is_dir()


def test_ensure_dirs_is_idempotent(zero_home: Path) -> None:
    marker = zero_home / "config.json"
    marker.write_text("{}")

    paths.ensure_dirs()
    paths.ensure_dirs()

    assert marker.read_text() == "{}"
    assert paths.files_dir().is_dir()
