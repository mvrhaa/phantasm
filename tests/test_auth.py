"""Tests for zero.server.auth."""

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from zero.paths import token_path
from zero.server.auth import get_or_create_token, verify


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def permissive_umask() -> Iterator[None]:
    """Run the test with umask 0, the worst case for new file permissions."""
    previous = os.umask(0)
    yield
    os.umask(previous)


def test_token_is_created_on_first_call() -> None:
    assert not token_path().exists()

    token = get_or_create_token()

    assert len(token) == 43
    assert token_path().read_text(encoding="utf-8") == token


def test_token_file_has_mode_600() -> None:
    get_or_create_token()

    assert file_mode(token_path()) == 0o600


def test_token_file_has_mode_600_even_with_permissive_umask(permissive_umask: None) -> None:
    get_or_create_token()

    assert file_mode(token_path()) == 0o600


def test_second_call_returns_the_same_token() -> None:
    first = get_or_create_token()
    second = get_or_create_token()

    assert first == second


def test_each_new_token_is_different(tmp_path: Path) -> None:
    first = get_or_create_token(tmp_path / "a" / "token")
    second = get_or_create_token(tmp_path / "b" / "token")

    assert first != second


def test_missing_parent_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "home" / "token"

    get_or_create_token(path)

    assert path.is_file()


def test_no_temporary_file_is_left(zero_home: Path) -> None:
    get_or_create_token()

    assert [p.name for p in zero_home.iterdir()] == ["token"]


def test_existing_token_is_read_without_surrounding_whitespace() -> None:
    token_path().write_text("my-existing-token\n", encoding="utf-8")
    token_path().chmod(0o600)

    assert get_or_create_token() == "my-existing-token"


def test_loose_permissions_on_existing_token_are_tightened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token_path().write_text("my-existing-token", encoding="utf-8")
    token_path().chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="zero.server.auth"):
        token = get_or_create_token()

    assert token == "my-existing-token"
    assert file_mode(token_path()) == 0o600
    assert "644" in caplog.text


@pytest.mark.parametrize("content", ["", "  \n"])
def test_blank_token_file_is_regenerated(content: str) -> None:
    token_path().write_text(content, encoding="utf-8")
    token_path().chmod(0o644)

    token = get_or_create_token()

    assert len(token) == 43
    assert token_path().read_text(encoding="utf-8") == token
    assert file_mode(token_path()) == 0o600


def test_token_never_appears_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        created = get_or_create_token()
        token_path().chmod(0o644)
        get_or_create_token()

    assert caplog.records
    assert created not in caplog.text


def test_verify_accepts_the_right_token() -> None:
    token = get_or_create_token()

    assert verify(token, token) is True


@pytest.mark.parametrize(
    "candidate", [None, "", "wrong", "тоукен", "secret-token ", "SECRET-TOKEN"]
)
def test_verify_rejects_everything_else(candidate: str | None) -> None:
    assert verify(candidate, "secret-token") is False


def test_verify_never_accepts_an_empty_expected_token() -> None:
    assert verify("", "") is False
    assert verify("anything", "") is False


def test_corrupted_token_file_is_regenerated(caplog: pytest.LogCaptureFixture) -> None:
    token_path().write_bytes(b"\xff\xfe\x00garbage")

    with caplog.at_level(logging.WARNING, logger="zero.server.auth"):
        token = get_or_create_token()

    assert len(token) == 43
    assert token_path().read_text(encoding="utf-8") == token
    assert file_mode(token_path()) == 0o600
    assert "corrupted" in caplog.text
