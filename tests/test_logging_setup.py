"""Tests for zero.logging_setup."""

import logging
import re
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from zero import logging_setup
from zero.logging_setup import reset_logging, setup_logging
from zero.paths import logs_dir


@pytest.fixture(autouse=True)
def clean_root_logger() -> Iterator[None]:
    """Undo the changes setup_logging makes to the global root logger."""
    root = logging.getLogger()
    level = root.level
    yield
    reset_logging()
    root.setLevel(level)


def own_handlers() -> list[logging.Handler]:
    names = ("zero-file", "zero-console")
    return [h for h in logging.getLogger().handlers if h.get_name() in names]


def test_log_file_appears_in_logs_dir() -> None:
    path = setup_logging()

    logging.getLogger("zero.test").info("hello from the test")

    assert path == logs_dir() / "zero.log"
    assert "hello from the test" in path.read_text(encoding="utf-8")


def test_log_directory_is_created_when_missing(zero_home: Path) -> None:
    assert not (zero_home / "logs").exists()

    setup_logging()

    assert (zero_home / "logs").is_dir()


def test_log_line_has_utc_timestamp_level_and_logger_name() -> None:
    path = setup_logging()

    logging.getLogger("zero.test").warning("formatted")

    line = path.read_text(encoding="utf-8").strip()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z WARNING  zero\.test: formatted", line
    )


def test_file_handler_rotates_five_files_of_five_megabytes() -> None:
    setup_logging()

    handlers = [h for h in own_handlers() if isinstance(h, RotatingFileHandler)]

    assert len(handlers) == 1
    assert handlers[0].maxBytes == 5 * 1024 * 1024
    assert handlers[0].backupCount == 5


def test_console_handler_is_installed() -> None:
    setup_logging()

    console = [h for h in own_handlers() if not isinstance(h, RotatingFileHandler)]

    assert len(console) == 1
    assert isinstance(console[0], logging.StreamHandler)


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_LOG", raising=False)

    path = setup_logging()
    logging.getLogger("zero.test").debug("hidden")
    logging.getLogger("zero.test").info("visible")

    text = path.read_text(encoding="utf-8")
    assert logging.getLogger().level == logging.INFO
    assert "hidden" not in text
    assert "visible" in text


@pytest.mark.parametrize(
    ("value", "expected"),
    [("DEBUG", logging.DEBUG), ("debug", logging.DEBUG), (" Warning ", logging.WARNING)],
)
def test_level_comes_from_env(value: str, expected: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_LOG", value)

    setup_logging()

    assert logging.getLogger().level == expected


def test_unknown_level_falls_back_to_info_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_LOG", "loud")

    path = setup_logging()

    assert logging.getLogger().level == logging.INFO
    assert "Unknown ZERO_LOG level 'loud'" in path.read_text(encoding="utf-8")


def test_repeated_setup_does_not_duplicate_handlers() -> None:
    setup_logging()
    path = setup_logging()

    logging.getLogger("zero.test").info("only once")

    assert len(own_handlers()) == 2
    assert path.read_text(encoding="utf-8").count("only once") == 1


def test_setup_keeps_handlers_installed_by_others() -> None:
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        setup_logging()
        reset_logging()

        assert foreign in root.handlers
        assert own_handlers() == []
    finally:
        root.removeHandler(foreign)


def test_constants_match_spec() -> None:
    assert logging_setup.LOG_FILE_NAME == "zero.log"
    assert logging_setup.LOG_ENV == "ZERO_LOG"
