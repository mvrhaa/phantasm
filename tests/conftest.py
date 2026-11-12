"""Shared pytest fixtures."""

import logging
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.turns import Harness
from zero import clock
from zero.logging_setup import reset_logging
from zero.paths import db_path
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo


@pytest.fixture(autouse=True)
def zero_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ZERO_HOME at a temporary directory so tests never touch real app data.

    The .env file moves next to it, so no test ever reads the real API keys.
    """
    monkeypatch.setenv("ZERO_HOME", str(tmp_path))
    monkeypatch.setenv("ZERO_ENV_FILE", str(tmp_path.parent / f"{tmp_path.name}.env"))
    return tmp_path


@pytest.fixture
def clean_root_logger() -> Iterator[None]:
    """Undo the changes ``setup_logging`` makes to the global root logger."""
    root = logging.getLogger()
    level = root.level
    yield
    reset_logging()
    root.setLevel(level)


class FrozenClock:
    """A time source that only moves when a test tells it to."""

    def __init__(self, start: datetime) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        """Move the clock forward by the given number of seconds."""
        self.current += timedelta(seconds=seconds)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> FrozenClock:
    """Make ``zero.clock.now()`` return a fixed time controlled by the test."""
    frozen = FrozenClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(clock, "_source", frozen)
    return frozen


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    """A fresh session in a migrated database, for running turns without a WebSocket."""
    async with Database(db_path()) as db:
        await db.migrate()
        sessions = SessionsRepo(db)
        session = await sessions.create()
        yield Harness(sessions, MessagesRepo(db), session.id)
