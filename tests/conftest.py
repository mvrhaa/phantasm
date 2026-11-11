"""Shared pytest fixtures."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from zero import clock


@pytest.fixture(autouse=True)
def zero_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ZERO_HOME at a temporary directory so tests never touch real app data."""
    monkeypatch.setenv("ZERO_HOME", str(tmp_path))
    return tmp_path


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
