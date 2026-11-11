"""Tests for zero.clock."""

from datetime import UTC, datetime, timedelta

from zero import clock
from zero.clock import now as imported_now


def test_now_is_timezone_aware_utc() -> None:
    current = clock.now()

    assert current.tzinfo is UTC
    assert current.utcoffset() == timedelta(0)


def test_now_follows_the_system_clock() -> None:
    before = datetime.now(UTC)
    current = clock.now()
    after = datetime.now(UTC)

    assert before <= current <= after


def test_frozen_clock_replaces_now(frozen_clock) -> None:
    assert clock.now() == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert clock.now() == clock.now()


def test_frozen_clock_advances_on_request(frozen_clock) -> None:
    start = clock.now()

    frozen_clock.advance(90)

    assert clock.now() - start == timedelta(seconds=90)


def test_frozen_clock_affects_directly_imported_now(frozen_clock) -> None:
    assert imported_now() == frozen_clock.current
