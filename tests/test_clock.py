"""Tests for zero.clock."""

from datetime import UTC, datetime, timedelta, timezone

from zero import clock
from zero.clock import from_iso, to_iso
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


def test_to_iso_is_fixed_width_utc() -> None:
    whole_second = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    other_zone = datetime(2026, 1, 1, 14, 0, 0, 5, tzinfo=timezone(timedelta(hours=2)))

    assert to_iso(whole_second) == "2026-01-01T12:00:00.000000+00:00"
    assert to_iso(other_zone) == "2026-01-01T12:00:00.000005+00:00"


def test_iso_strings_sort_chronologically() -> None:
    moments = [datetime(2026, 1, 1, 12, 0, 0, micro, tzinfo=UTC) for micro in (0, 5, 100000)]

    assert sorted(to_iso(moment) for moment in reversed(moments)) == [
        to_iso(moment) for moment in moments
    ]


def test_from_iso_reverses_to_iso() -> None:
    moment = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)

    restored = from_iso(to_iso(moment))

    assert restored == moment
    assert restored.utcoffset() == timedelta(0)
