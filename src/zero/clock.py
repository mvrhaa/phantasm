"""The single source of the current time.

All code asks ``clock.now()`` instead of calling ``datetime.now`` directly,
so tests can replace the time source in one place (see the ``frozen_clock``
fixture in ``tests/conftest.py``).
"""

from collections.abc import Callable
from datetime import UTC, datetime


def _system_now() -> datetime:
    return datetime.now(UTC)


_source: Callable[[], datetime] = _system_now


def now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return _source()


def to_iso(moment: datetime) -> str:
    """Format a time for storage: ISO-8601 in UTC, always with microseconds.

    The fixed width makes these strings sort in chronological order.
    """
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def from_iso(text: str) -> datetime:
    """Parse a time written by ``to_iso`` back into an aware UTC datetime."""
    return datetime.fromisoformat(text).astimezone(UTC)
