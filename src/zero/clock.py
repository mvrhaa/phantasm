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
