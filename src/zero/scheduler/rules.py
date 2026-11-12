"""Schedule rules: how "when" is written, and the next time a rule fires.

The forms of ``when``:

- ``in 2m``, ``in 90 minutes``, ``in 1 hour``, ``in 2d`` — once, that long from now;
- ``at 2026-11-12T15:30`` — once, at that moment (local time unless an offset is given);
- ``every 30m``, ``every 2 hours`` — again and again, at least a minute apart;
- ``daily at 09:00``, ``every day at 09:00`` — every day at that local time;
- ``every mon,wed,fri at 09:00``, ``every weekdays at 08:30``, ``every weekends at 10:00``.

Weekly rules are kept in local wall time, so 09:00 stays 09:00 across a
change to or from summer time. Every time handed out is in UTC.
"""

import os
import re
from datetime import UTC, datetime, time, timedelta, tzinfo
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, TypeAdapter

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_GROUPS = {"day": list(range(7)), "weekdays": list(range(5)), "weekends": [5, 6]}
MIN_INTERVAL = timedelta(minutes=1)

_UNITS = {
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}  # fmt: skip
_DURATION = re.compile(r"^(\d+)\s*([a-z]+)$")
_CLOCK = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

HELP = (
    "Write when as one of: 'in 2m', 'at 2026-11-12T15:30', 'every 30m', "
    "'daily at 09:00', 'every mon,wed,fri at 09:00' (also 'every weekdays at 08:30')."
)


class Once(BaseModel):
    """Run once, at ``at``."""

    kind: Literal["once"] = "once"
    at: datetime


class Interval(BaseModel):
    """Run every ``every_s`` seconds, counted from ``anchor``."""

    kind: Literal["interval"] = "interval"
    every_s: int = Field(ge=int(MIN_INTERVAL.total_seconds()))
    anchor: datetime


class Weekly(BaseModel):
    """Run on the given weekdays (0 is Monday) at a local time of day."""

    kind: Literal["weekly"] = "weekly"
    days: list[int]
    at: time


Rule = Annotated[Once | Interval | Weekly, Field(discriminator="kind")]
rule_adapter: TypeAdapter[Rule] = TypeAdapter(Rule)


class RuleError(ValueError):
    """``when`` cannot be understood; the message says how to write it."""


def local_zone() -> ZoneInfo:
    """The computer's time zone by its IANA name (``$TZ``, then /etc/localtime), or UTC."""
    names = [os.environ.get("TZ", "")]
    try:
        names.append(os.readlink("/etc/localtime").split("zoneinfo/", 1)[1])
    except (OSError, IndexError):
        pass
    for name in names:
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                continue
    return ZoneInfo("UTC")


def zone_named(name: str) -> ZoneInfo:
    """A time zone by its IANA name; raises ``RuleError`` for an unknown one."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise RuleError(f"Unknown time zone {name!r}; use a name like 'Europe/Prague'.") from None


def _duration(text: str) -> timedelta:
    match = _DURATION.match(text.strip())
    if match is None or match.group(2) not in _UNITS:
        raise RuleError(f"{text!r} is not a duration such as 30m, 2h or 1d. {HELP}")
    return timedelta(seconds=int(match.group(1)) * _UNITS[match.group(2)])


def _clock(text: str) -> time:
    match = _CLOCK.match(text)
    if match is None:
        raise RuleError(f"{text!r} is not a time of day such as 09:00. {HELP}")
    return time(int(match.group(1)), int(match.group(2)))


def _days(text: str) -> list[int]:
    if text in DAY_GROUPS:
        return DAY_GROUPS[text]
    days = set()
    for part in text.split(","):
        part = part.strip()[:3]
        if part not in WEEKDAYS:
            raise RuleError(f"{text!r} is not a list of weekdays such as mon,wed,fri. {HELP}")
        days.add(WEEKDAYS.index(part))
    return sorted(days)


def parse_rule(when: str, now: datetime, zone: tzinfo) -> Rule:
    """Turn ``when`` into a rule; raises ``RuleError`` with a hint if it cannot.

    ``now`` anchors relative forms; a one-off time must lie in the future.
    """
    text = " ".join(when.strip().lower().split())
    if text.startswith("in "):
        delay = _duration(text[3:])
        if delay < MIN_INTERVAL:
            raise RuleError("A schedule can start a minute from now at the soonest.")
        return Once(at=(now + delay).astimezone(UTC))
    if text.startswith("at "):
        return _at(when.strip()[3:].strip(), now, zone)
    if text.startswith("daily at "):
        return Weekly(days=DAY_GROUPS["day"], at=_clock(text[len("daily at ") :]))
    if text.startswith("every "):
        rest = text[len("every ") :]
        if " at " in rest:
            days, clock = rest.split(" at ", 1)
            return Weekly(days=_days(days.replace(" ", "")), at=_clock(clock))
        every = _duration(rest)
        if every < MIN_INTERVAL:
            raise RuleError("Schedules repeat at most once a minute.")
        return Interval(every_s=int(every.total_seconds()), anchor=now.astimezone(UTC))
    raise RuleError(f"Cannot understand {when!r}. {HELP}")


def _at(text: str, now: datetime, zone: tzinfo) -> Once:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise RuleError(
            f"{text!r} is not a date and time such as 2026-11-12T15:30. {HELP}"
        ) from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    if moment <= now:
        raise RuleError(f"{text} is already in the past.")
    return Once(at=moment.astimezone(UTC))


def next_run(rule: Rule, now: datetime, zone: tzinfo) -> datetime | None:
    """The first time after ``now`` that the rule fires, in UTC; None if it never will again."""
    if isinstance(rule, Once):
        return rule.at if rule.at > now else None
    if isinstance(rule, Interval):
        if now < rule.anchor:
            return rule.anchor
        every = timedelta(seconds=rule.every_s)
        steps = (now - rule.anchor) // every + 1
        return (rule.anchor + steps * every).astimezone(UTC)
    local_today = now.astimezone(zone).date()
    for offset in range(8):
        day = local_today + timedelta(days=offset)
        if day.weekday() in rule.days:
            candidate = datetime.combine(day, rule.at, tzinfo=zone)
            if candidate > now:
                return candidate.astimezone(UTC)
    return None


def describe(rule: Rule, zone: tzinfo) -> str:
    """The rule in a few words, in local time."""
    if isinstance(rule, Once):
        return f"once at {rule.at.astimezone(zone):%a %Y-%m-%d %H:%M}"
    if isinstance(rule, Interval):
        seconds = rule.every_s
        for unit, size in (("d", 86400), ("h", 3600)):
            if seconds % size == 0:
                return f"every {seconds // size}{unit}"
        return f"every {seconds // 60}m"
    days = "day" if rule.days == DAY_GROUPS["day"] else ",".join(WEEKDAYS[d] for d in rule.days)
    return f"every {days} at {rule.at:%H:%M}"
