"""Tests of schedules: the rules and their next runs, the repository and the tools."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.turns import FakeApprover, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.paths import db_path
from zero.providers.base import ToolCall
from zero.scheduler import rules
from zero.scheduler.rules import (
    Interval,
    Once,
    RuleError,
    Weekly,
    describe,
    local_zone,
    next_run,
    parse_rule,
)
from zero.store.db import Database
from zero.store.schedules import SchedulesRepo
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.permissions import PathGuard

PRAGUE = ZoneInfo("Europe/Prague")
NEW_YORK = ZoneInfo("America/New_York")
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
EVERY_DAY = [0, 1, 2, 3, 4, 5, 6]


def utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("when", "rule"),
    [
        ("in 2m", Once(at=NOW + timedelta(minutes=2))),
        ("in 90 minutes", Once(at=NOW + timedelta(minutes=90))),
        ("In 1 Hour", Once(at=NOW + timedelta(hours=1))),
        ("in 2d", Once(at=NOW + timedelta(days=2))),
        ("at 2026-01-01T18:30", Once(at=utc(2026, 1, 1, 17, 30))),
        ("at 2026-01-02T09:00+00:00", Once(at=utc(2026, 1, 2, 9, 0))),
        ("every 30m", Interval(every_s=1800, anchor=NOW)),
        ("every 2 hours", Interval(every_s=7200, anchor=NOW)),
        ("daily at 09:00", Weekly(days=EVERY_DAY, at=time(9, 0))),
        ("every day at 9:05", Weekly(days=EVERY_DAY, at=time(9, 5))),
        ("every mon,wed,fri at 08:30", Weekly(days=[0, 2, 4], at=time(8, 30))),
        ("every Fri, Mon at 08:30", Weekly(days=[0, 4], at=time(8, 30))),
        ("every weekdays at 07:00", Weekly(days=[0, 1, 2, 3, 4], at=time(7, 0))),
        ("every weekends at 10:00", Weekly(days=[5, 6], at=time(10, 0))),
        ("every monday at 10:00", Weekly(days=[0], at=time(10, 0))),
    ],
)
def test_parse(when: str, rule: Any) -> None:
    assert parse_rule(when, NOW, PRAGUE) == rule


@pytest.mark.parametrize(
    ("when", "problem"),
    [
        ("tomorrow", "Cannot understand 'tomorrow'"),
        ("in soon", "'soon' is not a duration"),
        ("in 2 weeks", "'2 weeks' is not a duration"),
        ("in 0m", "a minute from now at the soonest"),
        ("every 0m", "at most once a minute"),
        ("every 30s", "'30s' is not a duration"),
        ("at yesterday", "'yesterday' is not a date and time"),
        ("at 2025-12-31T10:00", "already in the past"),
        ("daily at 25:00", "'25:00' is not a time of day"),
        ("every someday at 09:00", "'someday' is not a list of weekdays"),
    ],
)
def test_parse_errors_explain_the_forms(when: str, problem: str) -> None:
    with pytest.raises(RuleError) as error:
        parse_rule(when, NOW, PRAGUE)

    assert problem in str(error.value)


def test_once_runs_once() -> None:
    rule = Once(at=utc(2026, 1, 1, 13, 0))

    assert next_run(rule, NOW, PRAGUE) == utc(2026, 1, 1, 13, 0)
    assert next_run(rule, utc(2026, 1, 1, 13, 0), PRAGUE) is None


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (NOW, utc(2026, 1, 1, 12, 30)),
        (utc(2026, 1, 1, 12, 29), utc(2026, 1, 1, 12, 30)),
        (utc(2026, 1, 1, 12, 30), utc(2026, 1, 1, 13, 0)),
        (utc(2026, 1, 1, 15, 10), utc(2026, 1, 1, 15, 30)),
        (utc(2025, 12, 31, 0, 0), NOW),
    ],
)
def test_interval_keeps_its_rhythm(now: datetime, expected: datetime) -> None:
    assert next_run(Interval(every_s=1800, anchor=NOW), now, PRAGUE) == expected


def test_interval_catches_up_after_a_long_sleep() -> None:
    rule = Interval(every_s=3600, anchor=NOW)

    later = NOW + timedelta(days=3, minutes=5)

    assert next_run(rule, later, PRAGUE) == NOW + timedelta(days=3, hours=1)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (NOW, utc(2026, 1, 2, 8, 0)),
        (utc(2026, 1, 1, 7, 59), utc(2026, 1, 1, 8, 0)),
        (utc(2026, 1, 1, 8, 0), utc(2026, 1, 2, 8, 0)),
    ],
)
def test_daily_in_local_time(now: datetime, expected: datetime) -> None:
    assert next_run(Weekly(days=EVERY_DAY, at=time(9, 0)), now, PRAGUE) == expected


def test_weekly_skips_to_the_next_listed_day() -> None:
    mondays_and_wednesdays = Weekly(days=[0, 2], at=time(9, 0))

    assert next_run(mondays_and_wednesdays, NOW, PRAGUE) == utc(2026, 1, 5, 8, 0)


def test_weekly_on_the_same_weekday_a_week_later() -> None:
    assert next_run(Weekly(days=[3], at=time(9, 0)), NOW, PRAGUE) == utc(2026, 1, 8, 8, 0)


def test_weekly_keeps_wall_time_across_summer_time() -> None:
    daily = Weekly(days=EVERY_DAY, at=time(9, 0))
    before = next_run(daily, utc(2026, 3, 28, 12, 0), PRAGUE)
    after = next_run(daily, utc(2026, 3, 29, 12, 0), PRAGUE)

    assert before == utc(2026, 3, 29, 7, 0)
    assert after == utc(2026, 3, 30, 7, 0)
    assert before is not None and before.astimezone(PRAGUE).hour == 9


def test_the_same_rule_in_another_zone() -> None:
    assert next_run(Weekly(days=EVERY_DAY, at=time(9, 0)), NOW, NEW_YORK) == utc(2026, 1, 1, 14, 0)


def test_a_rule_without_days_never_runs() -> None:
    assert next_run(Weekly(days=[], at=time(9, 0)), NOW, PRAGUE) is None


@pytest.mark.parametrize(
    ("rule", "text"),
    [
        (Once(at=utc(2026, 1, 1, 17, 30)), "once at Thu 2026-01-01 18:30"),
        (Interval(every_s=1800, anchor=NOW), "every 30m"),
        (Interval(every_s=5400, anchor=NOW), "every 90m"),
        (Interval(every_s=7200, anchor=NOW), "every 2h"),
        (Interval(every_s=86400, anchor=NOW), "every 1d"),
        (Weekly(days=EVERY_DAY, at=time(9, 0)), "every day at 09:00"),
        (Weekly(days=[0, 2], at=time(8, 30)), "every mon,wed at 08:30"),
    ],
)
def test_describe(rule: Any, text: str) -> None:
    assert describe(rule, PRAGUE) == text


def test_local_zone_follows_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "America/New_York")

    assert local_zone().key == "America/New_York"


def test_local_zone_falls_back_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Not/AZone")
    monkeypatch.setattr(rules.os, "readlink", lambda path: "/nowhere")

    assert local_zone().key == "UTC"


@pytest.fixture
async def repo() -> AsyncIterator[SchedulesRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield SchedulesRepo(db)


async def test_add_works_out_the_first_run(repo: SchedulesRepo, frozen_clock: Any) -> None:
    rule = parse_rule("in 2m", frozen_clock(), PRAGUE)

    schedule = await repo.add("  drink   water ", rule, "in 2m", "Europe/Prague")

    assert schedule.task == "drink water"
    assert schedule.next_run_at == frozen_clock() + timedelta(minutes=2)
    assert (schedule.enabled, schedule.last_run_at) == (True, None)
    assert await repo.get(schedule.id) == schedule


async def test_add_refuses_empty_tasks_and_rules_that_never_run(repo: SchedulesRepo) -> None:
    with pytest.raises(ValueError, match="needs a task"):
        await repo.add("  ", Once(at=NOW + timedelta(days=9999)), "x", "UTC")
    with pytest.raises(ValueError, match="never run"):
        await repo.add("x", Weekly(days=[], at=time(9, 0)), "x", "UTC")


async def test_due_and_mark_run(repo: SchedulesRepo, frozen_clock: Any) -> None:
    start = frozen_clock()
    once = await repo.add("once", Once(at=start + timedelta(minutes=2)), "in 2m", "UTC")
    hourly = await repo.add("hourly", Interval(every_s=3600, anchor=start), "every 1h", "UTC")

    assert await repo.due(start) == []
    soon = start + timedelta(minutes=5)
    assert [s.task for s in await repo.due(soon)] == ["once"]

    ran_once = await repo.mark_run(once.id, soon)
    assert ran_once is not None
    assert (ran_once.enabled, ran_once.next_run_at, ran_once.last_run_at) == (False, None, soon)
    later = start + timedelta(hours=1, minutes=1)
    assert [s.task for s in await repo.due(later)] == ["hourly"]
    ran_hourly = await repo.mark_run(hourly.id, later)
    assert ran_hourly is not None and ran_hourly.next_run_at == start + timedelta(hours=2)
    assert await repo.mark_run("missing", later) is None


async def test_list_hides_finished_schedules_unless_asked(
    repo: SchedulesRepo, frozen_clock: Any
) -> None:
    start = frozen_clock()
    done = await repo.add("done", Once(at=start + timedelta(minutes=1)), "in 1m", "UTC")
    await repo.add("later", Once(at=start + timedelta(hours=1)), "in 1h", "UTC")
    await repo.mark_run(done.id, start + timedelta(minutes=1))

    assert [s.task for s in await repo.list()] == ["later"]
    assert [s.task for s in await repo.list(include_disabled=True)] == ["later", "done"]
    assert await repo.delete(done.id)
    assert not await repo.delete(done.id)


async def use(
    repo: SchedulesRepo, tool: str, approver: FakeApprover | None = None, **tool_input: Any
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
    )
    call = ToolCall(id="c", name=tool, input=tool_input)
    return await builtin_registry(schedules=repo).execute(call, ctx)


async def test_create_and_list(repo: SchedulesRepo, frozen_clock: Any) -> None:
    created = await use(
        repo,
        "schedule_create",
        task="Remind me to drink water",
        when="in 2m",
        timezone="Europe/Prague",
    )
    listed = await use(repo, "schedule_list")

    (schedule,) = await repo.list()
    line = (
        f"[{schedule.id}] once at Thu 2026-01-01 13:02, next Thu 2026-01-01 13:02 "
        "(Europe/Prague): Remind me to drink water"
    )
    assert text_of(created) == f"Scheduled: {line}"
    assert created.summary == "once at Thu 2026-01-01 13:02"
    assert (text_of(listed), listed.summary) == (f"- {line}", "1 schedule")


async def test_create_uses_the_computers_zone_by_default(
    repo: SchedulesRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "America/New_York")

    await use(repo, "schedule_create", task="stretch", when="daily at 09:00")

    (schedule,) = await repo.list()
    assert schedule.timezone == "America/New_York"


@pytest.mark.parametrize(
    ("tool_input", "problem"),
    [
        ({"task": "x", "when": "sometime"}, "Nothing was scheduled: Cannot understand 'sometime'"),
        ({"task": "x", "when": "in 2m", "timezone": "Mars/Base"}, "Unknown time zone 'Mars/Base'"),
        ({"task": " ", "when": "in 2m"}, "Nothing was scheduled: a schedule needs a task"),
    ],
)
async def test_create_errors(repo: SchedulesRepo, tool_input: dict[str, Any], problem: str) -> None:
    result = await use(repo, "schedule_create", **tool_input)

    assert result.is_error
    assert problem in text_of(result)
    assert await repo.list() == []


async def test_empty_list(repo: SchedulesRepo) -> None:
    result = await use(repo, "schedule_list")

    assert (text_of(result), result.summary) == ("Nothing is scheduled.", "0 schedules")


async def test_delete_asks_with_the_task_and_removes_it(repo: SchedulesRepo) -> None:
    await use(repo, "schedule_create", task="drink water", when="every 1h")
    (schedule,) = await repo.list()
    approver = FakeApprover("allow")

    result = await use(repo, "schedule_delete", approver, id=schedule.id)

    assert [ask.label for ask in approver.asked] == ["cancel: drink water"]
    assert text_of(result) == "Cancelled: drink water"
    assert await repo.list() == []


async def test_delete_of_an_unknown_id_is_refused_without_asking(repo: SchedulesRepo) -> None:
    approver = FakeApprover("allow")

    result = await use(repo, "schedule_delete", approver, id="missing")

    assert text_of(result) == "There is no schedule with id missing."
    assert approver.asked == []
