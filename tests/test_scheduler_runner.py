"""Tests of the scheduler runner and of scheduled turns in the application."""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.turns import until
from zero.config import Config
from zero.paths import db_path
from zero.providers.base import Message
from zero.providers.fake import FakeProvider
from zero.scheduler.rules import Interval, Once
from zero.scheduler.runner import SchedulerRunner
from zero.server.app import create_app
from zero.server.state import get_state
from zero.server.turns import start_scheduled_turn
from zero.store.db import Database
from zero.store.schedules import Schedule, SchedulesRepo


@pytest.fixture
async def repo() -> AsyncIterator[SchedulesRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield SchedulesRepo(db)


async def test_tick_starts_what_is_due_after_recording_the_run(
    repo: SchedulesRepo, frozen_clock: Any
) -> None:
    start = frozen_clock()
    soon = await repo.add("soon", Once(at=start + timedelta(minutes=2)), "in 2m", "UTC")
    await repo.add("later", Once(at=start + timedelta(hours=1)), "in 1h", "UTC")
    seen: list[Schedule | None] = []

    async def record(schedule: Schedule) -> None:
        seen.append(await repo.get(schedule.id))

    runner = SchedulerRunner(repo, record)
    assert await runner.tick() == []

    frozen_clock.advance(180)
    started = await runner.tick()

    assert [s.task for s in started] == ["soon"]
    (stored,) = seen
    assert stored is not None
    assert (stored.id, stored.last_run_at, stored.enabled) == (soon.id, frozen_clock(), False)
    assert await runner.tick() == []


async def test_a_long_pause_runs_a_schedule_once(repo: SchedulesRepo, frozen_clock: Any) -> None:
    await repo.add("hourly", Interval(every_s=3600, anchor=frozen_clock()), "every 1h", "UTC")
    runs: list[str] = []

    async def record(schedule: Schedule) -> None:
        runs.append(schedule.task)

    runner = SchedulerRunner(repo, record)
    frozen_clock.advance(10 * 3600 + 60)

    await runner.tick()
    await runner.tick()

    assert runs == ["hourly"]


async def test_one_failing_task_does_not_stop_the_others(
    repo: SchedulesRepo, frozen_clock: Any, caplog: pytest.LogCaptureFixture
) -> None:
    start = frozen_clock()
    for task in ("first", "second"):
        await repo.add(task, Once(at=start + timedelta(minutes=1)), "in 1m", "UTC")
    runs: list[str] = []

    async def flaky(schedule: Schedule) -> None:
        if schedule.task == "first":
            raise RuntimeError("boom")
        runs.append(schedule.task)

    frozen_clock.advance(120)
    await SchedulerRunner(repo, flaky).tick()

    assert runs == ["second"]
    assert "could not start" in caplog.text


async def test_the_loop_ticks_until_stopped(repo: SchedulesRepo, frozen_clock: Any) -> None:
    await repo.add("hourly", Interval(every_s=3600, anchor=frozen_clock()), "every 1h", "UTC")
    runs: list[str] = []

    async def record(schedule: Schedule) -> None:
        runs.append(schedule.task)

    runner = SchedulerRunner(repo, record, tick_s=0.01)
    runner.start()
    assert runner.running
    frozen_clock.advance(3600)
    await until(lambda: runs == ["hourly"])
    await runner.stop()

    assert not runner.running
    frozen_clock.advance(3600)
    await asyncio.sleep(0.05)
    assert runs == ["hourly"]


def system_text(messages: list[Message]) -> str:
    return messages[0].content[0].text  # type: ignore[union-attr]


def read_until(ws: Any, event_type: str) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] != event_type:
        events.append(ws.receive_json())
    return events


def test_a_due_schedule_runs_a_turn_in_its_own_session(frozen_clock: Any) -> None:
    provider = FakeProvider()
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            rule = Once(at=frozen_clock() + timedelta(minutes=2))
            schedule = client.portal.call(state.schedules.add, "drink water", rule, "in 2m", "UTC")

            frozen_clock.advance(150)
            client.portal.call(state.scheduler.tick)
            events = read_until(ws, "turn.finished")
            again = client.portal.call(state.scheduler.tick)
            sessions = client.portal.call(state.sessions.list)

    created, started, *_, finished = events
    assert created["type"] == "session.created"
    assert created["session"] == {
        "id": f"scheduled-{schedule.id}",
        "title": "Scheduled: drink water",
        "updated_at": created["session"]["updated_at"],
        "kind": "scheduled",
    }
    assert (started["type"], started["source"]) == ("turn.started", "scheduler")
    assert finished["text"] == "echo: drink water"
    assert "started by a schedule" in system_text(provider.calls[0].messages)
    assert again == []
    assert [s.id for s in sessions] == [f"scheduled-{schedule.id}"]


def test_a_repeating_schedule_keeps_its_session(frozen_clock: Any) -> None:
    provider = FakeProvider()
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        rule = Interval(every_s=3600, anchor=frozen_clock())
        schedule = client.portal.call(state.schedules.add, "stretch", rule, "every 1h", "UTC")
        for _ in range(2):
            frozen_clock.advance(3600)
            client.portal.call(state.scheduler.tick)
            client.portal.call(state.turns.wait_idle)
        history = client.portal.call(state.messages.list, f"scheduled-{schedule.id}")

    assert [(m.role, m.source) for m in history] == [
        ("user", "scheduler"),
        ("assistant", "scheduler"),
        ("user", "scheduler"),
        ("assistant", "scheduler"),
    ]


def test_a_run_is_skipped_while_the_previous_one_is_still_going(
    frozen_clock: Any, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeProvider(infinite=True)
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        rule = Interval(every_s=3600, anchor=frozen_clock())
        schedule = client.portal.call(state.schedules.add, "sing", rule, "every 1h", "UTC")
        frozen_clock.advance(3600)
        client.portal.call(state.scheduler.tick)
        frozen_clock.advance(3600)
        client.portal.call(state.scheduler.tick)

        assert len(state.turns.active) == 1
    assert f"Schedule {schedule.id} skipped: its previous turn is still running" in caplog.text


def test_without_a_provider_nothing_starts(
    frozen_clock: Any, caplog: pytest.LogCaptureFixture
) -> None:
    config = Config()
    config.provider.name = "nope"
    with TestClient(create_app(config)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        rule = Once(at=frozen_clock() + timedelta(minutes=1))
        schedule = client.portal.call(state.schedules.add, "x", rule, "in 1m", "UTC")

        client.portal.call(start_scheduled_turn, state, schedule)

        assert state.turns.active == []
        assert client.portal.call(state.sessions.list) == []
    assert "skipped: no usable provider" in caplog.text


def test_the_app_starts_and_stops_the_scheduler() -> None:
    with TestClient(create_app(Config())) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        assert state.scheduler.running
        scheduler = state.scheduler

    assert not scheduler.running
