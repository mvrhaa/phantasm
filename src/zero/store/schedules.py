"""Schedules repository: tasks Zero runs on its own, and when they are due."""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.scheduler.rules import Rule, next_run, rule_adapter, zone_named
from zero.store.db import Database


class Schedule(BaseModel):
    """A stored schedule."""

    id: str
    task: str
    rule: Rule
    rule_text: str
    timezone: str
    next_run_at: datetime | None
    last_run_at: datetime | None
    enabled: bool
    created_at: datetime


def _time(value: str | None) -> datetime | None:
    return clock.from_iso(value) if value else None


def _to_schedule(row: aiosqlite.Row) -> Schedule:
    return Schedule(
        id=row["id"],
        task=row["task"],
        rule=rule_adapter.validate_json(row["rule_json"]),
        rule_text=row["rule_text"],
        timezone=row["timezone"],
        next_run_at=_time(row["next_run_at"]),
        last_run_at=_time(row["last_run_at"]),
        enabled=bool(row["enabled"]),
        created_at=clock.from_iso(row["created_at"]),
    )


class SchedulesRepo:
    """Stores schedules. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, task: str, rule: Rule, rule_text: str, timezone: str) -> Schedule:
        """Store a schedule with its first run worked out.

        Raises ``ValueError`` if the task is empty or the rule would never fire.
        """
        task = " ".join(task.split())
        if not task:
            raise ValueError("a schedule needs a task")
        now = clock.now()
        first = next_run(rule, now, zone_named(timezone))
        if first is None:
            raise ValueError("this schedule would never run")
        schedule_id = str(uuid.uuid4())
        await self._db.conn.execute(
            "INSERT INTO schedules (id, task, rule_json, rule_text, timezone, next_run_at, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                schedule_id,
                task,
                rule_adapter.dump_json(rule).decode(),
                rule_text,
                timezone,
                clock.to_iso(first),
                clock.to_iso(now),
            ),
        )
        await self._db.conn.commit()
        schedule = await self.get(schedule_id)
        assert schedule is not None
        return schedule

    async def get(self, schedule_id: str) -> Schedule | None:
        """The schedule with this id, or None."""
        async with self._db.conn.execute(
            "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _to_schedule(row) if row else None

    async def due(self, now: datetime) -> list[Schedule]:
        """Enabled schedules whose next run has come, the oldest first."""
        async with self._db.conn.execute(
            "SELECT * FROM schedules WHERE enabled = 1 AND next_run_at IS NOT NULL "
            "AND next_run_at <= ? ORDER BY next_run_at",
            (clock.to_iso(now),),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_schedule(row) for row in rows]

    async def next_due(self) -> datetime | None:
        """When the earliest enabled schedule runs next; None if nothing is planned."""
        async with self._db.conn.execute(
            "SELECT MIN(next_run_at) FROM schedules WHERE enabled = 1 AND next_run_at IS NOT NULL"
        ) as cursor:
            row = await cursor.fetchone()
        return _time(row[0]) if row else None

    async def mark_run(self, schedule_id: str, ran_at: datetime) -> Schedule | None:
        """Record a run and work out the next one; a schedule with no next run is turned off."""
        schedule = await self.get(schedule_id)
        if schedule is None:
            return None
        following = next_run(schedule.rule, ran_at, zone_named(schedule.timezone))
        await self._db.conn.execute(
            "UPDATE schedules SET last_run_at = ?, next_run_at = ?, enabled = ? WHERE id = ?",
            (
                clock.to_iso(ran_at),
                clock.to_iso(following) if following else None,
                int(following is not None),
                schedule_id,
            ),
        )
        await self._db.conn.commit()
        return await self.get(schedule_id)

    async def delete(self, schedule_id: str) -> bool:
        """Remove a schedule. False if there was none with this id."""
        cursor = await self._db.conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        await self._db.conn.commit()
        return cursor.rowcount > 0

    async def list(self, include_disabled: bool = False) -> list[Schedule]:
        """Schedules by their next run; finished one-off ones only on request."""
        where = "" if include_disabled else "WHERE enabled = 1"
        async with self._db.conn.execute(
            f"SELECT * FROM schedules {where} ORDER BY next_run_at IS NULL, next_run_at, created_at"
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_schedule(row) for row in rows]
