"""Tool runs repository: a record of every visible tool call, for the history."""

import json
import uuid
from collections.abc import Collection

import aiosqlite

from zero import clock
from zero.protocol.messages import ToolRun
from zero.store.db import Database


def _to_run(row: aiosqlite.Row) -> ToolRun:
    return ToolRun(
        call_id=row["call_id"],
        turn_id=row["turn_id"],
        tool=row["tool"],
        label=row["label"] or "",
        input=json.loads(row["input_json"]) if row["input_json"] else {},
        status=row["status"],
        summary=row["summary"] or "",
        output_preview=row["output_preview"] or "",
        duration_ms=row["duration_ms"] or 0,
        created_at=clock.from_iso(row["created_at"]),
    )


class ToolRunsRepo:
    """Stores tool runs and reads them back by turn. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, session_id: str, run: ToolRun) -> None:
        """Record a finished tool call. Call ids may repeat; every record is kept."""
        await self._db.conn.execute(
            "INSERT INTO tool_runs (id, call_id, session_id, turn_id, tool, label, input_json, "
            "status, summary, output_preview, duration_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                run.call_id,
                session_id,
                run.turn_id,
                run.tool,
                run.label,
                json.dumps(run.input, ensure_ascii=False),
                run.status,
                run.summary,
                run.output_preview,
                run.duration_ms,
                clock.to_iso(run.created_at),
            ),
        )
        await self._db.conn.commit()

    async def for_turns(self, session_id: str, turn_ids: Collection[str]) -> list[ToolRun]:
        """The runs of the given turns of a session, oldest first."""
        if not turn_ids:
            return []
        marks = ", ".join("?" for _ in turn_ids)
        async with self._db.conn.execute(
            f"SELECT * FROM tool_runs WHERE session_id = ? AND turn_id IN ({marks}) "
            "ORDER BY created_at, rowid",
            (session_id, *turn_ids),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_run(row) for row in rows]

    async def delete_turns(self, session_id: str, turn_ids: Collection[str]) -> None:
        """Forget the tool runs of these turns of the session."""
        if not turn_ids:
            return
        marks = ", ".join("?" for _ in turn_ids)
        await self._db.conn.execute(
            f"DELETE FROM tool_runs WHERE session_id = ? AND turn_id IN ({marks})",
            (session_id, *turn_ids),
        )
        await self._db.conn.commit()
