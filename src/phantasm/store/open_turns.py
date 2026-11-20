"""Turns in progress, kept on disk so that turns a crash cut off can go on after a restart."""

from dataclasses import dataclass
from datetime import datetime

from phantasm import clock
from phantasm.store.db import Database


@dataclass(frozen=True)
class OpenTurn:
    turn_id: str
    session_id: str
    source: str
    agent_id: str | None
    started_at: datetime


class OpenTurnsRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, turn_id: str, session_id: str, source: str, agent_id: str | None) -> None:
        await self._db.conn.execute(
            "INSERT OR REPLACE INTO open_turns (turn_id, session_id, source, agent_id, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (turn_id, session_id, source, agent_id, clock.to_iso(clock.now())),
        )
        await self._db.conn.commit()

    async def remove(self, turn_id: str) -> None:
        await self._db.conn.execute("DELETE FROM open_turns WHERE turn_id = ?", (turn_id,))
        await self._db.conn.commit()

    async def take_all(self) -> list[OpenTurn]:
        """Every open turn, oldest first; the table is empty after."""
        async with self._db.conn.execute(
            "SELECT * FROM open_turns ORDER BY started_at, rowid"
        ) as cursor:
            rows = await cursor.fetchall()
        await self._db.conn.execute("DELETE FROM open_turns")
        await self._db.conn.commit()
        return [
            OpenTurn(
                row["turn_id"],
                row["session_id"],
                row["source"],
                row["agent_id"],
                clock.from_iso(row["started_at"]),
            )
            for row in rows
        ]
