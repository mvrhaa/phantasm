"""Session memory: what the model keeps for one conversation only.

An instruction such as "answer in English here" or a decision for the task at
hand belongs to its conversation, not to the user's lasting memory. Entries
are listed whole in that session's system prompt, so they survive compaction,
and go away with the session.
"""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.store.db import Database


class SessionMemory(BaseModel):
    """One thing kept for one session."""

    id: str
    session_id: str
    text: str
    created_at: datetime


def _to_entry(row: aiosqlite.Row) -> SessionMemory:
    return SessionMemory(
        id=row["id"],
        session_id=row["session_id"],
        text=row["text"],
        created_at=clock.from_iso(row["created_at"]),
    )


class SessionMemoryRepo:
    """Stores session memory. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, session_id: str, text: str) -> SessionMemory:
        """Keep ``text`` for the session; the same text again keeps the entry there is."""
        text = text.strip()
        if not text:
            raise ValueError("a memory needs some text")
        await self._db.conn.execute(
            "INSERT OR IGNORE INTO session_memories (id, session_id, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (f"sm-{uuid.uuid4().hex[:12]}", session_id, text, clock.to_iso(clock.now())),
        )
        await self._db.conn.commit()
        async with self._db.conn.execute(
            "SELECT * FROM session_memories WHERE session_id = ? AND text = ?",
            (session_id, text),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return _to_entry(row)

    async def list(self, session_id: str) -> list[SessionMemory]:
        """The session's entries, oldest first."""
        async with self._db.conn.execute(
            "SELECT * FROM session_memories WHERE session_id = ? ORDER BY created_at, rowid",
            (session_id,),
        ) as cursor:
            return [_to_entry(row) for row in await cursor.fetchall()]

    async def get(self, entry_id: str) -> SessionMemory | None:
        async with self._db.conn.execute(
            "SELECT * FROM session_memories WHERE id = ?", (entry_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _to_entry(row) if row else None

    async def delete(self, entry_id: str) -> bool:
        """Forget an entry. False if there was none with this id."""
        cursor = await self._db.conn.execute(
            "DELETE FROM session_memories WHERE id = ?", (entry_id,)
        )
        await self._db.conn.commit()
        return cursor.rowcount > 0
