"""The journal: short dated notes about what happened in conversations.

A note is written for a conversation after it goes quiet (``zero.core.review``).
Only the newest ``JOURNAL_NOTES`` are kept. The newest notes of other
conversations go into the system prompt, so one conversation knows what
happened in another; ``journal_search`` finds the rest. A note outlives its
conversation.
"""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.store.db import Database

JOURNAL_NOTES = 50


class JournalNote(BaseModel):
    """What happened in one conversation, in a sentence or three."""

    id: str
    session_id: str
    text: str
    created_at: datetime


def note_time(moment: datetime) -> str:
    """When a note was written, in the computer's time zone: ``2026-11-16 09:12 CET``."""
    local = moment.astimezone()
    return f"{local:%Y-%m-%d %H:%M} {local.tzname()}"


def _to_note(row: aiosqlite.Row) -> JournalNote:
    return JournalNote(
        id=row["id"],
        session_id=row["session_id"],
        text=row["text"],
        created_at=clock.from_iso(row["created_at"]),
    )


class JournalRepo:
    """Stores journal notes. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, session_id: str, text: str) -> JournalNote:
        """Write a note; the oldest ones go once there are more than ``JOURNAL_NOTES``."""
        note = JournalNote(
            id=f"j-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            text=text.strip(),
            created_at=clock.now(),
        )
        await self._db.conn.execute(
            "INSERT INTO journal (id, session_id, text, created_at) VALUES (?, ?, ?, ?)",
            (note.id, session_id, note.text, clock.to_iso(note.created_at)),
        )
        await self._db.conn.execute(
            "DELETE FROM journal WHERE id NOT IN "
            "(SELECT id FROM journal ORDER BY created_at DESC, rowid DESC LIMIT ?)",
            (JOURNAL_NOTES,),
        )
        await self._db.conn.commit()
        return note

    async def newest(self, exclude_session: str, max_chars: int) -> list[JournalNote]:
        """The newest notes of other conversations that fit ``max_chars``, oldest first."""
        async with self._db.conn.execute(
            "SELECT * FROM journal WHERE session_id != ? ORDER BY created_at DESC, rowid DESC",
            (exclude_session,),
        ) as cursor:
            rows = await cursor.fetchall()
        chosen: list[JournalNote] = []
        used = 0
        for row in rows:
            note = _to_note(row)
            if used + len(note.text) > max_chars:
                break
            used += len(note.text)
            chosen.append(note)
        return list(reversed(chosen))

    async def search(self, query: str, limit: int = 10) -> list[JournalNote]:
        """Notes with any word of ``query`` in them, newest first."""
        words = [word.lower() for word in query.split() if word.strip()]
        if not words:
            return []
        where = " OR ".join("lower(text) LIKE ?" for _ in words)
        async with self._db.conn.execute(
            f"SELECT * FROM journal WHERE {where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (*(f"%{word}%" for word in words), max(limit, 0)),
        ) as cursor:
            return [_to_note(row) for row in await cursor.fetchall()]

    async def list(self) -> list[JournalNote]:
        """Every kept note, oldest first."""
        async with self._db.conn.execute(
            "SELECT * FROM journal ORDER BY created_at, rowid"
        ) as cursor:
            return [_to_note(row) for row in await cursor.fetchall()]
