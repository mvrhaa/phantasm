"""Sessions repository."""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.store.db import Database


class Session(BaseModel):
    """A conversation."""

    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    archived: bool = False


def _to_session(row: aiosqlite.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        created_at=clock.from_iso(row["created_at"]),
        updated_at=clock.from_iso(row["updated_at"]),
        archived=bool(row["archived"]),
    )


class SessionsRepo:
    """Stores and retrieves sessions. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, title: str | None = None, *, session_id: str | None = None) -> Session:
        """Create a session, optionally with a title, and return it.

        The id is a new UUID unless ``session_id`` names one, as the scheduler
        does for the session of a schedule.
        """
        now = clock.now()
        session = Session(
            id=session_id or str(uuid.uuid4()), title=title, created_at=now, updated_at=now
        )
        await self._db.conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session.id, session.title, clock.to_iso(now), clock.to_iso(now)),
        )
        await self._db.conn.commit()
        return session

    async def get(self, session_id: str) -> Session | None:
        """Return the session, or None if there is no such id."""
        async with self._db.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _to_session(row) if row else None

    async def rename(self, session_id: str, title: str) -> Session | None:
        """Change the title and mark the session as updated. None if it does not exist."""
        cursor = await self._db.conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, clock.to_iso(clock.now()), session_id),
        )
        await self._db.conn.commit()
        return await self.get(session_id) if cursor.rowcount else None

    async def set_title_if_missing(self, session_id: str, title: str) -> Session | None:
        """Give an untitled session a title, leaving ``updated_at`` as it is.

        Returns None if the session already has a title or does not exist, so
        an automatic title never replaces one the user chose.
        """
        cursor = await self._db.conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL", (title, session_id)
        )
        await self._db.conn.commit()
        return await self.get(session_id) if cursor.rowcount else None

    async def touch(self, session_id: str, updated_at: datetime | None = None) -> Session | None:
        """Set ``updated_at`` (now by default). None if the session does not exist."""
        cursor = await self._db.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (clock.to_iso(updated_at or clock.now()), session_id),
        )
        await self._db.conn.commit()
        return await self.get(session_id) if cursor.rowcount else None

    async def delete(self, session_id: str) -> bool:
        """Delete the session with its messages. False if it did not exist."""
        cursor = await self._db.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._db.conn.commit()
        return cursor.rowcount > 0

    # Keep this method last: inside the class body its name shadows the built-in
    # ``list``, which would break ``list[...]`` annotations written after it.
    async def list(self, include_archived: bool = False) -> list[Session]:
        """Return sessions, most recently updated first."""
        where = "" if include_archived else "WHERE archived = 0"
        async with self._db.conn.execute(
            f"SELECT * FROM sessions {where} ORDER BY updated_at DESC, created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_session(row) for row in rows]
