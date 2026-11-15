"""Summaries repository: the running summary of the start of a long session."""

from datetime import datetime

from pydantic import BaseModel

from zero import clock
from zero.store.db import Database


class Summary(BaseModel):
    """What a session's dropped messages said, up to ``until_message_id``."""

    session_id: str
    text: str
    until_message_id: str
    updated_at: datetime


class SummariesRepo:
    """Keeps one summary per session. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, session_id: str) -> Summary | None:
        """The session's summary, or None if it has none yet."""
        async with self._db.conn.execute(
            "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Summary(
            session_id=row["session_id"],
            text=row["text"],
            until_message_id=row["until_message_id"],
            updated_at=clock.from_iso(row["updated_at"]),
        )

    async def save(self, session_id: str, text: str, until_message_id: str) -> Summary:
        """Store the session's summary, replacing the one before."""
        now = clock.now()
        await self._db.conn.execute(
            "INSERT INTO session_summaries (session_id, text, until_message_id, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (session_id) DO UPDATE SET "
            "text = excluded.text, until_message_id = excluded.until_message_id, "
            "updated_at = excluded.updated_at",
            (session_id, text, until_message_id, clock.to_iso(now)),
        )
        await self._db.conn.commit()
        return Summary(
            session_id=session_id, text=text, until_message_id=until_message_id, updated_at=now
        )

    async def delete(self, session_id: str) -> None:
        """Forget the session's summary; it is made again when it is needed."""
        await self._db.conn.execute(
            "DELETE FROM session_summaries WHERE session_id = ?", (session_id,)
        )
        await self._db.conn.commit()
