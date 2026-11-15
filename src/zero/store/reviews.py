"""How far each conversation was gone over (``zero.core.review``)."""

from zero import clock
from zero.store.db import Database


class ReviewsRepo:
    """The last message of each conversation that a review covered."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def until(self, session_id: str) -> str | None:
        """The id of the last message the last review covered, or None."""
        async with self._db.conn.execute(
            "SELECT until_message_id FROM session_reviews WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row["until_message_id"] if row else None

    async def mark(self, session_id: str, until_message_id: str) -> None:
        await self._db.conn.execute(
            "INSERT INTO session_reviews (session_id, until_message_id, reviewed_at) "
            "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
            "until_message_id = excluded.until_message_id, reviewed_at = excluded.reviewed_at",
            (session_id, until_message_id, clock.to_iso(clock.now())),
        )
        await self._db.conn.commit()

    async def reset(self, session_id: str, until_message_id: str | None) -> None:
        """Move the review mark back to ``until_message_id``, or drop it with None."""
        if until_message_id is None:
            await self._db.conn.execute(
                "DELETE FROM session_reviews WHERE session_id = ?", (session_id,)
            )
            await self._db.conn.commit()
            return
        await self.mark(session_id, until_message_id)
