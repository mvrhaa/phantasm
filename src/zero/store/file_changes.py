"""What ``write_file`` and ``edit_file`` changed, so going back to a message can undo it.

Each change keeps the file as it was before (``None`` when it did not exist)
and a hash of what was written, so a file changed again outside Zero since is
never overwritten. Files over ``MAX_BEFORE_BYTES`` are not kept.
"""

import hashlib
import uuid
from collections.abc import Collection
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.store.db import Database

MAX_BEFORE_BYTES = 5 * 1024 * 1024
KEEP_DAYS = 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileChange(BaseModel):
    """One change of one file."""

    id: str
    session_id: str
    turn_id: str | None
    path: str
    before: bytes | None
    after_sha256: str
    created_at: datetime


def _to_change(row: aiosqlite.Row) -> FileChange:
    return FileChange(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        path=row["path"],
        before=row["before"],
        after_sha256=row["after_sha256"],
        created_at=clock.from_iso(row["created_at"]),
    )


class FileChangesRepo:
    """Stores file changes. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        session_id: str,
        turn_id: str | None,
        path: Path,
        before: bytes | None,
        after: bytes,
    ) -> None:
        """Remember that ``path`` went from ``before`` to ``after``; big files are skipped."""
        if before is not None and len(before) > MAX_BEFORE_BYTES:
            return
        await self._db.conn.execute(
            "INSERT INTO file_changes (id, session_id, turn_id, path, before, after_sha256, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                turn_id,
                str(path),
                before,
                sha256(after),
                clock.to_iso(clock.now()),
            ),
        )
        await self._db.conn.commit()

    async def since(self, session_ids: Collection[str], moment: datetime) -> list[FileChange]:
        """Changes made in these sessions at ``moment`` or later, oldest first."""
        if not session_ids:
            return []
        marks = ", ".join("?" for _ in session_ids)
        async with self._db.conn.execute(
            f"SELECT * FROM file_changes WHERE session_id IN ({marks}) AND created_at >= ? "
            "ORDER BY created_at, rowid",
            (*session_ids, clock.to_iso(moment)),
        ) as cursor:
            return [_to_change(row) for row in await cursor.fetchall()]

    async def forget(self, change_ids: Collection[str]) -> None:
        if not change_ids:
            return
        marks = ", ".join("?" for _ in change_ids)
        await self._db.conn.execute(
            f"DELETE FROM file_changes WHERE id IN ({marks})", tuple(change_ids)
        )
        await self._db.conn.commit()

    async def prune(self, keep_days: int = KEEP_DAYS) -> None:
        """Forget changes older than ``keep_days``: chats older than a week are deleted."""
        oldest = clock.now() - timedelta(days=keep_days)
        await self._db.conn.execute(
            "DELETE FROM file_changes WHERE created_at < ?", (clock.to_iso(oldest),)
        )
        await self._db.conn.commit()
