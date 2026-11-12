"""Todo repository: a numbered todo list per session."""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.protocol.messages import TodoItem, TodoStatus
from zero.store.db import Database


class Todo(BaseModel):
    """One stored todo item."""

    id: str
    session_id: str
    number: int
    text: str
    status: TodoStatus = "pending"
    created_at: datetime
    updated_at: datetime

    def item(self) -> TodoItem:
        """The item as clients are told about it."""
        return TodoItem(number=self.number, text=self.text, status=self.status)


def _to_todo(row: aiosqlite.Row) -> Todo:
    return Todo(
        id=row["id"],
        session_id=row["session_id"],
        number=row["number"],
        text=row["text"],
        status=row["status"],
        created_at=clock.from_iso(row["created_at"]),
        updated_at=clock.from_iso(row["updated_at"]),
    )


class TodoRepo:
    """Stores todo items. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, session_id: str, text: str) -> Todo:
        """Add a pending item at the end of the session's list."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("a todo item needs some text")
        now = clock.to_iso(clock.now())
        todo_id = str(uuid.uuid4())
        await self._db.conn.execute(
            "INSERT INTO todos (id, session_id, number, text, status, created_at, updated_at) "
            "VALUES (?, ?, (SELECT COALESCE(MAX(number), 0) + 1 FROM todos WHERE session_id = ?), "
            "?, 'pending', ?, ?)",
            (todo_id, session_id, session_id, text, now, now),
        )
        await self._db.conn.commit()
        async with self._db.conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return _to_todo(row)

    async def set_status(self, session_id: str, number: int, status: TodoStatus) -> Todo | None:
        """Change an item's status. None if the session has no item with that number."""
        cursor = await self._db.conn.execute(
            "UPDATE todos SET status = ?, updated_at = ? WHERE session_id = ? AND number = ?",
            (status, clock.to_iso(clock.now()), session_id, number),
        )
        await self._db.conn.commit()
        if not cursor.rowcount:
            return None
        async with self._db.conn.execute(
            "SELECT * FROM todos WHERE session_id = ? AND number = ?", (session_id, number)
        ) as found:
            row = await found.fetchone()
        return _to_todo(row) if row else None

    async def list(self, session_id: str) -> list[Todo]:
        """The session's items in order."""
        async with self._db.conn.execute(
            "SELECT * FROM todos WHERE session_id = ? ORDER BY number", (session_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_todo(row) for row in rows]
