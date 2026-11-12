"""Messages repository: the stored history of a session."""

import json
import uuid
from datetime import datetime
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field, TypeAdapter

from zero import clock
from zero.protocol.messages import ContentPart
from zero.store.db import Database

Role = Literal["user", "assistant", "tool"]
Status = Literal["ok", "cancelled", "error"]
Source = Literal["desktop", "phone", "scheduler", "agent"]

_content_adapter: TypeAdapter[list[ContentPart]] = TypeAdapter(list[ContentPart])


class StoredMessage(BaseModel):
    """One message of a session as it is kept in the database."""

    id: str
    session_id: str
    turn_id: str | None = None
    role: Role
    content: list[ContentPart]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    status: Status | None = None
    source: Source | None = None
    created_at: datetime


def _to_message(row: aiosqlite.Row) -> StoredMessage:
    return StoredMessage(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        role=row["role"],
        content=_content_adapter.validate_json(row["content_json"]),
        tool_calls=json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else [],
        tool_call_id=row["tool_call_id"],
        is_error=bool(row["is_error"]),
        status=row["status"],
        source=row["source"],
        created_at=clock.from_iso(row["created_at"]),
    )


class MessagesRepo:
    """Appends messages and reads them back page by page."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def append(
        self,
        session_id: str,
        role: Role,
        content: list[ContentPart],
        *,
        turn_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
        status: Status | None = None,
        source: Source | None = None,
    ) -> StoredMessage:
        """Add a message to the end of a session and return it.

        Raises ``sqlite3.IntegrityError`` if the session does not exist.
        """
        message = StoredMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            turn_id=turn_id,
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id,
            is_error=is_error,
            status=status,
            source=source,
            created_at=clock.now(),
        )
        await self._db.conn.execute(
            "INSERT INTO messages (id, session_id, turn_id, role, content_json, tool_calls_json, "
            "tool_call_id, is_error, status, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.session_id,
                message.turn_id,
                message.role,
                _content_adapter.dump_json(message.content).decode("utf-8"),
                json.dumps(message.tool_calls, ensure_ascii=False) if message.tool_calls else None,
                message.tool_call_id,
                int(message.is_error),
                message.status,
                message.source,
                clock.to_iso(message.created_at),
            ),
        )
        await self._db.conn.commit()
        return message

    async def file_paths(self, session_id: str) -> list[str]:
        """Paths of every file part in the messages of a session."""
        async with self._db.conn.execute(
            "SELECT json_extract(part.value, '$.path') AS path "
            "FROM messages, json_each(messages.content_json) AS part "
            "WHERE messages.session_id = ? AND json_extract(part.value, '$.type') = 'file'",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["path"] for row in rows if row["path"]]

    async def list(
        self, session_id: str, before: str | None = None, limit: int = 50
    ) -> list[StoredMessage]:
        """Return up to ``limit`` latest messages of a session, oldest first.

        ``before`` is a message id: only messages older than it are returned,
        which is how a client pages back through history. An unknown id gives
        an empty list.
        """
        sql = "SELECT * FROM messages WHERE session_id = ?"
        params: list[Any] = [session_id]
        if before is not None:
            sql += (
                " AND (created_at, rowid) < "
                "(SELECT created_at, rowid FROM messages WHERE id = ? AND session_id = ?)"
            )
            params += [before, session_id]
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(limit, 0))

        async with self._db.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_to_message(row) for row in reversed(list(rows))]
