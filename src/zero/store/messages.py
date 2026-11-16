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
Source = Literal["desktop", "phone", "scheduler", "agent", "recovery"]

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
    command: str | None = None
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
        command=row["command"],
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
        command: str | None = None,
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
            command=command,
            created_at=clock.now(),
        )
        await self._db.conn.execute(
            "INSERT INTO messages (id, session_id, turn_id, role, content_json, tool_calls_json, "
            "tool_call_id, is_error, status, source, command, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                message.command,
                clock.to_iso(message.created_at),
            ),
        )
        await self._db.conn.commit()
        return message

    async def search_text(
        self, query: str, exclude_session: str, limit: int = 10
    ) -> list[tuple[StoredMessage, str]]:
        """User and assistant messages of other sessions whose text has any word of
        ``query``, newest first, each with the text that matched."""
        words = [word.lower() for word in query.split() if word.strip()]
        if not words:
            return []
        match = " OR ".join("lower(json_extract(part.value, '$.text')) LIKE ?" for _ in words)
        async with self._db.conn.execute(
            "SELECT messages.*, json_extract(part.value, '$.text') AS matched "
            "FROM messages, json_each(messages.content_json) AS part "
            "WHERE messages.role IN ('user', 'assistant') AND messages.session_id != ? "
            f"AND json_extract(part.value, '$.type') = 'text' AND ({match}) "
            "ORDER BY messages.created_at DESC, messages.rowid DESC LIMIT ?",
            (exclude_session, *(f"%{word}%" for word in words), max(limit, 0)),
        ) as cursor:
            rows = await cursor.fetchall()
        return [(_to_message(row), row["matched"]) for row in rows]

    async def get(self, message_id: str) -> StoredMessage | None:
        async with self._db.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _to_message(row) if row else None

    async def delete_from(self, session_id: str, message_id: str) -> list[StoredMessage]:
        """Delete this message of the session and every later one; return what went."""
        async with self._db.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND rowid >= "
            "(SELECT rowid FROM messages WHERE id = ?) ORDER BY rowid",
            (session_id, message_id),
        ) as cursor:
            removed = [_to_message(row) for row in await cursor.fetchall()]
        if removed:
            marks = ", ".join("?" for _ in removed)
            await self._db.conn.execute(
                f"DELETE FROM messages WHERE id IN ({marks})", tuple(m.id for m in removed)
            )
            await self._db.conn.commit()
        return removed

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
