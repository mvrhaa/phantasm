"""Memory repository: facts about the user, found again with full-text search (SQLite FTS5)."""

import re
import uuid
from collections.abc import Collection, Sequence
from datetime import datetime

import aiosqlite
from pydantic import BaseModel, Field

from phantasm import clock
from phantasm.store.db import Database

_WORD = re.compile(r"\w+", re.UNICODE)
MIN_PREFIX_CHARS = 3
STEM_FROM_CHARS = 5
STEM_DROP = 2
MIN_STEM_CHARS = 4


class Memory(BaseModel):
    """One remembered fact."""

    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    uses: int = 0


def _clean_tags(tags: Sequence[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    for tag in tags or ():
        tag = " ".join(tag.replace(",", " ").split())
        if tag:
            seen.setdefault(tag, None)
    return list(seen)


def _to_memory(row: aiosqlite.Row) -> Memory:
    return Memory(
        id=row["id"],
        text=row["text"],
        tags=[tag for tag in (row["tags"] or "").split(",") if tag],
        created_at=clock.from_iso(row["created_at"]),
        updated_at=clock.from_iso(row["updated_at"]),
        uses=row["uses"],
    )


def _stem(word: str) -> str:
    """A rough stem: long words lose their last letters, so other endings match too.

    Russian changes word endings all the time ("Прага", "Праге"); a real
    stemmer would be a new dependency, and FTS ranking keeps the extra matches
    in check.
    """
    if len(word) < STEM_FROM_CHARS:
        return word
    return word[: max(MIN_STEM_CHARS, len(word) - STEM_DROP)]


def fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: its words joined with OR.

    Every word is quoted, so FTS syntax in the text (quotes, AND, NEAR, ``*``,
    ``:``, parentheses) is taken literally. Words of three letters or more
    match as a prefix of their rough stem, which copes with word endings
    ("Прага" finds "Праге"). An empty string means nothing to search for.
    """
    terms = []
    for word in _WORD.findall(query):
        if len(word) >= MIN_PREFIX_CHARS:
            terms.append(f'"{_stem(word)}"*')
        else:
            terms.append(f'"{word}"')
    return " OR ".join(terms)


class MemoryRepo:
    """Stores memories and finds them. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, text: str, tags: Sequence[str] | None = None) -> Memory:
        """Remember ``text``. The same text again updates the existing memory instead."""
        text = text.strip()
        if not text:
            raise ValueError("a memory needs some text")
        now = clock.to_iso(clock.now())
        tags_text = ",".join(_clean_tags(tags)) or None
        async with self._db.conn.execute(
            "SELECT * FROM memories WHERE text = ?", (text,)
        ) as cursor:
            existing = await cursor.fetchone()
        if existing is not None:
            await self._db.conn.execute(
                "UPDATE memories SET tags = COALESCE(?, tags), updated_at = ? WHERE id = ?",
                (tags_text, now, existing["id"]),
            )
            memory_id = existing["id"]
        else:
            memory_id = str(uuid.uuid4())
            await self._db.conn.execute(
                "INSERT INTO memories (id, text, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, text, tags_text, now, now),
            )
        await self._db.conn.commit()
        memory = await self.get(memory_id)
        assert memory is not None
        return memory

    async def get(self, memory_id: str) -> Memory | None:
        """The memory with this id, or None."""
        async with self._db.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return _to_memory(row) if row else None

    async def search(self, query: str, limit: int = 10) -> list[Memory]:
        """Memories matching any word of ``query``, the best matches first."""
        match = fts_query(query)
        if not match:
            return []
        async with self._db.conn.execute(
            "SELECT memories.* FROM memories_fts "
            "JOIN memories ON memories.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY bm25(memories_fts), memories.updated_at DESC LIMIT ?",
            (match, max(limit, 0)),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_to_memory(row) for row in rows]

    async def top(self, n: int = 10) -> list[Memory]:
        """Memories for the system prompt: the most used half, then the newest ones.

        Used memories come first, most used first; the rest of the places go
        to the newest memories, so a fact saved today shows up right away.
        """
        if n <= 0:
            return []
        async with self._db.conn.execute(
            "SELECT * FROM memories WHERE uses > 0 ORDER BY uses DESC, updated_at DESC LIMIT ?",
            ((n + 1) // 2,),
        ) as cursor:
            chosen = [_to_memory(row) for row in await cursor.fetchall()]
        taken = [memory.id for memory in chosen]
        marks = ", ".join("?" for _ in taken)
        exclude = f"WHERE id NOT IN ({marks})" if taken else ""
        async with self._db.conn.execute(
            f"SELECT * FROM memories {exclude} ORDER BY updated_at DESC, rowid DESC LIMIT ?",
            (*taken, n - len(chosen)),
        ) as cursor:
            chosen += [_to_memory(row) for row in await cursor.fetchall()]
        return chosen

    async def bump_uses(self, memory_ids: Collection[str]) -> None:
        """Count one more use of each memory, e.g. because a search found it."""
        if not memory_ids:
            return
        marks = ", ".join("?" for _ in memory_ids)
        await self._db.conn.execute(
            f"UPDATE memories SET uses = uses + 1 WHERE id IN ({marks})", tuple(memory_ids)
        )
        await self._db.conn.commit()

    async def delete(self, memory_id: str) -> bool:
        """Forget a memory. False if there was none with this id."""
        cursor = await self._db.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self._db.conn.commit()
        return cursor.rowcount > 0
