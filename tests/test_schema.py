"""Tests for the real migrations shipped with the package."""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing

import pytest

from zero.paths import db_path
from zero.store.db import MIGRATIONS_DIR, Database

NOW = "2026-01-01T12:00:00Z"


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(db_path()) as database:
        await database.migrate()
        yield database


async def names(db: Database, kind: str) -> set[str]:
    async with db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'", (kind,)
    ) as cursor:
        return {row["name"] for row in await cursor.fetchall()}


async def columns(db: Database, table: str) -> set[str]:
    async with db.conn.execute(f"PRAGMA table_info({table})") as cursor:
        return {row["name"] for row in await cursor.fetchall()}


async def count(db: Database, table: str) -> int:
    async with db.conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


def test_initial_migration_ships_with_the_package() -> None:
    assert (MIGRATIONS_DIR / "0001_init.sql").is_file()


async def test_fresh_database_is_migrated_to_the_latest_version() -> None:
    async with Database(db_path()) as database:
        assert await database.migrate() == [1, 2]
        assert await database.version() == 2
        assert await database.migrate() == []


async def test_tables_exist_after_migration(db: Database) -> None:
    assert await names(db, "table") == {
        "schema_version",
        "sessions",
        "messages",
        "tool_runs",
        "approval_rules",
    }


async def test_messages_are_indexed_by_session_and_time(db: Database) -> None:
    assert "idx_messages_session" in await names(db, "index")


async def test_tool_runs_are_indexed_by_turn(db: Database) -> None:
    assert "idx_tool_runs_turn" in await names(db, "index")


async def test_a_call_id_may_repeat_in_tool_runs(db: Database) -> None:
    for run_id in ("r1", "r2"):
        await db.conn.execute(
            "INSERT INTO tool_runs (id, call_id, session_id, turn_id, tool, created_at) "
            "VALUES (?, 'call-1', 's1', 't1', 'add', ?)",
            (run_id, NOW),
        )

    assert await count(db, "tool_runs") == 2


async def test_migration_2_keeps_existing_tool_runs() -> None:
    async with Database(db_path()) as database:
        await database.migrate()
    # Roll the schema back to version 1 by hand and add a run in the old shape.
    with closing(sqlite3.connect(db_path())) as raw:
        raw.executescript(
            "DROP TABLE tool_runs;"
            "CREATE TABLE tool_runs (call_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
            " turn_id TEXT NOT NULL, tool TEXT NOT NULL, label TEXT, input_json TEXT,"
            " status TEXT, summary TEXT, output_preview TEXT, duration_ms INTEGER,"
            " created_at TEXT NOT NULL);"
            "INSERT INTO tool_runs (call_id, session_id, turn_id, tool, label, created_at)"
            f" VALUES ('old', 's1', 't1', 'read_file', '~/a.md', '{NOW}');"
            "DELETE FROM schema_version WHERE version = 2;"
        )

    async with Database(db_path()) as database:
        assert await database.migrate() == [2]
        async with database.conn.execute("SELECT id, call_id, label FROM tool_runs") as cursor:
            ((run_id, call_id, label),) = await cursor.fetchall()

    assert (call_id, label) == ("old", "~/a.md")
    assert len(run_id) == 32


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("sessions", {"id", "title", "created_at", "updated_at", "archived"}),
        (
            "messages",
            {
                "id",
                "session_id",
                "turn_id",
                "role",
                "content_json",
                "tool_calls_json",
                "tool_call_id",
                "is_error",
                "status",
                "source",
                "created_at",
            },
        ),
        (
            "tool_runs",
            {
                "id",
                "call_id",
                "session_id",
                "turn_id",
                "tool",
                "label",
                "input_json",
                "status",
                "summary",
                "output_preview",
                "duration_ms",
                "created_at",
            },
        ),
        ("approval_rules", {"tool", "decision", "created_at"}),
    ],
)
async def test_table_columns_match_the_spec(table: str, expected: set[str], db: Database) -> None:
    assert await columns(db, table) == expected


async def test_deleting_a_session_deletes_its_messages(db: Database) -> None:
    await db.conn.execute(
        "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s1', ?, ?)", (NOW, NOW)
    )
    await db.conn.execute(
        "INSERT INTO messages (id, session_id, role, content_json, created_at) "
        "VALUES ('m1', 's1', 'user', '[]', ?)",
        (NOW,),
    )

    await db.conn.execute("DELETE FROM sessions WHERE id = 's1'")

    assert await count(db, "messages") == 0


async def test_message_needs_an_existing_session(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(
            "INSERT INTO messages (id, session_id, role, content_json, created_at) "
            "VALUES ('m1', 'missing', 'user', '[]', ?)",
            (NOW,),
        )


async def test_column_defaults(db: Database) -> None:
    await db.conn.execute(
        "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s1', ?, ?)", (NOW, NOW)
    )
    await db.conn.execute(
        "INSERT INTO messages (id, session_id, role, content_json, created_at) "
        "VALUES ('m1', 's1', 'user', '[]', ?)",
        (NOW,),
    )

    async with db.conn.execute("SELECT title, archived FROM sessions") as cursor:
        session = await cursor.fetchone()
    async with db.conn.execute("SELECT is_error, turn_id FROM messages") as cursor:
        message = await cursor.fetchone()

    assert session is not None and message is not None
    assert (session["title"], session["archived"]) == (None, 0)
    assert (message["is_error"], message["turn_id"]) == (0, None)


async def test_one_saved_rule_per_tool(db: Database) -> None:
    insert = "INSERT INTO approval_rules (tool, decision, created_at) VALUES (?, 'allow', ?)"
    await db.conn.execute(insert, ("run_command", NOW))

    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(insert, ("run_command", NOW))
