"""Tests for the real migrations shipped with the package."""

import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

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


LATEST = max(int(path.name.split("_")[0]) for path in MIGRATIONS_DIR.glob("*.sql"))


async def test_fresh_database_is_migrated_to_the_latest_version() -> None:
    async with Database(db_path()) as database:
        assert await database.migrate() == list(range(1, LATEST + 1))
        assert await database.version() == LATEST
        assert await database.migrate() == []


async def test_tables_exist_after_migration(db: Database) -> None:
    assert await names(db, "table") == {
        "schema_version",
        "sessions",
        "messages",
        "tool_runs",
        "approval_rules",
        "memories",
        "memories_fts",
        "memories_fts_config",
        "memories_fts_data",
        "memories_fts_docsize",
        "memories_fts_idx",
        "todos",
        "schedules",
        "session_summaries",
        "agents",
        "model_calls",
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


async def test_migration_2_keeps_existing_tool_runs(tmp_path: Path) -> None:
    only_first = tmp_path / "migrations"
    only_first.mkdir()
    shutil.copy(MIGRATIONS_DIR / "0001_init.sql", only_first)
    async with Database(db_path(), migrations_dir=only_first) as database:
        assert await database.migrate() == [1]
        await database.conn.execute(
            "INSERT INTO tool_runs (call_id, session_id, turn_id, tool, label, created_at) "
            "VALUES ('old', 's1', 't1', 'read_file', '~/a.md', ?)",
            (NOW,),
        )
        await database.conn.commit()

    async with Database(db_path()) as database:
        assert (await database.migrate())[0] == 2
        async with database.conn.execute("SELECT id, call_id, label FROM tool_runs") as cursor:
            ((run_id, call_id, label),) = await cursor.fetchall()

    assert (call_id, label) == ("old", "~/a.md")
    assert len(run_id) == 32


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("sessions", {"id", "title", "created_at", "updated_at", "archived", "kind"}),
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
        ("memories", {"id", "text", "tags", "created_at", "updated_at", "uses"}),
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
