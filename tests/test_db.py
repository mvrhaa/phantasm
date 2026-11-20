"""Tests for phantasm.store.db."""

import sqlite3
import stat
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from phantasm.paths import db_path
from phantasm.store.db import Database, MigrationError

CREATE_NOTES = """
CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
INSERT INTO notes (text) VALUES ('first');
"""


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "migrations"
    directory.mkdir()
    return directory


@pytest.fixture
async def db(migrations_dir: Path) -> AsyncIterator[Database]:
    database = Database(db_path(), migrations_dir)
    await database.connect()
    yield database
    await database.close()


async def table_names(db: Database) -> set[str]:
    async with db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cursor:
        return {row["name"] for row in await cursor.fetchall()}


async def scalar(db: Database, sql: str) -> object:
    async with db.conn.execute(sql) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return row[0]


async def test_connect_creates_the_file_and_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "phantasm.db"

    async with Database(path):
        assert path.is_file()


async def test_database_file_is_private_to_the_owner(db: Database) -> None:
    assert stat.S_IMODE(db.path.stat().st_mode) == 0o600


async def test_foreign_keys_are_enabled(db: Database) -> None:
    assert await scalar(db, "PRAGMA foreign_keys") == 1


async def test_journal_mode_is_wal(db: Database) -> None:
    assert await scalar(db, "PRAGMA journal_mode") == "wal"


async def test_rows_are_addressable_by_column_name(db: Database) -> None:
    async with db.conn.execute("SELECT 1 AS one, 'x' AS letter") as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert (row["one"], row["letter"]) == (1, "x")


async def test_conn_is_unavailable_before_connect_and_after_close() -> None:
    database = Database(db_path())
    with pytest.raises(RuntimeError, match="not connected"):
        _ = database.conn

    await database.connect()
    await database.close()

    with pytest.raises(RuntimeError, match="not connected"):
        _ = database.conn


async def test_connect_and_close_can_be_repeated() -> None:
    database = Database(db_path())

    await database.connect()
    first = database.conn
    await database.connect()
    assert database.conn is first

    await database.close()
    await database.close()


async def test_fresh_database_has_version_0(db: Database) -> None:
    assert await db.version() == 0
    assert await db.migrate() == []


async def test_missing_migrations_directory_means_nothing_to_apply(tmp_path: Path) -> None:
    async with Database(db_path(), tmp_path / "does-not-exist") as database:
        assert await database.migrate() == []
        assert await database.version() == 0


async def test_migration_is_applied(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)

    applied = await db.migrate()

    assert applied == [1]
    assert await db.version() == 1
    assert "notes" in await table_names(db)


async def test_repeated_migrate_does_nothing(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)
    await db.migrate()

    applied = await db.migrate()

    assert applied == []
    assert await db.version() == 1
    assert await scalar(db, "SELECT COUNT(*) FROM notes") == 1
    assert await scalar(db, "SELECT COUNT(*) FROM schema_version") == 1


async def test_migration_is_not_reapplied_after_reconnect(migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)
    async with Database(db_path(), migrations_dir) as first:
        await first.migrate()

    async with Database(db_path(), migrations_dir) as second:
        assert await second.migrate() == []
        assert await second.version() == 1
        assert await scalar(second, "SELECT COUNT(*) FROM notes") == 1


async def test_migrations_run_in_numeric_order(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "10_third.sql").write_text("INSERT INTO log (step) VALUES ('third');")
    (migrations_dir / "2_second.sql").write_text("INSERT INTO log (step) VALUES ('second');")
    (migrations_dir / "1_first.sql").write_text(
        "CREATE TABLE log (id INTEGER PRIMARY KEY, step TEXT); "
        "INSERT INTO log (step) VALUES ('first');"
    )

    applied = await db.migrate()

    async with db.conn.execute("SELECT step FROM log ORDER BY id") as cursor:
        steps = [row["step"] for row in await cursor.fetchall()]
    assert applied == [1, 2, 10]
    assert steps == ["first", "second", "third"]


async def test_only_new_migrations_are_applied(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)
    await db.migrate()
    (migrations_dir / "0002_tags.sql").write_text("CREATE TABLE tags (name TEXT);")

    applied = await db.migrate()

    assert applied == [2]
    assert await db.version() == 2
    assert await scalar(db, "SELECT COUNT(*) FROM notes") == 1


async def test_failed_migration_is_rolled_back_completely(
    db: Database, migrations_dir: Path
) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)
    broken = migrations_dir / "0002_broken.sql"
    broken.write_text("CREATE TABLE half_done (id INTEGER); INSERT INTO no_such_table VALUES (1);")

    with pytest.raises(MigrationError, match="0002_broken.sql"):
        await db.migrate()

    assert await db.version() == 1
    assert "half_done" not in await table_names(db)
    assert "notes" in await table_names(db)

    broken.write_text("CREATE TABLE half_done (id INTEGER);")
    assert await db.migrate() == [2]
    assert "half_done" in await table_names(db)


async def test_migration_without_trailing_semicolon_is_applied(
    db: Database, migrations_dir: Path
) -> None:
    (migrations_dir / "0001_tags.sql").write_text("CREATE TABLE tags (name TEXT)  -- no semicolon")

    assert await db.migrate() == [1]
    assert await db.version() == 1


@pytest.mark.parametrize("name", ["notes.sql", "one_notes.sql", "0001.sql", "0001-notes.sql"])
async def test_badly_named_migration_is_rejected(
    name: str, db: Database, migrations_dir: Path
) -> None:
    (migrations_dir / name).write_text("CREATE TABLE notes (id INTEGER);")

    with pytest.raises(MigrationError, match="Bad migration file name"):
        await db.migrate()

    assert "notes" not in await table_names(db)


async def test_duplicate_migration_number_is_rejected(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text("CREATE TABLE notes (id INTEGER);")
    (migrations_dir / "1_tags.sql").write_text("CREATE TABLE tags (id INTEGER);")

    with pytest.raises(MigrationError, match="share number 1"):
        await db.migrate()


async def test_database_newer_than_the_code_is_rejected(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_notes.sql").write_text(CREATE_NOTES)
    (migrations_dir / "0002_tags.sql").write_text("CREATE TABLE tags (name TEXT);")
    await db.migrate()
    (migrations_dir / "0002_tags.sql").unlink()

    with pytest.raises(MigrationError, match="newer than this version"):
        await db.migrate()


async def test_foreign_keys_are_enforced(db: Database, migrations_dir: Path) -> None:
    (migrations_dir / "0001_family.sql").write_text(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
        "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL "
        "REFERENCES parent(id) ON DELETE CASCADE);"
    )
    await db.migrate()

    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute("INSERT INTO child (parent_id) VALUES (42)")
