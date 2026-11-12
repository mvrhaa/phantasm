"""SQLite connection and the migration runner."""

import logging
import re
from pathlib import Path
from types import TracebackType
from typing import Self

import aiosqlite

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DB_FILE_MODE = 0o600

_MIGRATION_NAME = re.compile(r"^(\d+)_.+\.sql$")

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migrations are malformed or one of them fails to apply."""


class Database:
    """One SQLite connection plus the schema migrations for it.

    Migration files are named ``<number>_<name>.sql`` and are applied in
    numeric order, each exactly once. They must not contain BEGIN or COMMIT:
    every file runs inside its own transaction.
    """

    def __init__(self, path: Path, migrations_dir: Path | None = None) -> None:
        self.path = path
        self.migrations_dir = migrations_dir or MIGRATIONS_DIR
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        """The open connection. Raises if ``connect()`` has not been called."""
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn

    async def connect(self) -> None:
        """Open the database file, creating it and its directory if needed."""
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA journal_mode=WAL")
        self.path.chmod(DB_FILE_MODE)
        self._conn = conn

    async def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def version(self) -> int:
        """Return the number of the last applied migration, 0 for a fresh database."""
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        async with self.conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    def _migration_files(self) -> list[tuple[int, Path]]:
        """Return ``(number, path)`` for every migration file, sorted by number."""
        if not self.migrations_dir.is_dir():
            return []
        found: dict[int, Path] = {}
        for path in self.migrations_dir.glob("*.sql"):
            match = _MIGRATION_NAME.match(path.name)
            if match is None:
                raise MigrationError(
                    f"Bad migration file name {path.name!r}, expected <number>_<name>.sql"
                )
            number = int(match.group(1))
            if number in found:
                raise MigrationError(
                    f"Migrations {found[number].name!r} and {path.name!r} share number {number}"
                )
            found[number] = path
        return sorted(found.items())

    async def migrate(self) -> list[int]:
        """Apply the migrations that are newer than the database.

        Returns the numbers that were applied; an empty list means the schema
        was already up to date.
        """
        current = await self.version()
        files = self._migration_files()
        if files and current > files[-1][0]:
            raise MigrationError(
                f"Database {self.path} has schema version {current}, "
                f"newer than this version of Zero supports ({files[-1][0]})"
            )

        applied: list[int] = []
        for number, path in files:
            if number <= current:
                continue
            sql = path.read_text(encoding="utf-8")
            script = (
                f"BEGIN;\n{sql}\n;INSERT INTO schema_version (version) VALUES ({number});\nCOMMIT;"
            )
            try:
                await self.conn.executescript(script)
            except aiosqlite.Error as exc:
                await self.conn.rollback()
                raise MigrationError(f"Migration {path.name} failed: {exc}") from exc
            logger.info("Applied migration %s", path.name)
            applied.append(number)
        return applied
