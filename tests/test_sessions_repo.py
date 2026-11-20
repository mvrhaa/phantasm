"""Tests for phantasm.store.sessions."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from phantasm.paths import db_path
from phantasm.store.db import Database
from phantasm.store.sessions import Session, SessionsRepo

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(db_path()) as database:
        await database.migrate()
        yield database


@pytest.fixture
def repo(db: Database, frozen_clock) -> SessionsRepo:
    return SessionsRepo(db)


async def test_create_returns_a_new_session(repo: SessionsRepo) -> None:
    session = await repo.create()

    assert isinstance(session, Session)
    assert uuid.UUID(session.id).version == 4
    assert session.title is None
    assert session.created_at == session.updated_at == START
    assert session.archived is False


async def test_create_with_title(repo: SessionsRepo) -> None:
    session = await repo.create("Plans")

    assert session.title == "Plans"


async def test_get_returns_the_stored_session(repo: SessionsRepo) -> None:
    created = await repo.create("Plans")

    assert await repo.get(created.id) == created


async def test_get_unknown_session_returns_none(repo: SessionsRepo) -> None:
    assert await repo.get("missing") is None


async def test_sessions_survive_a_reconnect(repo: SessionsRepo) -> None:
    created = await repo.create("Plans")

    async with Database(db_path()) as other:
        assert await SessionsRepo(other).get(created.id) == created


async def test_list_is_empty_at_first(repo: SessionsRepo) -> None:
    assert await repo.list() == []


async def test_list_puts_recently_updated_first(repo: SessionsRepo, frozen_clock) -> None:
    first = await repo.create("first")
    frozen_clock.advance(1)
    second = await repo.create("second")
    frozen_clock.advance(1)
    third = await repo.create("third")
    frozen_clock.advance(1)
    await repo.touch(first.id)

    titles = [session.title for session in await repo.list()]

    assert titles == ["first", "third", "second"]
    assert {second.id, third.id} < {session.id for session in await repo.list()}


async def test_list_hides_archived_sessions_by_default(repo: SessionsRepo, db: Database) -> None:
    kept = await repo.create("kept")
    archived = await repo.create("archived")
    await db.conn.execute("UPDATE sessions SET archived = 1 WHERE id = ?", (archived.id,))
    await db.conn.commit()

    visible = await repo.list()
    everything = await repo.list(include_archived=True)

    assert [session.id for session in visible] == [kept.id]
    assert {session.id for session in everything} == {kept.id, archived.id}
    assert next(s for s in everything if s.id == archived.id).archived is True


async def test_rename_changes_title_and_updated_at(repo: SessionsRepo, frozen_clock) -> None:
    created = await repo.create("old")
    frozen_clock.advance(60)

    renamed = await repo.rename(created.id, "new")

    assert renamed is not None
    assert renamed.title == "new"
    assert renamed.created_at == START
    assert renamed.updated_at == START + timedelta(seconds=60)
    assert await repo.get(created.id) == renamed


async def test_rename_unknown_session_returns_none(repo: SessionsRepo) -> None:
    assert await repo.rename("missing", "new") is None
    assert await repo.list() == []


async def test_set_title_if_missing_titles_an_untitled_session(
    repo: SessionsRepo, frozen_clock
) -> None:
    created = await repo.create()
    frozen_clock.advance(60)

    titled = await repo.set_title_if_missing(created.id, "Trip to Rome")

    assert titled is not None
    assert titled.title == "Trip to Rome"
    assert titled.updated_at == START
    assert await repo.get(created.id) == titled


async def test_set_title_if_missing_keeps_an_existing_title(repo: SessionsRepo) -> None:
    created = await repo.create("Chosen by the user")

    assert await repo.set_title_if_missing(created.id, "Automatic") is None
    assert await repo.get(created.id) == created


async def test_set_title_if_missing_on_unknown_session_returns_none(repo: SessionsRepo) -> None:
    assert await repo.set_title_if_missing("missing", "Title") is None


async def test_touch_sets_updated_at_to_now(repo: SessionsRepo, frozen_clock) -> None:
    created = await repo.create("Plans")
    frozen_clock.advance(30)

    touched = await repo.touch(created.id)

    assert touched is not None
    assert touched.updated_at == START + timedelta(seconds=30)
    assert touched.title == "Plans"


async def test_touch_accepts_an_explicit_time(repo: SessionsRepo) -> None:
    created = await repo.create()
    moment = datetime(2026, 6, 1, 8, 30, 0, tzinfo=UTC)

    touched = await repo.touch(created.id, moment)

    assert touched is not None
    assert touched.updated_at == moment


async def test_touch_unknown_session_returns_none(repo: SessionsRepo) -> None:
    assert await repo.touch("missing") is None


async def test_delete_removes_the_session(repo: SessionsRepo) -> None:
    created = await repo.create()

    assert await repo.delete(created.id) is True
    assert await repo.get(created.id) is None
    assert await repo.list() == []


async def test_delete_unknown_session_returns_false(repo: SessionsRepo) -> None:
    assert await repo.delete("missing") is False


async def test_delete_cascades_to_messages(repo: SessionsRepo, db: Database) -> None:
    doomed = await repo.create("doomed")
    survivor = await repo.create("survivor")
    for message_id, session_id in (("m1", doomed.id), ("m2", doomed.id), ("m3", survivor.id)):
        await db.conn.execute(
            "INSERT INTO messages (id, session_id, role, content_json, created_at) "
            "VALUES (?, ?, 'user', '[]', '2026-01-01T12:00:00.000000+00:00')",
            (message_id, session_id),
        )
    await db.conn.commit()

    await repo.delete(doomed.id)

    async with db.conn.execute("SELECT id FROM messages") as cursor:
        remaining = [row["id"] for row in await cursor.fetchall()]
    assert remaining == ["m3"]


async def test_titles_are_stored_as_given(repo: SessionsRepo) -> None:
    title = "Планы на лето ☀️ '; DROP TABLE sessions; --"

    created = await repo.create(title)

    stored = await repo.get(created.id)
    assert stored is not None
    assert stored.title == title
