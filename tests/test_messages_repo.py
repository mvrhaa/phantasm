"""Tests for zero.store.messages."""

import sqlite3
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from zero.paths import db_path
from zero.protocol.messages import FilePart, ImagePart, TextPart
from zero.store.db import Database
from zero.store.messages import MessagesRepo, StoredMessage
from zero.store.sessions import SessionsRepo

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
async def db(frozen_clock) -> AsyncIterator[Database]:
    async with Database(db_path()) as database:
        await database.migrate()
        yield database


@pytest.fixture
def repo(db: Database) -> MessagesRepo:
    return MessagesRepo(db)


@pytest.fixture
async def session_id(db: Database) -> str:
    return (await SessionsRepo(db).create()).id


async def add_texts(repo: MessagesRepo, session_id: str, count: int) -> list[StoredMessage]:
    return [
        await repo.append(session_id, "user", [TextPart(text=f"message {number}")])
        for number in range(1, count + 1)
    ]


def texts(messages: list[StoredMessage]) -> list[str]:
    return [part.text for message in messages for part in message.content]  # type: ignore[union-attr]


async def test_append_returns_the_stored_message(repo: MessagesRepo, session_id: str) -> None:
    message = await repo.append(
        session_id, "user", [TextPart(text="hi")], turn_id="t1", status="ok", source="desktop"
    )

    assert uuid.UUID(message.id).version == 4
    assert message.session_id == session_id
    assert message.role == "user"
    assert message.content == [TextPart(text="hi")]
    assert (message.turn_id, message.status, message.source) == ("t1", "ok", "desktop")
    assert message.created_at == START
    assert await repo.list(session_id) == [message]


async def test_optional_fields_default_to_empty(repo: MessagesRepo, session_id: str) -> None:
    message = await repo.append(session_id, "assistant", [])

    stored = (await repo.list(session_id))[0]
    assert stored == message
    assert stored.content == []
    assert stored.tool_calls == []
    assert stored.tool_call_id is None
    assert stored.is_error is False
    assert (stored.turn_id, stored.status, stored.source) == (None, None, None)


async def test_every_kind_of_content_part_is_restored(repo: MessagesRepo, session_id: str) -> None:
    content = [
        TextPart(text="Смотри: ☀️ and 'quotes'"),
        ImagePart(mime="image/png", data_b64="AAAA"),
        FilePart(name="report.pdf", path="/tmp/report.pdf"),
    ]

    await repo.append(session_id, "user", content)

    stored = (await repo.list(session_id))[0]
    assert stored.content == content
    assert [type(part) for part in stored.content] == [TextPart, ImagePart, FilePart]


async def test_tool_calls_and_tool_results_are_restored(
    repo: MessagesRepo, session_id: str
) -> None:
    calls = [{"id": "c1", "name": "read_file", "input": {"path": "~/заметки.md"}}]

    await repo.append(session_id, "assistant", [], turn_id="t1", tool_calls=calls)
    await repo.append(
        session_id,
        "tool",
        [TextPart(text="no such file")],
        turn_id="t1",
        tool_call_id="c1",
        is_error=True,
    )

    assistant, tool = await repo.list(session_id)
    assert assistant.tool_calls == calls
    assert tool.tool_call_id == "c1"
    assert tool.is_error is True
    assert tool.tool_calls == []


async def test_messages_survive_a_reconnect(repo: MessagesRepo, session_id: str) -> None:
    message = await repo.append(session_id, "user", [TextPart(text="hi")])

    async with Database(db_path()) as other:
        assert await MessagesRepo(other).list(session_id) == [message]


async def test_append_to_unknown_session_fails(repo: MessagesRepo) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        await repo.append("missing", "user", [TextPart(text="hi")])


async def test_list_is_empty_for_a_new_session(repo: MessagesRepo, session_id: str) -> None:
    assert await repo.list(session_id) == []


async def test_list_returns_messages_oldest_first(
    repo: MessagesRepo, session_id: str, frozen_clock
) -> None:
    for number in (1, 2, 3):
        await repo.append(session_id, "user", [TextPart(text=f"message {number}")])
        frozen_clock.advance(1)

    assert texts(await repo.list(session_id)) == ["message 1", "message 2", "message 3"]


async def test_messages_with_the_same_time_keep_insertion_order(
    repo: MessagesRepo, session_id: str
) -> None:
    await add_texts(repo, session_id, 5)

    assert texts(await repo.list(session_id)) == [f"message {n}" for n in (1, 2, 3, 4, 5)]


async def test_limit_keeps_the_latest_messages(repo: MessagesRepo, session_id: str) -> None:
    await add_texts(repo, session_id, 5)

    assert texts(await repo.list(session_id, limit=2)) == ["message 4", "message 5"]


async def test_default_limit_is_50(repo: MessagesRepo, session_id: str) -> None:
    await add_texts(repo, session_id, 55)

    page = await repo.list(session_id)

    assert len(page) == 50
    assert texts(page)[0] == "message 6"
    assert texts(page)[-1] == "message 55"


async def test_before_pages_back_through_history(repo: MessagesRepo, session_id: str) -> None:
    await add_texts(repo, session_id, 7)

    newest = await repo.list(session_id, limit=3)
    middle = await repo.list(session_id, before=newest[0].id, limit=3)
    oldest = await repo.list(session_id, before=middle[0].id, limit=3)
    nothing = await repo.list(session_id, before=oldest[0].id, limit=3)

    assert texts(newest) == ["message 5", "message 6", "message 7"]
    assert texts(middle) == ["message 2", "message 3", "message 4"]
    assert texts(oldest) == ["message 1"]
    assert nothing == []


async def test_before_works_across_different_times(
    repo: MessagesRepo, session_id: str, frozen_clock
) -> None:
    stored = []
    for number in (1, 2, 3, 4):
        stored.append(await repo.append(session_id, "user", [TextPart(text=f"message {number}")]))
        frozen_clock.advance(1)

    assert texts(await repo.list(session_id, before=stored[2].id)) == ["message 1", "message 2"]


async def test_unknown_before_gives_an_empty_page(repo: MessagesRepo, session_id: str) -> None:
    await add_texts(repo, session_id, 3)

    assert await repo.list(session_id, before="missing") == []


async def test_sessions_do_not_see_each_others_messages(
    repo: MessagesRepo, session_id: str, db: Database
) -> None:
    other_id = (await SessionsRepo(db).create()).id
    mine = await repo.append(session_id, "user", [TextPart(text="mine")])
    theirs = await repo.append(other_id, "user", [TextPart(text="theirs")])

    assert await repo.list(session_id) == [mine]
    assert await repo.list(other_id) == [theirs]
    assert await repo.list(session_id, before=theirs.id) == []


@pytest.mark.parametrize("limit", [0, -5])
async def test_non_positive_limit_gives_nothing(
    limit: int, repo: MessagesRepo, session_id: str
) -> None:
    await add_texts(repo, session_id, 3)

    assert await repo.list(session_id, limit=limit) == []
