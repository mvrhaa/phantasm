"""Tests of session kinds: the user's chats kept apart from agent and schedule sessions."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from phantasm import clock
from phantasm.paths import db_path
from phantasm.protocol.messages import SessionCreated
from phantasm.store.db import Database
from phantasm.store.sessions import SessionsRepo
from tests.agents import RunnerKit


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(db_path()) as database:
        await database.migrate()
        yield database


async def test_new_sessions_are_chats_unless_told_otherwise(db: Database) -> None:
    sessions = SessionsRepo(db)

    chat = await sessions.create("Hello")
    agent = await sessions.create("Agent: reader", kind="agent")

    assert (chat.kind, agent.kind) == ("chat", "agent")
    assert chat.info().kind == "chat"
    assert agent.info().model_dump(exclude={"updated_at"}) == {
        "id": agent.id,
        "title": "Agent: reader",
        "kind": "agent",
    }
    assert {s.id: s.kind for s in await sessions.list()} == {chat.id: "chat", agent.id: "agent"}
    stored = await sessions.get(agent.id)
    assert stored is not None and stored.kind == "agent"


async def test_an_agents_session_is_announced_as_such(db: Database) -> None:
    kit = RunnerKit(db)
    parent = await kit.sessions.create()

    agent = await kit.runner.start("reader", "task", parent.id)
    await kit.runner.close()

    created = [e for e in kit.client.received if isinstance(e, SessionCreated)]
    assert [(e.session.id, e.session.kind) for e in created] == [(agent.session_id, "agent")]


async def test_the_migration_sorts_out_existing_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_files = Database._migration_files

    def up_to_8(self: Database) -> list[tuple[int, Path]]:
        return [(number, path) for number, path in all_files(self) if number <= 8]

    now = clock.to_iso(clock.now())
    async with Database(db_path()) as database:
        monkeypatch.setattr(Database, "_migration_files", up_to_8)
        await database.migrate()
        for session_id in ("chat-1", "scheduled-abc", "agent-own", "parent"):
            await database.conn.execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
        await database.conn.execute(
            "INSERT INTO agents (id, name, parent_session_id, session_id, task, created_at, "
            "updated_at) VALUES ('a1', 'reader', 'parent', 'agent-own', 'task', ?, ?)",
            (now, now),
        )
        await database.conn.commit()

        monkeypatch.setattr(Database, "_migration_files", all_files)
        assert 9 in await database.migrate()
        kinds = {s.id: s.kind for s in await SessionsRepo(database).list()}

    assert kinds == {
        "chat-1": "chat",
        "scheduled-abc": "scheduled",
        "agent-own": "agent",
        "parent": "chat",
    }
