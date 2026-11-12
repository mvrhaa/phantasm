"""Tests of the agents repository."""

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from tests.conftest import FrozenClock
from zero.paths import db_path
from zero.store.agents import Agent, AgentsRepo
from zero.store.db import Database
from zero.store.sessions import SessionsRepo


@pytest.fixture
async def stores() -> AsyncIterator[tuple[SessionsRepo, AgentsRepo]]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield SessionsRepo(db), AgentsRepo(db)


async def started(sessions: SessionsRepo, agents: AgentsRepo, parent: str, name: str) -> Agent:
    own = await sessions.create(f"Agent: {name}")
    return await agents.add(name, parent, own.id, f"Task of {name}")


async def test_a_new_agent_is_at_work(
    stores: tuple[SessionsRepo, AgentsRepo], frozen_clock: FrozenClock
) -> None:
    sessions, agents = stores
    parent, own = await sessions.create(), await sessions.create()

    agent = await agents.add("  Code   reader ", parent.id, own.id, "  Read main.py  \n")

    assert agent.name == "Code reader"
    assert agent.task == "Read main.py"
    assert (agent.parent_session_id, agent.session_id) == (parent.id, own.id)
    assert (agent.status, agent.report, agent.active) == ("working", None, True)
    assert agent.created_at == agent.updated_at == frozen_clock.current
    assert await agents.get(agent.id) == agent
    assert await agents.by_session(own.id) == agent


@pytest.mark.parametrize(("name", "task"), [("  ", "task"), ("name", " \n ")])
async def test_name_and_task_are_required(
    stores: tuple[SessionsRepo, AgentsRepo], name: str, task: str
) -> None:
    sessions, agents = stores
    parent, own = await sessions.create(), await sessions.create()

    with pytest.raises(ValueError, match="needs a"):
        await agents.add(name, parent.id, own.id, task)


async def test_unknown_ids_give_none(stores: tuple[SessionsRepo, AgentsRepo]) -> None:
    sessions, agents = stores
    ordinary = await sessions.create()

    assert await agents.get("nope") is None
    assert await agents.by_session(ordinary.id) is None
    assert await agents.set_status("nope", "done") is None


async def test_status_and_report(
    stores: tuple[SessionsRepo, AgentsRepo], frozen_clock: FrozenClock
) -> None:
    sessions, agents = stores
    agent = await started(sessions, agents, (await sessions.create()).id, "a")
    frozen_clock.advance(60)

    asking = await agents.set_status(agent.id, "asking")
    done = await agents.set_status(agent.id, "done", report="All read.")
    again = await agents.set_status(agent.id, "stopped")

    assert asking is not None and asking.updated_at == frozen_clock.current
    assert asking.report is None
    assert done is not None and (done.status, done.report, done.active) == (
        "done",
        "All read.",
        False,
    )
    assert again is not None and again.report == "All read."


async def test_agents_of_a_parent_and_the_active_ones(
    stores: tuple[SessionsRepo, AgentsRepo],
) -> None:
    sessions, agents = stores
    first, second = await sessions.create(), await sessions.create()
    await started(sessions, agents, first.id, "a")
    b = await started(sessions, agents, first.id, "b")
    c = await started(sessions, agents, second.id, "c")
    await agents.set_status(b.id, "failed", report="crashed")
    await agents.set_status(c.id, "waiting_user")

    assert [agent.name for agent in await agents.list(first.id)] == ["a", "b"]
    assert [agent.name for agent in await agents.active()] == ["a", "c"]
    assert [agent.name for agent in await agents.active(first.id)] == ["a"]
    assert await agents.list("no-such-session") == []


async def test_deleting_the_parent_removes_its_agents(
    stores: tuple[SessionsRepo, AgentsRepo],
) -> None:
    sessions, agents = stores
    parent = await sessions.create()
    agent = await started(sessions, agents, parent.id, "a")

    await sessions.delete(parent.id)

    assert await agents.get(agent.id) is None


async def test_deleting_the_agents_session_keeps_the_record(
    stores: tuple[SessionsRepo, AgentsRepo],
) -> None:
    sessions, agents = stores
    agent = await started(sessions, agents, (await sessions.create()).id, "a")
    await agents.set_status(agent.id, "stopped", report="stopped by the user")
    assert agent.session_id is not None

    await sessions.delete(agent.session_id)

    kept = await agents.get(agent.id)
    assert kept is not None
    assert (kept.session_id, kept.status, kept.report) == (None, "stopped", "stopped by the user")


async def test_a_session_belongs_to_one_agent(stores: tuple[SessionsRepo, AgentsRepo]) -> None:
    sessions, agents = stores
    parent = await sessions.create()
    agent = await started(sessions, agents, parent.id, "a")
    assert agent.session_id is not None

    with pytest.raises(Exception, match="UNIQUE"):
        await agents.add("b", parent.id, agent.session_id, "task")


async def test_a_parent_must_exist(stores: tuple[SessionsRepo, AgentsRepo]) -> None:
    sessions, agents = stores
    own = await sessions.create()

    with pytest.raises(Exception, match="FOREIGN KEY"):
        await agents.add("a", "no-such-session", own.id, "task")


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Agent.model_validate(
            {
                "id": "x",
                "name": "a",
                "parent_session_id": "p",
                "session_id": None,
                "task": "t",
                "status": "sleeping",
                "report": None,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
