"""Tests of starting agents: their own session, their own context, their status."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.agents import RoleProvider, RunnerKit
from tests.turns import FuncTool, registry, text_of
from zero.agents.runner import NO_REPORT, RESTARTED, AgentError
from zero.config import Config
from zero.core.context import build_system_prompt
from zero.paths import db_path
from zero.protocol.messages import TurnFinished, Usage
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import AppState, get_state
from zero.store.agents import FINAL_STATUSES, Agent, AgentsRepo
from zero.store.db import Database
from zero.store.sessions import SessionsRepo
from zero.tools.base import ToolContext
from zero.tools.builtin.agents import AgentStart, AgentStartInput

HELLO = {"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}}


def start_call(name: str = "reader", task: str = "Count the files in ~/notes") -> list[Any]:
    call = ToolCall(id="a1", name="agent_start", input={"name": name, "task": task})
    return [ToolCallEvent(call=call)]


# --- Through the service -----------------------------------------------------


@contextmanager
def service(provider: FakeProvider) -> Iterator[tuple[AppState, WebSocketTestSession]]:
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(HELLO)
            assert ws.receive_json()["type"] == "hello.ok"
            yield state, ws


def events_until_agent(ws: WebSocketTestSession, status: str) -> list[dict[str, Any]]:
    """Every event up to the agent's final agent.updated, which must have ``status``."""
    seen = []
    while True:
        event = ws.receive_json()
        seen.append(event)
        if event["type"] == "agent.updated" and event["agent"]["status"] in FINAL_STATUSES:
            assert event["agent"]["status"] == status, event["agent"]
            return seen


def test_agent_works_on_its_task_in_its_own_session() -> None:
    provider = RoleProvider(
        main=FakeProvider([start_call(), [TextDelta(text="Started a reader.")]]),
        agent=FakeProvider([[TextDelta(text="There are 3 files.")]]),
    )
    with service(provider) as (state, ws):
        ws.send_json({"type": "message.send", "id": "m1", "text": "count", "source": "desktop"})
        seen = events_until_agent(ws, "done")

        updates = [e["agent"] for e in seen if e["type"] == "agent.updated"]
        working, done = updates
        main_session = working["parent_session_id"]
        agent_session = working["session_id"]
        assert (working["name"], working["status"]) == ("reader", "working")
        assert working["task"] == "Count the files in ~/notes"
        assert (done["status"], done["report"]) == ("done", "There are 3 files.")
        assert agent_session != main_session

        created = [e["session"] for e in seen if e["type"] == "session.created"]
        assert {"id": agent_session, "title": "Agent: reader"}.items() <= created[-1].items()
        started = [e for e in seen if e["type"] == "turn.started"]
        assert {(e["session_id"], e["source"]) for e in started} == {
            (main_session, "desktop"),
            (agent_session, "agent"),
        }
        finished = next(e for e in seen if e["type"] == "tool.finished")
        assert finished["status"] == "ok"

        (agent_call,) = provider.agent.calls
        system, *history = agent_call.messages
        assert "# Your role" in text_of(system) and "You are reader" in text_of(system)
        assert [(m.role, text_of(m)) for m in history] == [("user", "Count the files in ~/notes")]
        agent_tools = {tool.name for tool in agent_call.tools}
        assert "read_file" in agent_tools
        assert not any(name.startswith("agent_") for name in agent_tools)
        assert "agent_start" in {tool.name for tool in provider.main.calls[0].tools}
        assert "# Your role" not in text_of(provider.main.calls[0].messages[0])
        tool_result = provider.main.calls[1].messages[-1]
        assert text_of(tool_result).startswith("Started agent 'reader' (id ")


def test_an_agent_whose_turn_fails_is_marked_failed() -> None:
    provider = RoleProvider(
        main=FakeProvider([start_call(), [TextDelta(text="Started.")]]),
        agent=FakeProvider([[TextDelta(text="x")]], fail_times=1, fail_retryable=False),
    )
    with service(provider) as (state, ws):
        ws.send_json({"type": "message.send", "id": "m1", "text": "go", "source": "desktop"})
        seen = events_until_agent(ws, "failed")

    failed = seen[-1]["agent"]
    assert failed["report"] == "FakeProvider simulated failure"


def test_the_main_turn_does_not_wait_for_the_agent() -> None:
    provider = RoleProvider(
        main=FakeProvider([start_call(), [TextDelta(text="Started.")]]),
        agent=FakeProvider(infinite=True),
    )
    with service(provider) as (state, ws):
        ws.send_json({"type": "message.send", "id": "m1", "text": "go", "source": "desktop"})
        while (event := ws.receive_json())["type"] != "turn.started":
            pass
        main_session = event["session_id"]  # the main turn starts before the agent's
        while not ((event := ws.receive_json())["type"] == "turn.finished"):
            pass

        assert (event["session_id"], event["text"]) == (main_session, "Started.")
        (still_working,) = [turn for turn in state.turns.active if turn.agent is not None]
        assert still_working.agent is not None and still_working.agent.name == "reader"


async def test_the_service_fails_agents_left_by_its_last_run() -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        sessions, agents = SessionsRepo(db), AgentsRepo(db)
        parent, own = await sessions.create(), await sessions.create()
        left = await agents.add("left", parent.id, own.id, "task")

    with TestClient(create_app(Config(), FakeProvider([]))) as client:
        stored = client.portal.call(get_state(client.app).agents.get, left.id)  # type: ignore[union-attr,arg-type]

    assert stored is not None and (stored.status, stored.report) == ("failed", RESTARTED)


# --- AgentRunner -------------------------------------------------------------


@pytest.fixture
async def stores() -> AsyncIterator[RunnerKit]:
    async with Database(db_path()) as db:
        await db.migrate()
        kit = RunnerKit(db)
        yield kit
        await kit.runner.close()


def result(status: str, text: str = "", error: str | None = None) -> TurnFinished:
    return TurnFinished(
        session_id="s", turn_id="t", status=status, text=text, usage=Usage(), error=error
    )


async def test_start_announces_and_starts_the_first_turn(stores: RunnerKit) -> None:
    parent = await stores.sessions.create()

    agent = await stores.runner.start("reader", "  Read it  ", parent.id)

    assert stores.started == [(agent.id, "Read it")]
    assert stores.updates == [("working", None)]
    assert agent.session_id is not None
    session = await stores.sessions.get(agent.session_id)
    assert session is not None and session.title == "Agent: reader"


@pytest.mark.parametrize(
    ("finished", "expected"),
    [
        (result("ok", "Report."), ("done", "Report.")),
        (result("max_steps", "Partly done."), ("done", "Partly done.")),
        (result("ok", ""), ("done", NO_REPORT)),
        (result("error", error="overloaded"), ("failed", "overloaded")),
        (result("cancelled"), ("stopped", None)),
        (None, ("failed", "internal error")),
    ],
)
async def test_how_a_turn_ended_sets_the_status(
    stores: RunnerKit, finished: TurnFinished | None, expected: tuple[str, str | None]
) -> None:
    agent = await stores.runner.start("a", "task", (await stores.sessions.create()).id)

    await stores.runner.finished(agent.id, finished)

    stored = await stores.agents.get(agent.id)
    assert stored is not None and (stored.status, stored.report) == expected
    assert stores.updates[-1] == expected


async def test_a_stopped_agent_stays_stopped(stores: RunnerKit) -> None:
    agent = await stores.runner.start("a", "task", (await stores.sessions.create()).id)
    await stores.agents.set_status(agent.id, "stopped")

    await stores.runner.finished(agent.id, result("ok", "late report"))
    await stores.runner.finished("no-such-agent", result("ok"))

    stored = await stores.agents.get(agent.id)
    assert stored is not None and (stored.status, stored.report) == ("stopped", None)
    assert stores.updates == [("working", None)]


async def test_agents_do_not_start_agents(stores: RunnerKit) -> None:
    agent = await stores.runner.start("a", "task", (await stores.sessions.create()).id)
    assert agent.session_id is not None

    with pytest.raises(AgentError, match="cannot start agents"):
        await stores.runner.start("b", "task", agent.session_id)


async def test_a_bad_agent_leaves_no_session_behind(stores: RunnerKit) -> None:
    parent = await stores.sessions.create()

    with pytest.raises(AgentError, match="needs a task"):
        await stores.runner.start("a", "   ", parent.id)

    assert [s.id for s in await stores.sessions.list()] == [parent.id]
    assert stores.started == []


async def test_recover_fails_the_agents_an_earlier_run_left(stores: RunnerKit) -> None:
    parent = await stores.sessions.create()
    left = await stores.runner.start("left", "task", parent.id)
    finished = await stores.runner.start("finished", "task", parent.id)
    await stores.runner.finished(finished.id, result("ok", "done"))

    assert await stores.runner.recover() == 1

    stored = await stores.agents.get(left.id)
    assert stored is not None and (stored.status, stored.report) == ("failed", RESTARTED)
    assert await stores.runner.recover() == 0


# --- Pieces ------------------------------------------------------------------


def test_registry_without_a_prefix() -> None:
    async def nothing(params: Any, ctx: ToolContext) -> str:
        return ""

    tools = registry(*(FuncTool(name, nothing) for name in ("read", "agent_start", "agent_x")))

    assert [spec.name for spec in tools.without("agent_").specs()] == ["read"]
    assert len(tools.specs()) == 3  # the original is untouched


def test_agent_role_follows_the_environment() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)

    prompt = build_system_prompt(None, "agent", ["likes tea"], now=now, agent_name="reader")

    assert prompt.index("# Environment") < prompt.index("# Your role") < prompt.index("tea")
    assert "You are reader, an agent" in prompt
    assert "# Your role" not in build_system_prompt(None, "desktop", now=now)


async def test_agent_start_reports_why_it_could_not_start() -> None:
    class Refusing:
        async def start(self, name: str, task: str, parent_session_id: str) -> Agent:
            raise AgentError("an agent cannot start agents")

    tool = AgentStart(Refusing())
    ctx = SimpleNamespace(session_id="s")
    out = await tool.run(AgentStartInput(name="a", task="t"), ctx)  # type: ignore[arg-type]

    assert out.is_error
    assert out.content[0].text == (  # type: ignore[union-attr]
        "The agent could not be started: an agent cannot start agents."
    )
