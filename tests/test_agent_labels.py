"""Tests of what clients see of agents: labeled events and approvals, and the agent limit."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.testclient import WebSocketTestSession

from phantasm.agents.runner import AgentApprovals, AgentError
from phantasm.approvals.manager import ApprovalAsk
from phantasm.config import AgentConfig, Config
from phantasm.core.cancellation import CancelToken
from phantasm.core.events import EventBus, ReplyTo
from phantasm.paths import db_path
from phantasm.protocol.messages import (
    AgentRef,
    SessionCreated,
    SessionInfo,
    TextDelta,
    TurnStarted,
)
from phantasm.providers.base import TextDelta as TextDeltaEvent
from phantasm.providers.base import ToolCall, ToolCallEvent
from phantasm.providers.fake import FakeProvider
from phantasm.server.app import create_app
from phantasm.server.state import AppState, get_state
from phantasm.store.db import Database
from tests.agents import RoleProvider, RunnerKit
from tests.turns import Audience, FakeApprover, RecordingClient

HELLO = {"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}}
READER = AgentRef(id="a1", name="reader")


async def test_an_agents_bus_marks_turn_events_only() -> None:
    first, second = RecordingClient(id="one"), RecordingClient(id="two")
    bus = EventBus(Audience([first, second])).for_agent(READER)
    other = AgentRef(id="a2", name="writer")
    session = SessionInfo(id="s", updated_at=datetime(2026, 1, 1, tzinfo=UTC))

    await bus.publish(TurnStarted(session_id="s", turn_id="t", source="agent"))
    await bus.publish(TextDelta(session_id="s", turn_id="t", delta="hi", agent=other))
    await bus.publish(SessionCreated(session=session), reply=ReplyTo("one", "m1"))

    started, delta, created = first.received
    assert started.agent == READER  # type: ignore[attr-defined]
    assert delta.agent == other  # type: ignore[attr-defined]
    assert created == SessionCreated(session=session, reply_to="m1")
    assert second.received[0].agent == READER  # type: ignore[attr-defined]
    assert second.received[2].reply_to is None


async def test_a_plain_bus_marks_nothing() -> None:
    client = RecordingClient()

    await EventBus(Audience([client])).publish(
        TurnStarted(session_id="s", turn_id="t", source="desktop")
    )

    assert client.received[0].agent is None  # type: ignore[attr-defined]
    assert "agent" not in client.received[0].model_dump_json(exclude_none=True)


@pytest.fixture
async def kit() -> AsyncIterator[RunnerKit]:
    async with Database(db_path()) as db:
        await db.migrate()
        kit = RunnerKit(db, max_agents=2)
        yield kit
        await kit.runner.close()


ASK = ApprovalAsk(session_id="s", turn_id="t", tool="run_command", label="$ ls", agent=READER)


async def test_an_agent_waits_for_the_user_while_an_approval_is_open(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    agent = await kit.runner.start("reader", "task", parent.id)
    statuses: list[str] = []

    class Watching(FakeApprover):
        async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Any:
            stored = await kit.agents.get(agent.id)
            assert stored is not None
            statuses.append(stored.status)
            return await super().request(ask, cancel)

    inner = Watching("deny")
    answer = await AgentApprovals(inner, kit.runner, agent.id).request(ASK, CancelToken())

    assert answer.decision == "deny"
    assert inner.asked == [ASK]
    assert statuses == ["waiting_user"]
    assert [status for status, _ in kit.updates] == ["working", "waiting_user", "working"]


async def test_a_stopped_agent_does_not_come_back_after_an_approval(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    agent = await kit.runner.start("reader", "task", parent.id)

    class StoppedMeanwhile(FakeApprover):
        async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Any:
            await kit.agents.set_status(agent.id, "stopped")
            return await super().request(ask, cancel)

    await AgentApprovals(StoppedMeanwhile(), kit.runner, agent.id).request(ASK, CancelToken())

    stored = await kit.agents.get(agent.id)
    assert stored is not None and stored.status == "stopped"


async def test_no_more_agents_than_the_limit(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    first = await kit.runner.start("a", "task", parent.id)
    second = await kit.runner.start("b", "task", parent.id)
    await kit.runner.ask(second.session_id or "", "?")

    with pytest.raises(AgentError, match="2 agents are already at work and the limit is 2"):
        await kit.runner.start("c", "task", parent.id)

    await kit.agents.set_status(first.id, "done", "ok")
    third = await kit.runner.start("c", "task", parent.id)
    assert third.status == "working"


@pytest.mark.parametrize("value", [0, 21])
def test_the_limit_is_kept_sane(value: int) -> None:
    with pytest.raises(ValidationError):
        AgentConfig(max_agents=value)


@contextmanager
def service(provider: FakeProvider) -> Iterator[tuple[AppState, WebSocketTestSession]]:
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(HELLO)
            assert ws.receive_json()["type"] == "hello.ok"
            yield state, ws


def call(tool: str, **tool_input: Any) -> list[Any]:
    return [ToolCallEvent(call=ToolCall(id=f"{tool}-1", name=tool, input=tool_input))]


def until_agent_final(ws: WebSocketTestSession, on_approval: Any = None) -> list[dict[str, Any]]:
    seen = []
    while True:
        seen.append(event := ws.receive_json())
        if event["type"] == "approval.request" and on_approval is not None:
            on_approval(event)
        if event["type"] == "agent.updated" and event["agent"]["status"] in ("done", "failed"):
            return seen


def test_clients_see_who_acts_and_who_asks() -> None:
    provider = RoleProvider(
        main=FakeProvider(
            [
                call("agent_start", name="reader", task="List the folder"),
                [TextDeltaEvent(text="Started.")],
                [TextDeltaEvent(text="Noted.")],
            ]
        ),
        agent=FakeProvider(
            [call("run_command", command="ls"), [TextDeltaEvent(text="Not allowed to look.")]]
        ),
    )
    with service(provider) as (state, ws):
        ws.send_json({"type": "message.send", "id": "m1", "text": "go", "source": "desktop"})

        def deny(request: dict[str, Any]) -> None:
            ws.send_json(
                {
                    "type": "approval.respond",
                    "approval_id": request["approval_id"],
                    "decision": "deny",
                }
            )

        seen = until_agent_final(ws, on_approval=deny)

    agent = next(e["agent"] for e in seen if e["type"] == "agent.updated")
    label = {"id": agent["id"], "name": "reader"}
    agent_session = agent["session_id"]
    of_agent = [e for e in seen if e.get("session_id") == agent_session and "turn_id" in e]
    of_main = [e for e in seen if e.get("session_id") == agent["parent_session_id"]]
    kinds = {e["type"] for e in of_agent}
    assert {"turn.started", "tool.started", "tool.finished", "approval.request"} <= kinds
    assert {"approval.resolved", "text.delta", "turn.finished"} <= kinds
    assert all(e["agent"] == label for e in of_agent)
    assert of_main and all("agent" not in e for e in of_main)
    resolved = next(e for e in of_agent if e["type"] == "approval.resolved")
    assert resolved["decision"] == "deny"
    statuses = [e["agent"]["status"] for e in seen if e["type"] == "agent.updated"]
    assert statuses == ["working", "waiting_user", "working", "done"]
