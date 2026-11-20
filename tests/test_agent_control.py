"""Tests of talking to agents: messages, status, stop, questions and reports to the parent."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from phantasm.agents.runner import LAST_OUTPUT_CHARS, AgentError
from phantasm.agents.types import AgentProgress
from phantasm.config import Config
from phantasm.core.turn import Turn
from phantasm.paths import db_path
from phantasm.protocol.messages import SessionDeleted, TextPart, TurnFinished, Usage
from phantasm.providers.base import TextDelta, ToolCall, ToolCallEvent
from phantasm.providers.fake import FakeProvider
from phantasm.server.app import create_app
from phantasm.server.state import AppState, get_state
from phantasm.store.agents import Agent
from phantasm.store.db import Database
from phantasm.tools.builtin.agents import (
    AgentIdInput,
    AgentMessage,
    AgentMessageInput,
    AgentStatusTool,
    AgentStop,
    AskParent,
    AskParentInput,
)
from tests.agents import RoleProvider, RunnerKit
from tests.turns import text_of, until

HELLO = {"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}}


@pytest.fixture
async def kit() -> AsyncIterator[RunnerKit]:
    async with Database(db_path()) as db:
        await db.migrate()
        kit = RunnerKit(db)
        yield kit
        await kit.runner.close()
        await kit.turns.cancel_all(grace_s=1)


async def new_agent(kit: RunnerKit, name: str = "reader") -> tuple[str, Agent]:
    parent = await kit.sessions.create()
    return parent.id, await kit.runner.start(name, "Read the notes", parent.id)


def ended(status: str, text: str = "", error: str | None = None) -> TurnFinished:
    return TurnFinished(
        session_id="s", turn_id="t", status=status, text=text, usage=Usage(), error=error
    )


def endless_turn(kit: RunnerKit, session_id: str) -> Turn:
    """A turn that runs until it is cancelled, to keep a session busy."""
    return Turn(
        session_id=session_id,
        text="busy",
        source="desktop",
        client=None,
        config=Config(),
        provider=FakeProvider(infinite=True),
        sessions=kit.sessions,
        messages=kit.messages,
        events=kit.runner._events,
    )


async def test_a_message_puts_the_agent_back_to_work(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    await kit.runner.finished(agent.id, ended("ok", "First report."))

    updated = await kit.runner.message(agent.id, "Now count the lines", parent)

    assert updated.status == "working"
    assert kit.started[-1] == (agent.id, "Now count the lines")
    assert kit.updates[-1] == ("working", "First report.")


async def test_messages_reach_only_free_agents_of_this_conversation(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    stranger = await kit.sessions.create()
    assert agent.session_id is not None

    with pytest.raises(AgentError, match="no agent with id"):
        await kit.runner.message(agent.id, "hi", stranger.id)
    with pytest.raises(AgentError, match="no agent with id"):
        await kit.runner.message("nope", "hi", parent)

    kit.turns.start(endless_turn(kit, agent.session_id))
    with pytest.raises(AgentError, match="is still working"):
        await kit.runner.message(agent.id, "hi", parent)
    await kit.turns.stop_session(agent.session_id)

    await kit.runner.stop(agent.id, parent)
    with pytest.raises(AgentError, match="was stopped"):
        await kit.runner.message(agent.id, "hi", parent)


async def test_progress_counts_steps_and_shows_the_last_words(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    assert agent.session_id is not None
    say = kit.messages.append
    await say(agent.session_id, "user", [TextPart(text="Read the notes")], status="ok")
    await say(agent.session_id, "assistant", [TextPart(text="Looking.")], status="ok")
    call = [{"id": "c", "name": "read_file", "input": {}}]
    await say(agent.session_id, "assistant", [], tool_calls=call, status="ok")
    await say(agent.session_id, "tool", [TextPart(text="file text")], tool_call_id="c")
    await say(agent.session_id, "assistant", [TextPart(text="x" * 2000)], status="ok")

    progress = await kit.runner.progress(agent.id, parent)

    assert (progress.agent.id, progress.steps) == (agent.id, 3)
    assert progress.last_output is not None
    assert len(progress.last_output) == LAST_OUTPUT_CHARS
    assert progress.last_output.startswith("…x")


async def test_progress_of_a_new_or_stopped_agent(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)

    assert (await kit.runner.progress(agent.id, parent)).steps == 0
    await kit.runner.stop(agent.id, parent)
    stopped = await kit.runner.progress(agent.id, parent)
    assert (stopped.agent.status, stopped.steps, stopped.last_output) == ("stopped", 0, None)


async def test_stop_cancels_the_work_and_deletes_the_session(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    assert agent.session_id is not None
    turn = endless_turn(kit, agent.session_id)
    kit.turns.start(turn)
    await until(lambda: kit.turns.busy(agent.session_id or ""))

    stopped = await kit.runner.stop(agent.id, parent)

    assert (stopped.status, stopped.session_id) == ("stopped", None)
    assert turn.finished is not None and turn.finished.status == "cancelled"
    assert await kit.sessions.get(agent.session_id) is None
    assert SessionDeleted(session_id=agent.session_id) in kit.client.received
    assert kit.updates[-2:] == [("stopped", None), ("stopped", None)]
    assert kit.notices == []
    assert (await kit.runner.stop(agent.id, parent)).status == "stopped"


async def test_stopping_a_finished_agent_keeps_its_result(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    await kit.runner.finished(agent.id, ended("ok", "Report."))

    stopped = await kit.runner.stop(agent.id, parent)

    assert (stopped.status, stopped.report, stopped.session_id) == ("done", "Report.", None)


async def test_a_question_goes_to_the_parent_and_the_agent_waits(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    assert agent.session_id is not None

    asking = await kit.runner.ask(agent.session_id, "  Which folder?  ")
    await asyncio.sleep(0.05)
    assert kit.notices == []
    await kit.runner.finished(agent.id, ended("ok", "I asked."))
    await until(lambda: len(kit.notices) == 1)

    assert asking.status == "asking"
    assert kit.notices == [
        (
            parent,
            f"Agent 'reader' (id {agent.id}) asks:\nWhich folder?\n\n"
            "Answer it with agent_message, or ask the user first if only they know.",
        )
    ]
    stored = await kit.agents.get(agent.id)
    assert stored is not None and stored.status == "asking"


async def test_questions_of_one_turn_go_together(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    assert agent.session_id is not None

    await kit.runner.ask(agent.session_id, "Which folder?")
    await kit.runner.ask(agent.session_id, "Which format?")
    await kit.runner.finished(agent.id, ended("ok"))
    await until(lambda: len(kit.notices) == 1)

    assert "asks:\nWhich folder?\n\nWhich format?\n\nAnswer it" in kit.notices[0][1]


async def test_an_asking_agent_cancelled_meanwhile_asks_nothing(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    assert agent.session_id is not None

    await kit.runner.ask(agent.session_id, "Which folder?")
    await kit.runner.finished(agent.id, ended("cancelled"))
    await asyncio.sleep(0.05)

    stored = await kit.agents.get(agent.id)
    assert stored is not None and stored.status == "stopped"
    assert kit.notices == []


async def test_only_agents_ask_parents(kit: RunnerKit) -> None:
    ordinary = await kit.sessions.create()

    with pytest.raises(AgentError, match="only agents"):
        await kit.runner.ask(ordinary.id, "?")


@pytest.mark.parametrize(
    ("finished", "notice"),
    [
        (ended("ok", " All read. "), "has finished. Its report:\nAll read."),
        (ended("error", error="overloaded"), "failed: overloaded"),
    ],
)
async def test_reports_and_failures_go_to_the_parent(
    kit: RunnerKit, finished: TurnFinished, notice: str
) -> None:
    parent, agent = await new_agent(kit)

    await kit.runner.finished(agent.id, finished)
    await until(lambda: len(kit.notices) == 1)

    assert kit.notices == [(parent, f"Agent 'reader' (id {agent.id}) {notice}")]


async def test_a_notice_waits_until_the_parent_is_free(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    kit.turns.start(endless_turn(kit, parent))
    await until(lambda: kit.turns.busy(parent))

    await kit.runner.finished(agent.id, ended("ok", "Report."))
    await asyncio.sleep(0.05)
    assert kit.notices == []

    await kit.turns.stop_session(parent)
    await until(lambda: len(kit.notices) == 1)
    assert kit.notices[0][0] == parent


async def test_closing_drops_the_notices_still_waiting(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    kit.turns.start(endless_turn(kit, parent))
    await until(lambda: kit.turns.busy(parent))
    await kit.runner.finished(agent.id, ended("ok", "Report."))

    await kit.runner.close()
    await kit.turns.stop_session(parent)
    await asyncio.sleep(0.05)

    assert kit.notices == []
    other_parent, other = await new_agent(kit, "late")
    await kit.runner.finished(other.id, ended("ok", "Too late."))
    await asyncio.sleep(0.05)
    assert kit.notices == []


async def test_no_notice_for_a_parent_deleted_meanwhile(kit: RunnerKit) -> None:
    parent, agent = await new_agent(kit)
    kit.turns.start(endless_turn(kit, parent))
    await until(lambda: kit.turns.busy(parent))
    await kit.runner.finished(agent.id, ended("ok", "Report."))

    await kit.turns.stop_session(parent)
    await kit.sessions.delete(parent)
    await asyncio.sleep(0.05)

    assert kit.notices == []


class Control:
    """Answers the tools like the runner would, or refuses with ``error``."""

    def __init__(self, agent: Agent, error: str | None = None) -> None:
        self.agent = agent
        self.error = error
        self.calls: list[tuple[str, ...]] = []

    def _check(self, *call: str) -> None:
        self.calls.append(call)
        if self.error:
            raise AgentError(self.error)

    async def message(self, agent_id: str, text: str, parent_session_id: str) -> Agent:
        self._check("message", agent_id, text, parent_session_id)
        return self.agent

    async def progress(self, agent_id: str, parent_session_id: str) -> AgentProgress:
        self._check("progress", agent_id, parent_session_id)
        return AgentProgress(self.agent, 4, "Reading main.py")

    async def stop(self, agent_id: str, parent_session_id: str) -> Agent:
        self._check("stop", agent_id, parent_session_id)
        return self.agent

    async def ask(self, session_id: str, question: str) -> Agent:
        self._check("ask", session_id, question)
        return self.agent


@pytest.fixture
async def sample(kit: RunnerKit) -> Agent:
    _, agent = await new_agent(kit)
    done = await kit.agents.set_status(agent.id, "done", "Three files.")
    assert done is not None
    return done


CTX = SimpleNamespace(session_id="parent")


async def test_tools_pass_the_calling_session(sample: Agent) -> None:
    control = Control(sample)

    await AgentMessage(control).run(AgentMessageInput(id="a", text="go"), CTX)  # type: ignore[arg-type]
    await AgentStop(control).run(AgentIdInput(id="a"), CTX)  # type: ignore[arg-type]
    await AskParent(control).run(AskParentInput(question="?"), CTX)  # type: ignore[arg-type]

    assert control.calls == [
        ("message", "a", "go", "parent"),
        ("stop", "a", "parent"),
        ("ask", "parent", "?"),
    ]


async def test_status_tool_shows_everything_known(sample: Agent) -> None:
    out = await AgentStatusTool(Control(sample)).run(AgentIdInput(id="a"), CTX)  # type: ignore[arg-type]

    assert out.summary == "done"
    assert text_of(SimpleNamespace(content=out.content)) == (
        f"Agent 'reader' (id {sample.id}): done\n"
        "Task: Read the notes\n"
        "Steps so far: 4\n"
        "Last output:\nReading main.py\n"
        "Report:\nThree files."
    )


@pytest.mark.parametrize(
    ("tool", "params", "prefix"),
    [
        (AgentMessage, AgentMessageInput(id="a", text="go"), "The message was not sent"),
        (AgentStatusTool, AgentIdInput(id="a"), "No status"),
        (AgentStop, AgentIdInput(id="a"), "Nothing was stopped"),
        (AskParent, AskParentInput(question="?"), "The question was not sent"),
    ],
)
async def test_tools_explain_refusals(sample: Agent, tool: Any, params: Any, prefix: str) -> None:
    out = await tool(Control(sample, error="no luck")).run(params, CTX)

    assert out.is_error
    assert text_of(SimpleNamespace(content=out.content)) == f"{prefix}: no luck."


@contextmanager
def service(provider: FakeProvider) -> Iterator[tuple[AppState, WebSocketTestSession]]:
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json(HELLO)
            assert ws.receive_json()["type"] == "hello.ok"
            yield state, ws


def until_event(ws: WebSocketTestSession, check: Any) -> list[dict[str, Any]]:
    seen = []
    while True:
        seen.append(event := ws.receive_json())
        if check(event):
            return seen


def call(tool: str, **tool_input: Any) -> list[Any]:
    return [ToolCallEvent(call=ToolCall(id=f"{tool}-1", name=tool, input=tool_input))]


def test_the_report_comes_to_the_parent_as_an_agent_turn() -> None:
    provider = RoleProvider(
        main=FakeProvider(
            [
                call("agent_start", name="reader", task="Count the files"),
                [TextDelta(text="Started.")],
                [TextDelta(text="The reader found 3 files.")],
            ]
        ),
        agent=FakeProvider([[TextDelta(text="3 files.")]]),
    )
    with service(provider) as (state, ws):
        ws.send_json({"type": "message.send", "id": "m1", "text": "count", "source": "desktop"})
        seen = until_event(
            ws, lambda e: e["type"] == "turn.finished" and e["text"] == "The reader found 3 files."
        )

        main_session = seen[-1]["session_id"]
        started = [e for e in seen if e["type"] == "turn.started"]
        assert (started[-1]["session_id"], started[-1]["source"]) == (main_session, "agent")
        notice = provider.main.calls[2].messages[-1]
        assert notice.role == "user"
        assert text_of(notice).endswith("has finished. Its report:\n3 files.")
        assert "started by another agent" in text_of(provider.main.calls[2].messages[0])
