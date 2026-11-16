"""Agents that think through the user's Claude Code."""

import time
from collections.abc import AsyncIterator

import pytest

from tests.agents import RunnerKit
from tests.test_turn_text import connected, running, send, until_finished
from zero.agents.runner import AgentError
from zero.paths import db_path
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.store.db import Database


@pytest.fixture
async def kit() -> AsyncIterator[RunnerKit]:
    async with Database(db_path()) as db:
        await db.migrate()
        kit = RunnerKit(db)
        yield kit
        await kit.runner.close()
        await kit.turns.cancel_all(grace_s=1)


async def test_no_claude_agent_without_claude_code(kit: RunnerKit) -> None:
    parent = await kit.sessions.create()
    with pytest.raises(AgentError, match="Claude Code is not installed"):
        await kit.runner.start("coder", "Write it", parent.id, claude=True)
    assert await kit.agents.list(parent.id) == []


def test_a_claude_agent_thinks_through_claude_code() -> None:
    start = ToolCall(
        id="c1",
        name="agent_start",
        input={"name": "coder", "task": "Write the parser", "claude": True},
    )
    main = FakeProvider([[ToolCallEvent(call=start)], [TextDelta(text="Started.")]] + [[]] * 4)
    claude = FakeProvider([[TextDelta(text="Parser written by Claude.")]])
    with running(main) as (client, state), connected(client, state) as ws:
        state.claude = claude
        send(ws, "Have Claude write the parser")
        until_finished(ws)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not client.portal.call(state.agents.active) and claude.calls:
                break
            time.sleep(0.05)
    assert len(claude.calls) == 1
    assert "Write the parser" in str(claude.calls[0].messages[-1].content)
    assert all("Write the parser" not in str(call.messages) for call in main.calls[:1])
