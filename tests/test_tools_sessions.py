"""Tests for the tools that let the model look at other conversations."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from tests.conftest import FrozenClock
from tests.turns import FakeApprover, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.paths import db_path
from zero.protocol.messages import TextPart
from zero.providers.base import ToolCall
from zero.store.db import Database
from zero.store.journal import note_time
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.builtin.sessions import SessionStores
from zero.tools.permissions import PathGuard


@dataclass
class Rig:
    stores: SessionStores

    async def call(self, tool: str, in_session: str = "here", **tool_input: Any) -> ToolResult:
        config = Config()
        ctx = ToolContext(
            session_id=in_session,
            turn_id="t1",
            client_id="c1",
            cancel=CancelToken(),
            config=config,
            paths=PathGuard.from_config(config.permissions),
            approvals=FakeApprover("allow"),
        )
        registry = builtin_registry(sessions=self.stores)
        return await registry.execute(ToolCall(id="s", name=tool, input=tool_input), ctx)

    async def talk(self, session_id: str, *pairs: tuple[str, str]) -> None:
        messages = self.stores.messages
        for question, answer in pairs:
            await messages.append(session_id, "user", [TextPart(text=question)])
            call = {"id": "c", "name": "read_file", "input": {"path": "x"}}
            await messages.append(session_id, "assistant", [], tool_calls=[call])
            await messages.append(
                session_id, "tool", [TextPart(text="file text")], tool_call_id="c"
            )
            await messages.append(session_id, "assistant", [TextPart(text=answer)], status="ok")


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield Rig(SessionStores(SessionsRepo(db), MessagesRepo(db), SummariesRepo(db)))


async def test_sessions_list_shows_other_conversations_newest_first(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    here = await rig.stores.sessions.create("This one", session_id="here")
    trip = await rig.stores.sessions.create("Trip to Lisbon")
    frozen_clock.advance(60)
    agent = await rig.stores.sessions.create("log reader", kind="agent")

    result = await rig.call("sessions_list", here.id)

    when = note_time(agent.updated_at)
    lines = text_of(result).splitlines()
    assert lines[0] == f"- {agent.id} · log reader · agent · {when}"
    assert lines[1].startswith(f"- {trip.id} · Trip to Lisbon · chat · ")
    assert "This one" not in text_of(result)
    assert result.summary == "2 conversations"


async def test_sessions_read_gives_the_plain_turns(rig: Rig) -> None:
    session = await rig.stores.sessions.create("Trip")
    await rig.talk(session.id, ("When is the train?", "At 9:15."), ("And back?", "At 18:40."))

    result = await rig.call("sessions_read", session_id=session.id)

    assert text_of(result) == (
        "Trip\nUser: When is the train?\nZero: At 9:15.\nUser: And back?\nZero: At 18:40."
    )
    assert "file text" not in text_of(result)


async def test_sessions_read_shows_the_summary_of_what_is_left_out(rig: Rig) -> None:
    session = await rig.stores.sessions.create("Long")
    await rig.talk(session.id, *[(f"q{n}", f"a{n}") for n in range(5)])
    await rig.stores.summaries.save(session.id, "They planned a trip.", "m0")

    result = await rig.call("sessions_read", session_id=session.id, last=2)

    assert text_of(result) == (
        "Long\n[8 earlier lines not shown; summary of the start: They planned a trip.]\n"
        "User: q4\nZero: a4"
    )


async def test_sessions_read_of_an_unknown_session(rig: Rig) -> None:
    result = await rig.call("sessions_read", session_id="nope")
    assert result.is_error
    assert "sessions_list" in text_of(result)


async def test_sessions_search_finds_words_in_other_conversations(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    here = await rig.stores.sessions.create("Here", session_id="here")
    await rig.talk(here.id, ("the train", "here too"))
    trip = await rig.stores.sessions.create("Trip")
    await rig.talk(trip.id, ("When is the TRAIN to Lisbon?", "At 9:15."))

    result = await rig.call("sessions_search", here.id, query="train")
    nothing = await rig.call("sessions_search", here.id, query="ferry")

    assert text_of(result).splitlines() == [
        f"- {trip.id} · Trip · {note_time(frozen_clock())} · User: When is the TRAIN to Lisbon?"
    ]
    assert text_of(nothing) == "No other conversation mentions that."
