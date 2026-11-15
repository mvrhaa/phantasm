"""Tests for going over quiet conversations: a journal note and lasting memories."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from tests.conftest import FrozenClock
from zero.core.review import QUIET_S, RETRY_AFTER_S, ReviewRunner
from zero.paths import db_path
from zero.protocol.messages import TextPart
from zero.providers.base import StreamEvent, TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.store.db import Database
from zero.store.journal import JournalRepo
from zero.store.memory import MemoryRepo
from zero.store.messages import MessagesRepo
from zero.store.reviews import ReviewsRepo
from zero.store.sessions import SessionsRepo


class Writer(FakeProvider):
    """A scripted provider that may write prose, as a real model can."""

    writes_prose = True


def save(text: str, n: int = 1) -> ToolCallEvent:
    return ToolCallEvent(call=ToolCall(id=f"c{n}", name="memory_save", input={"text": text}))


@dataclass
class Rig:
    sessions: SessionsRepo
    messages: MessagesRepo
    memory: MemoryRepo
    journal: JournalRepo
    reviews: ReviewsRepo
    busy: set[str] = field(default_factory=set)

    def runner(self, provider: FakeProvider, enabled: bool = True) -> ReviewRunner:
        return ReviewRunner(
            sessions=self.sessions,
            messages=self.messages,
            memory=self.memory,
            journal=self.journal,
            reviews=self.reviews,
            provider=lambda: provider,
            model=lambda: "small-1",
            busy=lambda session_id: session_id in self.busy,
            enabled=lambda: enabled,
            retry_delays=[0],
        )

    async def talk(self, session_id: str, *texts: str) -> None:
        for text in texts:
            await self.messages.append(session_id, "user", [TextPart(text=text)])
            await self.messages.append(
                session_id, "assistant", [TextPart(text=f"ok: {text}")], status="ok"
            )


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield Rig(
            SessionsRepo(db), MessagesRepo(db), MemoryRepo(db), JournalRepo(db), ReviewsRepo(db)
        )


async def test_a_quiet_conversation_gets_a_note_and_its_facts(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "I moved to Prague last month", "find me a dentist nearby")
    frozen_clock.advance(QUIET_S)
    provider = Writer(
        [[TextDelta(text="Looked for a dentist in Prague."), save("The user lives in Prague")]]
    )

    reviewed = await rig.runner(provider).tick()

    assert reviewed == [session.id]
    assert [note.text for note in await rig.journal.list()] == ["Looked for a dentist in Prague."]
    assert [memory.text for memory in await rig.memory.top()] == ["The user lives in Prague"]
    (call,) = provider.calls
    assert call.model == "small-1"
    assert [tool.name for tool in call.tools] == ["memory_save"]
    asked = call.messages[-1].content[0].text  # type: ignore[union-attr]
    assert "User: I moved to Prague last month" in asked
    assert await rig.runner(provider).tick() == []


async def test_only_quiet_conversations_with_enough_said_are_reviewed(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    short = await rig.sessions.create()
    await rig.talk(short.id, "hi")
    recent = await rig.sessions.create()
    await rig.talk(recent.id, "one", "two")
    busy = await rig.sessions.create()
    await rig.talk(busy.id, "one", "two")
    agent = await rig.sessions.create(kind="agent")
    await rig.talk(agent.id, "one", "two")
    rig.busy.add(busy.id)
    frozen_clock.advance(QUIET_S - 1)
    provider = Writer([])

    assert await rig.runner(provider).tick() == []
    frozen_clock.advance(1)
    await rig.talk(recent.id, "three")
    assert await rig.runner(provider).tick() == []
    assert provider.calls == []


async def test_only_the_new_part_is_reviewed_next_time(rig: Rig, frozen_clock: FrozenClock) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "first question", "second question")
    frozen_clock.advance(QUIET_S)
    provider = Writer([[TextDelta(text="Two questions.")], [TextDelta(text="Two more.")]])
    await rig.runner(provider).tick()

    await rig.talk(session.id, "third question", "fourth question")
    frozen_clock.advance(QUIET_S)
    assert await rig.runner(provider).tick() == [session.id]

    asked = provider.calls[-1].messages[-1].content[0].text  # type: ignore[union-attr]
    assert "third question" in asked
    assert "first question" not in asked


async def test_nothing_to_note_writes_no_note(rig: Rig, frozen_clock: FrozenClock) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "hi", "thanks")
    frozen_clock.advance(QUIET_S)
    provider = Writer([[TextDelta(text="…")], [TextDelta(text="unused")]])

    assert await rig.runner(provider).tick() == [session.id]
    assert await rig.journal.list() == []
    assert await rig.runner(provider).tick() == []


async def test_at_most_five_facts_and_only_valid_ones(rig: Rig, frozen_clock: FrozenClock) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "a", "b")
    frozen_clock.advance(QUIET_S)
    calls: list[StreamEvent] = [save(f"fact {n}", n) for n in range(7)]
    calls.insert(0, ToolCallEvent(call=ToolCall(id="x", name="memory_save", input={})))
    calls.insert(0, ToolCallEvent(call=ToolCall(id="y", name="run_command", input={})))
    provider = Writer([[TextDelta(text="Note."), *calls]])

    await rig.runner(provider).tick()

    assert sorted(memory.text for memory in await rig.memory.top(20)) == [
        f"fact {n}" for n in range(5)
    ]


async def test_a_failed_review_waits_before_trying_again(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "a", "b")
    frozen_clock.advance(QUIET_S)
    provider = Writer([[TextDelta(text="Note.")]], fail_times=1, fail_retryable=False)
    runner = rig.runner(provider)

    assert await runner.tick() == []
    assert await runner.tick() == []
    frozen_clock.advance(RETRY_AFTER_S)
    assert await runner.tick() == [session.id]
    assert [note.text for note in await rig.journal.list()] == ["Note."]


async def test_no_review_without_a_model_or_when_turned_off(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    session = await rig.sessions.create()
    await rig.talk(session.id, "a", "b")
    frozen_clock.advance(QUIET_S)
    echo = FakeProvider()
    writer = Writer([[TextDelta(text="Note.")]])

    assert await rig.runner(echo).tick() == []
    assert await rig.runner(writer, enabled=False).tick() == []
    assert echo.calls == writer.calls == []


def test_the_app_goes_over_conversations_while_it_runs() -> None:
    from fastapi.testclient import TestClient

    from zero.config import Config
    from zero.server.app import create_app
    from zero.server.state import get_state

    app = create_app(Config())
    with TestClient(app):
        reviewer = get_state(app).reviewer
        assert reviewer.running
    assert not reviewer.running
    assert Config().agent.review_conversations is True
