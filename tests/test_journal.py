"""Tests for the journal: short notes about what happened in conversations."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.conftest import FrozenClock
from tests.turns import Audience, FakeApprover, RecordingClient, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.core.context import build_system_prompt
from zero.core.events import EventBus
from zero.core.prompting import PromptBuilder
from zero.paths import db_path
from zero.providers.base import ToolCall
from zero.providers.fake import FakeProvider
from zero.store.db import Database
from zero.store.journal import JOURNAL_NOTES, JournalRepo, note_time
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.permissions import PathGuard


@dataclass
class Stores:
    sessions: SessionsRepo
    messages: MessagesRepo
    journal: JournalRepo


@pytest.fixture
async def stores() -> AsyncIterator[Stores]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield Stores(SessionsRepo(db), MessagesRepo(db), JournalRepo(db))


async def test_only_the_newest_notes_are_kept(stores: Stores, frozen_clock: FrozenClock) -> None:
    for n in range(JOURNAL_NOTES + 2):
        await stores.journal.add("s1", f"note {n}")
        frozen_clock.advance(60)
    notes = await stores.journal.list()
    assert len(notes) == JOURNAL_NOTES
    assert notes[0].text == "note 2"
    assert notes[-1].text == f"note {JOURNAL_NOTES + 1}"


async def test_newest_from_other_sessions_fit_the_size(
    stores: Stores, frozen_clock: FrozenClock
) -> None:
    await stores.journal.add("other", "x" * 300)
    frozen_clock.advance(60)
    await stores.journal.add("own", "my own talk")
    frozen_clock.advance(60)
    await stores.journal.add("other", "planned the trip to Lisbon")
    frozen_clock.advance(60)
    await stores.journal.add("third", "fixed the backup script")

    shown = await stores.journal.newest(exclude_session="own", max_chars=100)

    assert [note.text for note in shown] == [
        "planned the trip to Lisbon",
        "fixed the backup script",
    ]


async def test_search_finds_any_word_newest_first(
    stores: Stores, frozen_clock: FrozenClock
) -> None:
    await stores.journal.add("a", "Talked about the Python release")
    frozen_clock.advance(60)
    await stores.journal.add("b", "Booked a table")
    frozen_clock.advance(60)
    await stores.journal.add("c", "python tests were fixed")

    found = await stores.journal.search("PYTHON trip")

    assert [note.text for note in found] == [
        "python tests were fixed",
        "Talked about the Python release",
    ]


async def search_tool(stores: Stores, **tool_input: Any) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=FakeApprover("allow"),
    )
    registry = builtin_registry(journal=stores.journal)
    return await registry.execute(ToolCall(id="j", name="journal_search", input=tool_input), ctx)


async def test_journal_search_tool(stores: Stores, frozen_clock: FrozenClock) -> None:
    await stores.journal.add("a", "Planned the trip to Lisbon")

    found = await search_tool(stores, query="trip")
    nothing = await search_tool(stores, query="cats")

    when = note_time(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    assert text_of(found) == f"- [{when}] Planned the trip to Lisbon"
    assert found.summary == "1 note"
    assert text_of(nothing) == "No journal notes match."


def test_the_journal_block_comes_after_memory() -> None:
    prompt = build_system_prompt(
        None,
        "desktop",
        ["The user is Alex"],
        journal=["[2026-01-01 12:00] Planned the trip"],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert prompt.index("# Memory") < prompt.index("# Lately in your other conversations")
    assert "- [2026-01-01 12:00] Planned the trip" in prompt


async def test_the_prompt_shows_other_sessions_notes(
    stores: Stores, frozen_clock: FrozenClock
) -> None:
    own = await stores.sessions.create()
    await stores.journal.add(own.id, "this very talk")
    await stores.journal.add("other", "Planned the trip")

    def builder(agent_name: str | None = None) -> PromptBuilder:
        return PromptBuilder(
            session_id=own.id,
            client=None,
            source="desktop",
            config=Config(),
            provider=FakeProvider(),
            messages=stores.messages,
            events=EventBus(Audience([RecordingClient()])),
            journal=stores.journal,
            agent_name=agent_name,
        )

    text = (await builder().build())[0].content[0].text  # type: ignore[union-attr]
    agent = (await builder("sorter").build())[0].content[0].text  # type: ignore[union-attr]
    assert "Planned the trip" in text
    assert "this very talk" not in text
    assert "# Lately in your other conversations" not in agent
