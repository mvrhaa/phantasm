"""Tests for session memory: what the model keeps for one conversation only."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

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
from zero.store.memory import MemoryRepo
from zero.store.messages import MessagesRepo
from zero.store.session_memory import SessionMemoryRepo
from zero.store.sessions import SessionsRepo
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.permissions import PathGuard


@dataclass
class Stores:
    sessions: SessionsRepo
    messages: MessagesRepo
    memory: MemoryRepo
    session_memory: SessionMemoryRepo
    a: str
    b: str


@pytest.fixture
async def stores() -> AsyncIterator[Stores]:
    async with Database(db_path()) as db:
        await db.migrate()
        sessions = SessionsRepo(db)
        a = await sessions.create()
        b = await sessions.create()
        yield Stores(sessions, MessagesRepo(db), MemoryRepo(db), SessionMemoryRepo(db), a.id, b.id)


async def use(stores: Stores, tool: str, session_id: str, **tool_input: Any) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id=session_id,
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=FakeApprover("allow"),
    )
    registry = builtin_registry(stores.memory, session_memory=stores.session_memory)
    return await registry.execute(ToolCall(id="m", name=tool, input=tool_input), ctx)


async def test_session_scope_keeps_a_fact_for_this_conversation_only(stores: Stores) -> None:
    result = await use(
        stores, "memory_save", stores.a, text="Answer in English here", scope="session"
    )

    assert not result.is_error
    assert "for this conversation" in text_of(result)
    assert [entry.text for entry in await stores.session_memory.list(stores.a)] == [
        "Answer in English here"
    ]
    assert await stores.session_memory.list(stores.b) == []
    assert await stores.memory.top() == []


async def test_lasting_memory_is_the_default(stores: Stores) -> None:
    await use(stores, "memory_save", stores.a, text="The user has a cat named Milk")
    assert [memory.text for memory in await stores.memory.top()] == [
        "The user has a cat named Milk"
    ]
    assert await stores.session_memory.list(stores.a) == []


async def test_the_same_session_fact_is_kept_once(stores: Stores) -> None:
    for _ in range(2):
        await use(stores, "memory_save", stores.a, text="Use tabs", scope="session")
    assert len(await stores.session_memory.list(stores.a)) == 1


async def test_memory_delete_forgets_a_session_entry(stores: Stores) -> None:
    entry = await stores.session_memory.add(stores.a, "Do not push yet")
    result = await use(stores, "memory_delete", stores.a, id=entry.id)
    assert text_of(result) == "Forgot: Do not push yet"
    assert await stores.session_memory.list(stores.a) == []


async def test_session_memory_goes_with_its_session(stores: Stores) -> None:
    await stores.session_memory.add(stores.a, "Staging is at 10.0.0.5")
    await stores.sessions.delete(stores.a)
    assert await stores.session_memory.list(stores.a) == []


def test_the_prompt_lists_session_memory_after_memory() -> None:
    from datetime import UTC, datetime

    prompt = build_system_prompt(
        None,
        "desktop",
        ["The user is Alex"],
        session_memories=["[sm-1] Answer in English here"],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert prompt.index("# Memory") < prompt.index("# This conversation's memory")
    assert "- [sm-1] Answer in English here" in prompt


async def test_each_session_sees_only_its_own_memory(stores: Stores) -> None:
    await stores.session_memory.add(stores.a, "Answer in English here")

    def builder(session_id: str) -> PromptBuilder:
        return PromptBuilder(
            session_id=session_id,
            client=None,
            source="desktop",
            config=Config(),
            provider=FakeProvider(),
            messages=stores.messages,
            events=EventBus(Audience([RecordingClient()])),
            memory=stores.memory,
            session_memory=stores.session_memory,
        )

    own = (await builder(stores.a).build())[0].content[0].text  # type: ignore[union-attr]
    other = (await builder(stores.b).build())[0].content[0].text  # type: ignore[union-attr]
    assert "Answer in English here" in own
    assert "# This conversation's memory" not in other


def test_the_rules_explain_the_session_scope() -> None:
    from zero.core.context import _base_rules

    assert 'scope "session"' in _base_rules()
