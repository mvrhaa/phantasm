"""Tests of the todo list: the repository, the tools and the todo.updated event."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.turns import Audience, Harness, RecordingClient, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.core.events import EventBus
from zero.paths import db_path
from zero.protocol.messages import TodoItem, TodoUpdated
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.store.db import Database
from zero.store.sessions import SessionsRepo
from zero.store.todo import TodoRepo
from zero.tools.base import ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.permissions import PathGuard


@pytest.fixture
async def stores() -> AsyncIterator[tuple[TodoRepo, SessionsRepo]]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield TodoRepo(db), SessionsRepo(db)


async def test_items_are_numbered_per_session(stores: tuple[TodoRepo, SessionsRepo]) -> None:
    todo, sessions = stores
    first, second = await sessions.create(), await sessions.create()

    await todo.add(first.id, "buy milk")
    await todo.add(first.id, "  call   mom ")
    await todo.add(second.id, "other session")

    items = [(t.number, t.text, t.status) for t in await todo.list(first.id)]
    assert items == [(1, "buy milk", "pending"), (2, "call mom", "pending")]
    assert [t.number for t in await todo.list(second.id)] == [1]


async def test_status_changes(stores: tuple[TodoRepo, SessionsRepo], frozen_clock: Any) -> None:
    todo, sessions = stores
    session = await sessions.create()
    added = await todo.add(session.id, "buy milk")
    frozen_clock.advance(5)

    changed = await todo.set_status(session.id, 1, "done")

    assert changed is not None
    assert (changed.status, changed.updated_at > added.updated_at) == ("done", True)
    assert await todo.set_status(session.id, 9, "done") is None


async def test_empty_text_is_refused(stores: tuple[TodoRepo, SessionsRepo]) -> None:
    todo, sessions = stores
    session = await sessions.create()

    with pytest.raises(ValueError, match="needs some text"):
        await todo.add(session.id, "   ")


async def test_deleting_a_session_deletes_its_list(stores: tuple[TodoRepo, SessionsRepo]) -> None:
    todo, sessions = stores
    session = await sessions.create()
    await todo.add(session.id, "buy milk")

    await sessions.delete(session.id)

    assert await todo.list(session.id) == []


async def use(
    todo: TodoRepo, session_id: str, client: RecordingClient, tool: str, **tool_input: Any
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id=session_id,
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        events=EventBus(Audience([client])),
    )
    return await builtin_registry(todo=todo).execute(
        ToolCall(id="c", name=tool, input=tool_input), ctx
    )


async def test_tools_keep_the_list_and_tell_the_clients(
    stores: tuple[TodoRepo, SessionsRepo],
) -> None:
    todo, sessions = stores
    session = await sessions.create()
    client = RecordingClient()

    await use(todo, session.id, client, "todo_add", text="read the spec")
    await use(todo, session.id, client, "todo_add", text="write the code")
    updated = await use(todo, session.id, client, "todo_update", number=1, status="done")
    listed = await use(todo, session.id, client, "todo_list")

    expected = "1. [done] read the spec\n2. [pending] write the code"
    assert text_of(updated) == text_of(listed) == expected
    assert updated.summary == "2 items, 1 done"
    assert [e.type for e in client.received] == ["todo.updated"] * 3
    assert client.received[-1] == TodoUpdated(
        session_id=session.id,
        items=[
            TodoItem(number=1, text="read the spec", status="done"),
            TodoItem(number=2, text="write the code", status="pending"),
        ],
    )


async def test_tool_errors(stores: tuple[TodoRepo, SessionsRepo]) -> None:
    todo, sessions = stores
    session = await sessions.create()
    client = RecordingClient()

    missing = await use(todo, session.id, client, "todo_update", number=3, status="done")
    bad_status = await use(todo, session.id, client, "todo_update", number=1, status="later")
    empty = await use(todo, session.id, client, "todo_list")

    assert text_of(missing) == "There is no item 3 in the todo list."
    assert "status: Input should be 'pending', 'in_progress' or 'done'" in text_of(bad_status)
    assert (text_of(empty), empty.summary) == ("The todo list is empty.", "0 items")
    assert client.received == []


def test_todo_tools_are_allowed_and_only_registered_with_a_store(
    stores: tuple[TodoRepo, SessionsRepo],
) -> None:
    todo, _ = stores
    registry = builtin_registry(todo=todo)

    assert "todo_add" not in [spec.name for spec in builtin_registry().specs()]
    policies = {
        name: registry.get(name).default_policy for name in ("todo_add", "todo_update", "todo_list")
    }  # type: ignore[union-attr]
    assert set(policies.values()) == {"allow"}


async def test_turn_publishes_todo_updates_on_the_bus(
    harness: Harness, stores: tuple[TodoRepo, SessionsRepo]
) -> None:
    todo, _ = stores
    add = ToolCallEvent(call=ToolCall(id="t1", name="todo_add", input={"text": "plan the trip"}))
    provider = FakeProvider([[add], [TextDelta(text="Planned.")]])

    await harness.run(provider, builtin_registry(todo=todo))

    assert harness.types == [
        "turn.started",
        "tool.started",
        "todo.updated",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    event = harness.client.received[2]
    assert isinstance(event, TodoUpdated)
    assert (event.session_id, [i.text for i in event.items]) == (
        harness.session_id,
        ["plan the trip"],
    )
