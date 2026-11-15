"""Tests for going back to a message: what came after it is undone where it can be."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FrozenClock
from zero.core.rewind import Rewinder, RewindError
from zero.paths import db_path
from zero.protocol.messages import TextPart
from zero.store.agents import AgentsRepo
from zero.store.db import Database
from zero.store.file_changes import FileChangesRepo
from zero.store.messages import MessagesRepo
from zero.store.reviews import ReviewsRepo
from zero.store.session_memory import SessionMemoryRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.store.todo import TodoRepo
from zero.store.tool_runs import ToolRunsRepo


@dataclass
class Rig:
    sessions: SessionsRepo
    messages: MessagesRepo
    changes: FileChangesRepo
    todos: TodoRepo
    session_memory: SessionMemoryRepo
    agents: AgentsRepo
    summaries: SummariesRepo
    reviews: ReviewsRepo
    tool_runs: ToolRunsRepo
    session_id: str
    busy: set[str] = field(default_factory=set)
    stopped: list[str] = field(default_factory=list)

    def rewinder(self) -> Rewinder:
        async def stop(agent_id: str, parent_session_id: str) -> None:
            self.stopped.append(agent_id)

        return Rewinder(
            sessions=self.sessions,
            messages=self.messages,
            changes=self.changes,
            todos=self.todos,
            session_memory=self.session_memory,
            agents=self.agents,
            summaries=self.summaries,
            reviews=self.reviews,
            tool_runs=self.tool_runs,
            busy=lambda session_id: session_id in self.busy,
            stop_agent=stop,
        )

    async def user(self, text: str) -> str:
        message = await self.messages.append(
            self.session_id, "user", [TextPart(text=text)], turn_id=f"t-{text}"
        )
        return message.id

    async def answer(self, text: str, *calls: dict[str, Any]) -> None:
        turn = "t-x"
        if calls:
            await self.messages.append(
                self.session_id, "assistant", [], turn_id=turn, tool_calls=list(calls)
            )
            for call in calls:
                await self.messages.append(
                    self.session_id,
                    "tool",
                    [TextPart(text="done")],
                    turn_id=turn,
                    tool_call_id=call["id"],
                )
        await self.messages.append(
            self.session_id, "assistant", [TextPart(text=text)], turn_id=turn, status="ok"
        )


def call(name: str, n: int = 1, **tool_input: Any) -> dict[str, Any]:
    return {"id": f"{name}-{n}", "name": name, "input": tool_input}


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    async with Database(db_path()) as db:
        await db.migrate()
        sessions = SessionsRepo(db)
        session = await sessions.create()
        yield Rig(
            sessions,
            MessagesRepo(db),
            FileChangesRepo(db),
            TodoRepo(db),
            SessionMemoryRepo(db),
            AgentsRepo(db),
            SummariesRepo(db),
            ReviewsRepo(db),
            ToolRunsRepo(db),
            session.id,
        )


async def test_the_message_and_what_followed_are_removed(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    await rig.user("first")
    await rig.answer("a1")
    frozen_clock.advance(60)
    second = await rig.user("second")
    await rig.answer("a2", call("read_file", path="~/x"))

    report = await rig.rewinder().rewind(rig.session_id, second)

    assert report.text == "second"
    assert report.removed_messages == 4
    left = await rig.messages.list(rig.session_id)
    assert [m.role for m in left] == ["user", "assistant"]


async def test_files_changed_since_go_back(
    rig: Rig, frozen_clock: FrozenClock, tmp_path: Path
) -> None:
    a, b, c = (tmp_path / name for name in ("a.txt", "b.txt", "c.txt"))
    a.write_text("v0")
    await rig.changes.record(rig.session_id, "t0", a, b"v0", b"v1")
    a.write_text("v1")
    frozen_clock.advance(60)
    point = await rig.user("change things")
    await rig.changes.record(rig.session_id, "t1", a, b"v1", b"v2")
    a.write_text("v2")
    await rig.changes.record(rig.session_id, "t1", b, None, b"new")
    b.write_text("new")
    c.write_text("c0")
    await rig.changes.record(rig.session_id, "t1", c, b"c0", b"c1")
    await rig.changes.record(rig.session_id, "t1", c, b"c1", b"c2")
    c.write_text("c2")

    report = await rig.rewinder().rewind(rig.session_id, point)

    assert a.read_text() == "v1"
    assert not b.exists()
    assert c.read_text() == "c0"
    assert sorted((f.path, f.change) for f in report.files) == [
        (str(a), "restored"),
        (str(b), "deleted"),
        (str(c), "restored"),
    ]


async def test_a_file_changed_outside_zero_is_kept(
    rig: Rig, frozen_clock: FrozenClock, tmp_path: Path
) -> None:
    path = tmp_path / "notes.md"
    point = await rig.user("edit notes")
    await rig.changes.record(rig.session_id, "t1", path, b"old", b"zero's")
    path.write_text("the user's own edit")

    report = await rig.rewinder().rewind(rig.session_id, point)

    assert path.read_text() == "the user's own edit"
    assert [(f.path, f.change) for f in report.files] == [(str(path), "changed elsewhere")]


async def test_a_dry_run_changes_nothing(
    rig: Rig, frozen_clock: FrozenClock, tmp_path: Path
) -> None:
    path = tmp_path / "a.txt"
    point = await rig.user("make a file")
    await rig.changes.record(rig.session_id, "t1", path, None, b"x")
    path.write_text("x")
    await rig.answer("made")

    plan = await rig.rewinder().rewind(rig.session_id, point, dry_run=True)

    assert plan.removed_messages == 2
    assert [(f.path, f.change) for f in plan.files] == [(str(path), "deleted")]
    assert path.exists()
    assert len(await rig.messages.list(rig.session_id)) == 2


async def test_todos_session_memory_and_agents_since_are_undone(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    await rig.todos.add(rig.session_id, "kept todo")
    await rig.session_memory.add(rig.session_id, "kept rule")
    old_agent_session = await rig.sessions.create(kind="agent")
    old = await rig.agents.add("old", rig.session_id, old_agent_session.id, "old task")
    frozen_clock.advance(60)
    point = await rig.user("plan the trip")
    await rig.todos.add(rig.session_id, "new todo")
    await rig.session_memory.add(rig.session_id, "new rule")
    new_agent_session = await rig.sessions.create(kind="agent")
    new = await rig.agents.add("sorter", rig.session_id, new_agent_session.id, "sort")

    report = await rig.rewinder().rewind(rig.session_id, point)

    assert [t.text for t in await rig.todos.list(rig.session_id)] == ["kept todo"]
    assert [e.text for e in await rig.session_memory.list(rig.session_id)] == ["kept rule"]
    assert rig.stopped == [new.id]
    assert old.id not in rig.stopped
    assert (report.removed_todos, report.removed_session_memories) == (1, 1)
    assert report.stopped_agents == ["sorter"]


async def test_effects_that_stay_are_listed(rig: Rig, frozen_clock: FrozenClock) -> None:
    point = await rig.user("do things")
    await rig.answer(
        "done",
        call("run_command", 1, command="ls"),
        call("run_command", 2, command="pwd"),
        call("memory_save", 1, text="The user likes tea"),
        call("memory_save", 2, text="Here: English", scope="session"),
        call("read_file", 1, path="~/x"),
        call("write_file", 1, path="~/y", content="z"),
    )

    report = await rig.rewinder().rewind(rig.session_id, point)

    assert report.kept == {"run_command": 2, "memory_save": 1}


async def test_rewind_is_refused_while_busy_or_for_a_wrong_message(rig: Rig) -> None:
    point = await rig.user("hi")
    await rig.answer("hello")
    answer = (await rig.messages.list(rig.session_id))[-1].id
    rig.busy.add(rig.session_id)

    with pytest.raises(RewindError) as busy:
        await rig.rewinder().rewind(rig.session_id, point)
    rig.busy.clear()
    with pytest.raises(RewindError) as not_user:
        await rig.rewinder().rewind(rig.session_id, answer)
    with pytest.raises(RewindError) as missing:
        await rig.rewinder().rewind("nope", point)

    assert (busy.value.code, not_user.value.code, missing.value.code) == (
        "turn_busy",
        "invalid_message",
        "session_not_found",
    )


async def test_summary_and_review_that_covered_removed_messages_are_fixed(
    rig: Rig, frozen_clock: FrozenClock
) -> None:
    await rig.user("first")
    await rig.answer("a1")
    kept_last = (await rig.messages.list(rig.session_id))[-1].id
    point = await rig.user("second")
    await rig.answer("a2")
    last = (await rig.messages.list(rig.session_id))[-1].id
    await rig.summaries.save(rig.session_id, "everything", last)
    await rig.reviews.mark(rig.session_id, last)

    await rig.rewinder().rewind(rig.session_id, point)

    assert await rig.summaries.get(rig.session_id) is None
    assert await rig.reviews.until(rig.session_id) == kept_last
