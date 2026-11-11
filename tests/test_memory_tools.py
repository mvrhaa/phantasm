"""Tests for the built-in memory tools: memory_save, memory_search and memory_delete."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.turns import FakeApprover, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.paths import db_path
from zero.providers.base import ToolCall
from zero.store.db import Database
from zero.store.memory import MemoryRepo
from zero.tools.base import DENIED, ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.permissions import PathGuard


@pytest.fixture
async def memory() -> AsyncIterator[MemoryRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield MemoryRepo(db)


async def use(
    memory: MemoryRepo, tool: str, approver: FakeApprover | None = None, **tool_input: Any
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
    )
    call = ToolCall(id="m", name=tool, input=tool_input)
    return await builtin_registry(memory).execute(call, ctx)


def test_memory_tools_join_the_registry_only_with_a_memory(memory: MemoryRepo) -> None:
    names = [spec.name for spec in builtin_registry(memory).specs()]

    assert names[-3:] == ["memory_save", "memory_search", "memory_delete"]
    assert "memory_save" not in [spec.name for spec in builtin_registry().specs()]
    policies = {name: builtin_registry(memory).get(name).default_policy for name in names[-3:]}  # type: ignore[union-attr]
    assert policies == {"memory_save": "allow", "memory_search": "allow", "memory_delete": "ask"}


# --- memory_save -------------------------------------------------------------


async def test_save_remembers_a_fact(memory: MemoryRepo) -> None:
    result = await use(memory, "memory_save", text="The user writes Python", tags=["code"])

    (saved,) = await memory.search("python")
    assert (saved.text, saved.tags) == ("The user writes Python", ["code"])
    assert text_of(result) == f"Remembered: The user writes Python (id {saved.id})."
    assert result.summary == "saved"


async def test_save_needs_text(memory: MemoryRepo) -> None:
    for text in ("", "   "):
        result = await use(memory, "memory_save", text=text)

        assert result.is_error
    assert await memory.top() == []


async def test_save_limits_its_size(memory: MemoryRepo) -> None:
    result = await use(memory, "memory_save", text="x" * 1001)

    assert "text: String should have at most 1000 characters" in text_of(result)


# --- memory_search -----------------------------------------------------------


async def test_search_lists_matches_with_ids_and_counts_their_use(memory: MemoryRepo) -> None:
    cat = await memory.add("The user's cat is called Bublik", ["pets"])
    await memory.add("Lives in Prague")

    result = await use(memory, "memory_search", query="what is the cat called")

    assert text_of(result) == f"- [{cat.id}] The user's cat is called Bublik (tags: pets)"
    assert result.summary == "1 memory"
    found = await memory.get(cat.id)
    assert found is not None and found.uses == 1


async def test_search_without_matches(memory: MemoryRepo) -> None:
    result = await use(memory, "memory_search", query="dogs")

    assert (text_of(result), result.summary, result.is_error) == (
        "No memories match.",
        "0 memories",
        False,
    )


# --- memory_delete -----------------------------------------------------------


async def test_delete_asks_and_shows_what_will_be_forgotten(memory: MemoryRepo) -> None:
    saved = await memory.add("Likes tea")
    approver = FakeApprover("allow")

    result = await use(memory, "memory_delete", approver, id=saved.id)

    (ask,) = approver.asked
    assert ask.label == "forget: Likes tea"
    assert (text_of(result), result.summary) == ("Forgot: Likes tea", "forgotten")
    assert await memory.get(saved.id) is None


async def test_denied_delete_keeps_the_memory(memory: MemoryRepo) -> None:
    saved = await memory.add("Likes tea")

    result = await use(memory, "memory_delete", FakeApprover("deny"), id=saved.id)

    assert text_of(result) == DENIED
    assert await memory.get(saved.id) is not None


async def test_delete_of_an_unknown_id_is_refused_without_asking(memory: MemoryRepo) -> None:
    approver = FakeApprover("allow")

    result = await use(memory, "memory_delete", approver, id="missing")

    assert (result.is_error, text_of(result)) == (True, "There is no memory with id missing.")
    assert approver.asked == []
