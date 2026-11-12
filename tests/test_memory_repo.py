"""Tests for zero.store.memory: the memory repository and its full-text search."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from zero.paths import db_path
from zero.store.db import Database
from zero.store.memory import Memory, MemoryRepo, fts_query


@pytest.fixture
async def memory() -> AsyncIterator[MemoryRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield MemoryRepo(db)


def texts(memories: list[Memory]) -> list[str]:
    return [m.text for m in memories]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", '"pyth"*'),
        ("I write Python", '"I" OR "writ"* OR "Pyth"*'),
        ('"quoted" AND (NEAR x*)', '"quot"* OR "AND"* OR "NEAR"* OR "x"'),
        ("col:umn -minus", '"col"* OR "umn"* OR "minu"*'),
        ("cat", '"cat"*'),
        ("Прага", '"Праг"*'),
        ("питон", '"пито"*'),
        ("  ...  ", ""),
        ("", ""),
    ],
)
def test_fts_query_quotes_every_word(query: str, expected: str) -> None:
    assert fts_query(query) == expected


async def test_add_returns_the_memory(memory: MemoryRepo, frozen_clock: Any) -> None:
    saved = await memory.add("  The user writes Python  ", ["code", " work ", "code", ""])

    assert saved.text == "The user writes Python"
    assert saved.tags == ["code", "work"]
    assert saved.created_at == saved.updated_at == frozen_clock()
    assert saved.uses == 0
    assert await memory.get(saved.id) == saved


async def test_tags_cannot_smuggle_commas(memory: MemoryRepo) -> None:
    saved = await memory.add("fact", ["a,b"])

    assert saved.tags == ["a b"]


async def test_same_text_again_updates_instead_of_duplicating(
    memory: MemoryRepo, frozen_clock: Any
) -> None:
    first = await memory.add("Likes tea", ["drinks"])
    frozen_clock.advance(60)

    again = await memory.add("Likes tea")

    assert again.id == first.id
    assert again.tags == ["drinks"]
    assert again.updated_at > first.updated_at
    assert texts(await memory.top()) == ["Likes tea"]


async def test_empty_text_is_refused(memory: MemoryRepo) -> None:
    with pytest.raises(ValueError, match="needs some text"):
        await memory.add("   ")


async def test_get_and_delete(memory: MemoryRepo) -> None:
    saved = await memory.add("temporary")

    assert await memory.delete(saved.id)
    assert await memory.get(saved.id) is None
    assert not await memory.delete(saved.id)


@pytest.fixture
async def facts(memory: MemoryRepo) -> MemoryRepo:
    await memory.add("The user writes Python every day", ["programming"])
    await memory.add("The user's cat is called Bublik", ["pets"])
    await memory.add("Пользователь живёт в Праге", ["город"])
    await memory.add("Пишет бота на питоне", ["проект"])
    return memory


@pytest.mark.parametrize(
    ("query", "found"),
    [
        ("python", ["The user writes Python every day"]),
        ("PYTHON", ["The user writes Python every day"]),
        ("pets", ["The user's cat is called Bublik"]),
        ("what is my cat called?", ["The user's cat is called Bublik"]),
        ("Прага", ["Пользователь живёт в Праге"]),
        ("питон", ["Пишет бота на питоне"]),
        ("город", ["Пользователь живёт в Праге"]),
        ("dogs", []),
    ],
)
async def test_search(facts: MemoryRepo, query: str, found: list[str]) -> None:
    assert texts(await facts.search(query)) == found


@pytest.mark.parametrize(
    "query", ['"', "AND", "NEAR(", "a:b", "*", "-python", "(python) OR", "^", "''"]
)
async def test_fts_syntax_in_a_query_is_harmless(facts: MemoryRepo, query: str) -> None:
    await facts.search(query)


async def test_better_matches_come_first(memory: MemoryRepo) -> None:
    await memory.add("Writes Python at work")
    await memory.add("Python code about the cat")

    assert texts(await memory.search("python cat"))[0] == "Python code about the cat"


async def test_search_limit(memory: MemoryRepo) -> None:
    for n in range(5):
        await memory.add(f"fact number {n} about tea")

    assert len(await memory.search("tea", limit=3)) == 3
    assert await memory.search("tea", limit=0) == []


async def test_deleted_and_changed_memories_leave_the_index(facts: MemoryRepo) -> None:
    (cat,) = await facts.search("cat")
    await facts.delete(cat.id)

    assert await facts.search("cat") == []


async def test_changed_tags_are_searched(memory: MemoryRepo) -> None:
    await memory.add("Likes tea", ["drinks"])
    await memory.add("Likes tea", ["morning"])

    assert texts(await memory.search("morning")) == ["Likes tea"]
    assert await memory.search("drinks") == []


async def test_top_mixes_the_most_used_and_the_newest(
    memory: MemoryRepo, frozen_clock: Any
) -> None:
    saved = []
    for n in range(6):
        saved.append(await memory.add(f"fact {n}"))
        frozen_clock.advance(10)
    await memory.bump_uses([saved[0].id, saved[1].id])
    await memory.bump_uses([saved[0].id])

    top = await memory.top(4)

    assert texts(top) == ["fact 0", "fact 1", "fact 5", "fact 4"]
    assert [m.uses for m in top] == [2, 1, 0, 0]


async def test_top_without_uses_is_the_newest(memory: MemoryRepo, frozen_clock: Any) -> None:
    for n in range(3):
        await memory.add(f"fact {n}")
        frozen_clock.advance(10)

    assert texts(await memory.top(10)) == ["fact 2", "fact 1", "fact 0"]
    assert await memory.top(0) == []


async def test_bump_uses_does_not_change_freshness(memory: MemoryRepo) -> None:
    saved = await memory.add("fact")

    await memory.bump_uses([saved.id, "missing"])
    await memory.bump_uses([])

    after = await memory.get(saved.id)
    assert after is not None
    assert (after.uses, after.updated_at) == (1, saved.updated_at)
