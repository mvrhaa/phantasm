"""Tests that remembered facts reach the model through the system prompt."""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from tests.turns import Harness
from zero.config import Config
from zero.core.turn import Turn
from zero.paths import db_path
from zero.providers.base import Message, TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.store.db import Database
from zero.store.memory import MemoryRepo


@pytest.fixture
async def memory() -> AsyncIterator[MemoryRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield MemoryRepo(db)


def system_text(messages: list[Message]) -> str:
    first = messages[0]
    assert first.role == "system"
    return first.content[0].text  # type: ignore[union-attr]


def turn_with(harness: Harness, provider: FakeProvider, memory: MemoryRepo | None) -> Turn:
    return Turn(
        session_id=harness.session_id,
        text="hi",
        source="desktop",
        client=None,
        config=Config(),
        provider=provider,
        sessions=harness.sessions,
        messages=harness.messages,
        events=harness.bus,
        memory=memory,
    )


async def test_saved_facts_are_in_the_system_prompt(harness: Harness, memory: MemoryRepo) -> None:
    await memory.add("The user writes Python")
    await memory.add("The user lives in Prague")
    provider = FakeProvider([[TextDelta(text="ok")]])

    await turn_with(harness, provider, memory).run()

    text = system_text(provider.calls[0].messages)
    assert "- The user lives in Prague\n- The user writes Python" in text
    assert "find the rest with memory_search" in text


async def test_only_the_top_ten_go_into_the_prompt(
    harness: Harness, memory: MemoryRepo, frozen_clock: object
) -> None:
    for n in range(12):
        await memory.add(f"fact number {n}")
        frozen_clock.advance(1)  # type: ignore[attr-defined]
    provider = FakeProvider([[TextDelta(text="ok")]])

    await turn_with(harness, provider, memory).run()

    text = system_text(provider.calls[0].messages)
    assert text.count("- fact number") == 10
    assert "fact number 11" in text and "fact number 1\n" not in text


async def test_no_memories_no_block(harness: Harness, memory: MemoryRepo) -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])

    await turn_with(harness, provider, memory).run()

    assert "What you remember" not in system_text(provider.calls[0].messages)


def test_a_fact_saved_in_one_session_is_known_in_the_next() -> None:
    save = ToolCall(
        id="m1", name="memory_save", input={"text": "The user writes Python", "tags": ["code"]}
    )
    provider = FakeProvider(
        [
            [ToolCallEvent(call=save)],
            [TextDelta(text="I will remember that.")],
            [TextDelta(text="You write Python.")],
        ]
    )
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            for text in ("remember that I write Python", "what do I write in?"):
                ws.send_json({"type": "message.send", "text": text, "source": "desktop"})
                while ws.receive_json()["type"] != "turn.finished":
                    pass

    first_session, second_session = provider.calls[0], provider.calls[2]
    assert "What you remember" not in system_text(first_session.messages)
    assert "- The user writes Python" in system_text(second_session.messages)
    assert [m.role for m in second_session.messages] == ["system", "user"]  # a new session
