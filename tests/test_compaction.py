"""Tests of compaction: a long history's start folded into a running summary."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.turns import Audience, Harness, RecordingClient, text_of
from zero.config import Config
from zero.core import compaction
from zero.core.compaction import compact, digest, summarize, transcript
from zero.core.context import keep_from
from zero.core.events import EventBus
from zero.core.turn import Turn
from zero.paths import db_path
from zero.protocol.messages import SessionCompacted, TextPart
from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    TextDelta,
    ToolSpec,
)
from zero.providers.fake import FakeProvider
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo


class Writer:
    """A stand-in for a real model: answers with ``answer`` and remembers the requests."""

    name = "writer"

    def __init__(self, answer: str = "A summary.", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.requests: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> Any:
        self.requests.append(messages)
        if self.error is not None:
            raise self.error
        yield TextDelta(text=self.answer)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        return None

    def supports_images(self, model: str) -> bool:
        return False


@pytest.fixture
async def stores() -> AsyncIterator[tuple[SessionsRepo, MessagesRepo, SummariesRepo]]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield SessionsRepo(db), MessagesRepo(db), SummariesRepo(db)


async def chat(messages: MessagesRepo, session_id: str, *pairs: tuple[str, str]) -> None:
    for question, answer in pairs:
        await messages.append(session_id, "user", [TextPart(text=question)], status="ok")
        await messages.append(session_id, "assistant", [TextPart(text=answer)], status="ok")


# --- Text helpers ------------------------------------------------------------


async def test_transcript_names_who_said_what(stores: Any) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("Hello  there", "Hi!"))
    await messages.append(
        session.id,
        "assistant",
        [],
        tool_calls=[{"id": "c", "name": "read_file", "input": {}}],
        status="ok",
    )
    await messages.append(session.id, "tool", [TextPart(text="file text")], tool_call_id="c")

    lines = transcript(await messages.list(session.id))

    assert (
        lines == "User: Hello there\nAssistant: Hi!\nAssistant: [used read_file]\nTool: file text"
    )


async def test_digest_keeps_the_end_when_it_grows_too_long(
    stores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("q" * 50, "a" * 50))
    monkeypatch.setattr(compaction, "SUMMARY_MAX_CHARS", 40)

    text = digest("old summary", await messages.list(session.id))

    assert len(text) == 40
    assert text.startswith("…") and text.endswith("a" * 10)


# --- Summarizing -------------------------------------------------------------


async def test_fake_provider_gets_a_digest_without_a_call(stores: Any) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("I live in Prague", "Nice city."))
    provider = FakeProvider([[TextDelta(text="script step")]])

    text = await summarize(provider, "fake-1", None, await messages.list(session.id))

    assert text == "User: I live in Prague\nAssistant: Nice city."
    assert provider.calls == []


async def test_other_providers_write_the_summary(stores: Any) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("I live in Prague", "Nice city."))
    writer = Writer("  The user lives in Prague.  ")

    text = await summarize(writer, "m", "Earlier: likes tea.", await messages.list(session.id))

    assert text == "The user lives in Prague."
    (request,) = writer.requests
    assert text_of(request[0]) == compaction.SUMMARY_PROMPT
    assert text_of(request[1]) == (
        "Earlier summary:\nEarlier: likes tea.\n\n"
        "Conversation to add:\nUser: I live in Prague\nAssistant: Nice city."
    )


@pytest.mark.parametrize(
    "writer",
    [Writer(error=ProviderError("overloaded", retryable=True)), Writer("   ")],
    ids=["failure", "blank answer"],
)
async def test_digest_when_the_model_cannot_help(stores: Any, writer: Writer) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("hi", "hello"))

    assert await summarize(writer, "m", None, await messages.list(session.id)) == (
        "User: hi\nAssistant: hello"
    )


# --- The cut -----------------------------------------------------------------


async def test_keep_from_cuts_between_turns(stores: Any) -> None:
    sessions, messages, _ = stores
    session = await sessions.create()
    await chat(messages, session.id, ("a" * 35, "b" * 35), ("c" * 35, "d" * 35), ("e", "f"))
    stored = await messages.list(session.id)

    assert keep_from(stored, 10_000) == 0
    assert keep_from(stored, 25) == 2  # each long message is 10 tokens: two turns fit
    assert keep_from(stored, 0) == 4  # the latest turn is always kept


# --- compact -----------------------------------------------------------------


async def test_compact_summarizes_once_and_then_only_what_is_new(stores: Any) -> None:
    sessions, messages, summaries = stores
    session = await sessions.create()
    await chat(messages, session.id, ("one", "1"), ("two", "2"), ("three", "3"))
    client = RecordingClient()
    bus = EventBus(Audience([client]))
    writer = Writer("S1")

    async def run(keep: int) -> str:
        stored = await messages.list(session.id)
        return await compact(
            session.id, stored, keep, summaries=summaries, provider=writer, model="m", events=bus
        )

    first = await run(2)
    same = await run(2)
    await chat(messages, session.id, ("four", "4"))
    writer.answer = "S2"
    second = await run(4)

    assert (first, same, second) == ("S1", "S1", "S2")
    assert len(writer.requests) == 2
    assert text_of(writer.requests[1][1]) == (
        "Earlier summary:\nS1\n\nConversation to add:\nUser: two\nAssistant: 2"
    )
    stored = await messages.list(session.id)
    saved = await summaries.get(session.id)
    assert saved is not None and saved.until_message_id == stored[3].id
    assert client.received == [
        SessionCompacted(session_id=session.id, summary="S1", until_message_id=stored[1].id),
        SessionCompacted(session_id=session.id, summary="S2", until_message_id=stored[3].id),
    ]


# --- In a turn ---------------------------------------------------------------


def turn_with(
    harness: Harness, provider: Any, summaries: SummariesRepo | None, budget: int
) -> Turn:
    config = Config()
    config.agent.history_token_budget = budget
    return Turn(
        session_id=harness.session_id,
        text="what did I say first?",
        source="desktop",
        client=None,
        config=config,
        provider=provider,
        sessions=harness.sessions,
        messages=harness.messages,
        events=harness.bus,
        summaries=summaries,
    )


def system_text(messages: list[Message]) -> str:
    return messages[0].content[0].text  # type: ignore[union-attr]


async def test_long_history_reaches_the_model_as_a_summary_and_the_latest_turns(
    harness: Harness, stores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, summaries = stores
    monkeypatch.setattr("zero.core.prompting.SUMMARY_RESERVE_TOKENS", 0)
    await chat(
        harness.messages, harness.session_id, *[(f"q{n} " * 30, f"a{n} " * 30) for n in range(6)]
    )
    provider = FakeProvider([[TextDelta(text="You said q0.")]])

    await turn_with(harness, provider, summaries, budget=60).run()

    sent = provider.calls[0].messages
    assert "# Earlier in this conversation" in system_text(sent)
    assert "User: q0 q0" in system_text(sent)
    assert [m.role for m in sent[1:]][-1] == "user"
    assert len(sent) - 1 < 13  # fewer messages than the 12 stored plus the new one
    assert "session.compacted" in harness.types
    saved = await summaries.get(harness.session_id)
    assert saved is not None and saved.text.startswith("User: q0")


async def test_short_history_is_not_compacted(harness: Harness, stores: Any) -> None:
    _, _, summaries = stores
    await chat(harness.messages, harness.session_id, ("hi", "hello"))
    provider = FakeProvider([[TextDelta(text="ok")]])

    await turn_with(harness, provider, summaries, budget=60_000).run()

    assert "Earlier in this conversation" not in system_text(provider.calls[0].messages)
    assert "session.compacted" not in harness.types
    assert await summaries.get(harness.session_id) is None


async def test_without_summaries_the_history_is_just_cut(harness: Harness) -> None:
    await chat(
        harness.messages, harness.session_id, *[(f"q{n} " * 30, f"a{n} " * 30) for n in range(6)]
    )
    provider = FakeProvider([[TextDelta(text="ok")]])

    await turn_with(harness, provider, None, budget=60).run()

    sent = provider.calls[0].messages
    assert "Earlier in this conversation" not in system_text(sent)
    assert len(sent) < 14
