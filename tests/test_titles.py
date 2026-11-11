"""Tests of automatic session titles."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import Harness, text_of
from zero.config import Config
from zero.core import titles
from zero.core.titles import (
    TITLE_MAX_CHARS,
    TITLE_PROMPT,
    clean_title,
    first_words,
    make_title,
    title_new_session,
)
from zero.protocol.messages import SessionUpdated
from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolSpec,
)
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state

LONG_MESSAGE = "plan a trip to Rome next spring with three friends"


class TitleWriter:
    """A stand-in for a real provider that answers every request with ``chunks``."""

    name = "writer"

    def __init__(
        self,
        *chunks: str,
        error: Exception | None = None,
        delay_s: float = 0.0,
        before: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.delay_s = delay_s
        self.before = before
        self.requests: list[tuple[list[Message], list[ToolSpec], int]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append((messages, tools, max_tokens))
        if self.before is not None:
            await self.before()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield TextDelta(text=chunk)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        return None

    def supports_images(self, model: str) -> bool:
        return False


# --- Making a title ----------------------------------------------------------


def test_first_words_takes_six_words_and_squeezes_spaces() -> None:
    assert first_words("  plan a\ttrip  to Rome next spring\nwith friends ") == (
        "plan a trip to Rome next"
    )


def test_first_words_of_a_huge_word_are_cut_to_the_limit() -> None:
    assert first_words("x" * 5000) == "x" * TITLE_MAX_CHARS


@pytest.mark.parametrize(
    ("raw", "title"),
    [
        ("Trip to Rome", "Trip to Rome"),
        ('"Trip to Rome."\n', "Trip to Rome"),
        ("«Поездка в Рим»", "Поездка в Рим"),
        ("\n\n  Spring trip planning  \nand a second line", "Spring trip planning"),
        ("...", ""),
        ("", ""),
    ],
)
def test_clean_title(raw: str, title: str) -> None:
    assert clean_title(raw) == title


async def test_fake_provider_titles_with_the_first_words_without_a_call() -> None:
    provider = FakeProvider([[TextDelta(text="script step")]])

    title = await make_title(provider, "fake-1", LONG_MESSAGE)

    assert title == "plan a trip to Rome next"
    assert provider.calls == []
    assert provider.steps_left == 1


async def test_other_providers_are_asked_for_a_title() -> None:
    provider = TitleWriter('"Spring trip ', 'to Rome."')

    title = await make_title(provider, "model-x", LONG_MESSAGE)

    assert title == "Spring trip to Rome"
    ((messages, tools, max_tokens),) = provider.requests
    assert [m.role for m in messages] == ["system", "user"]
    assert (text_of(messages[0]), text_of(messages[1])) == (TITLE_PROMPT, LONG_MESSAGE)
    assert (tools, max_tokens) == ([], 32)


async def test_only_the_start_of_a_long_message_is_sent() -> None:
    provider = TitleWriter("Title")

    await make_title(provider, "model-x", "word " * 1000)

    assert len(text_of(provider.requests[0][0][1])) == 2000


@pytest.mark.parametrize(
    "provider",
    [
        TitleWriter(error=ProviderError("overloaded", retryable=True)),
        TitleWriter(),
        TitleWriter("  ", '""'),
    ],
    ids=["provider error", "no answer", "blank answer"],
)
async def test_first_words_are_the_fallback(provider: TitleWriter) -> None:
    assert await make_title(provider, "model-x", LONG_MESSAGE) == "plan a trip to Rome next"


async def test_slow_model_falls_back_to_the_first_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(titles, "TITLE_TIMEOUT_S", 0.01)

    title = await make_title(TitleWriter("Late", delay_s=1), "model-x", LONG_MESSAGE)

    assert title == "plan a trip to Rome next"


async def test_empty_message_gives_no_title_and_asks_nobody() -> None:
    provider = TitleWriter("Title")

    assert await make_title(provider, "model-x", "   ") == ""
    assert provider.requests == []


# --- Titling a session -------------------------------------------------------


async def title(harness: Harness, provider: Any, text: str = LONG_MESSAGE) -> None:
    await title_new_session(
        harness.session_id,
        text,
        provider=provider,
        model="model-x",
        sessions=harness.sessions,
        events=harness.bus,
    )


async def test_untitled_session_gets_a_title_and_everyone_hears(harness: Harness) -> None:
    before = await harness.sessions.get(harness.session_id)

    await title(harness, TitleWriter("Spring trip to Rome"))

    after = await harness.sessions.get(harness.session_id)
    assert before is not None and after is not None
    assert after.title == "Spring trip to Rome"
    assert after.updated_at == before.updated_at
    (event,) = harness.client.received
    assert isinstance(event, SessionUpdated)
    assert (event.session.id, event.session.title) == (harness.session_id, "Spring trip to Rome")


async def test_titled_session_is_left_alone(harness: Harness) -> None:
    await harness.sessions.rename(harness.session_id, "Mine")
    provider = TitleWriter("Automatic")

    await title(harness, provider)

    session = await harness.sessions.get(harness.session_id)
    assert session is not None and session.title == "Mine"
    assert provider.requests == []
    assert harness.client.received == []


async def test_rename_while_the_title_is_made_wins(harness: Harness) -> None:
    async def user_renames() -> None:
        await harness.sessions.rename(harness.session_id, "Mine")

    await title(harness, TitleWriter("Automatic", before=user_renames))

    session = await harness.sessions.get(harness.session_id)
    assert session is not None and session.title == "Mine"
    assert harness.client.received == []


async def test_deleted_session_is_not_titled(harness: Harness) -> None:
    await harness.sessions.delete(harness.session_id)

    await title(harness, TitleWriter("Automatic"))

    assert harness.client.received == []


async def test_empty_message_leaves_the_session_untitled(harness: Harness) -> None:
    await title(harness, FakeProvider(), text="")

    session = await harness.sessions.get(harness.session_id)
    assert session is not None and session.title is None
    assert harness.client.received == []


# --- Over the WebSocket ------------------------------------------------------


def connect(client: TestClient) -> Any:
    token = get_state(client.app).token  # type: ignore[arg-type]
    return client.websocket_connect(f"/ws?token={token}")


def hello(ws: WebSocketTestSession) -> None:
    ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
    assert ws.receive_json()["type"] == "hello.ok"


def send(ws: WebSocketTestSession, text: str, **fields: Any) -> list[dict[str, Any]]:
    """Send a message and read events up to its ``turn.finished``."""
    ws.send_json({"type": "message.send", "id": "m1", "text": text, "source": "desktop", **fields})
    events = [ws.receive_json()]
    while events[-1]["type"] != "turn.finished":
        events.append(ws.receive_json())
    return events


def test_first_turn_titles_the_session_for_every_client() -> None:
    with TestClient(create_app(Config(), FakeProvider())) as client:
        with connect(client) as ws, connect(client) as other:
            hello(ws)
            hello(other)
            events = send(ws, LONG_MESSAGE)
            session_id = events[0]["session"]["id"]
            updated = ws.receive_json()
            seen_by_other = [other.receive_json() for _ in events] + [other.receive_json()]

            send(ws, "and one more thing", session_id=session_id)
            ws.send_json({"type": "session.list", "id": "l1"})
            listed = ws.receive_json()

    assert updated["type"] == "session.updated"
    assert updated["session"]["id"] == session_id
    assert updated["session"]["title"] == "plan a trip to Rome next"
    assert seen_by_other[-1] == updated
    assert listed["type"] == "session.list.ok"
    assert [s["title"] for s in listed["sessions"]] == ["plan a trip to Rome next"]


def test_session_created_with_a_title_keeps_it() -> None:
    with TestClient(create_app(Config(), FakeProvider())) as client, connect(client) as ws:
        hello(ws)
        ws.send_json({"type": "session.create", "title": "Plans"})
        session_id = ws.receive_json()["session"]["id"]

        send(ws, LONG_MESSAGE, session_id=session_id)
        ws.send_json({"type": "ping", "id": "p1"})

        assert ws.receive_json() == {"type": "pong", "reply_to": "p1"}
