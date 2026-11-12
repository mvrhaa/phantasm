"""Tests of the record of model calls: what is written down, and the totals."""

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FrozenClock
from tests.turns import Harness
from zero import cli
from zero.config import Config
from zero.core.cancellation import TurnCancelled
from zero.core.compaction import summarize
from zero.core.model_call import metered
from zero.core.titles import make_title
from zero.paths import db_path
from zero.protocol.messages import TextPart
from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolSpec,
    UsageEvent,
)
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.store.db import Database
from zero.store.usage import ModelCall, UsageRepo


class Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[ModelCall] = []
        self.fail = fail

    async def record(self, call: ModelCall) -> None:
        if self.fail:
            raise RuntimeError("disk full")
        self.calls.append(call)


async def events(*items: StreamEvent | Exception) -> AsyncIterator[StreamEvent]:
    for item in items:
        if isinstance(item, Exception):
            raise item
        yield item


async def drain(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# --- metered -----------------------------------------------------------------


async def test_a_finished_call_is_recorded_with_its_tokens() -> None:
    recorder = Recorder()
    call = ModelCall("groq", "m", "turn", "s1")
    usage = [UsageEvent(input_tokens=10, output_tokens=2), UsageEvent(input_tokens=5)]

    seen = await drain(metered(events(TextDelta(text="hi"), *usage), call, recorder))

    assert len(seen) == 3
    assert recorder.calls == [ModelCall("groq", "m", "turn", "s1", 15, 2, ok=True)]


@pytest.mark.parametrize("error", [ProviderError("rate limited", retryable=True), TurnCancelled()])
async def test_failed_and_cancelled_calls_count_too(error: Exception) -> None:
    recorder = Recorder()
    stream = metered(
        events(UsageEvent(input_tokens=7), error), ModelCall("p", "m", "turn"), recorder
    )

    with pytest.raises(type(error)):
        await drain(stream)

    (call,) = recorder.calls
    assert (call.ok, call.input_tokens) == (False, 7)


async def test_a_record_that_fails_does_not_break_the_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen = await drain(
        metered(events(TextDelta(text="hi")), ModelCall("p", "m", "turn"), Recorder(fail=True))
    )

    assert seen == [TextDelta(text="hi")]
    assert "Could not record a call to p" in caplog.text


# --- In a turn ---------------------------------------------------------------


async def test_every_model_call_of_a_turn_is_recorded(harness: Harness) -> None:
    recorder = Recorder()
    provider = FakeProvider(fail_times=1)  # echo after one retryable failure
    await harness.turn(provider, usage=recorder).run()

    failed, answered = recorder.calls
    assert (failed.provider, failed.model, failed.purpose, failed.ok) == (
        "fake",
        "fake-1",
        "turn",
        False,
    )
    assert answered.ok and answered.session_id == harness.session_id
    assert (answered.input_tokens, answered.output_tokens) == (1, 2)  # echo of "hi"


# --- Titles and summaries ----------------------------------------------------


class Writer:
    name = "writer"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> Any:
        yield TextDelta(text=self.answer)
        yield UsageEvent(input_tokens=30, output_tokens=4)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        return None

    def supports_images(self, model: str) -> bool:
        return False


async def test_titles_and_summaries_are_recorded_by_purpose(harness: Harness) -> None:
    recorder = Recorder()
    await harness.messages.append(
        harness.session_id, "user", [TextPart(text="hello there")], status="ok"
    )
    stored = await harness.messages.list(harness.session_id)

    await make_title(Writer("Greeting"), "m1", "hello there", session_id="s", usage=recorder)
    await summarize(Writer("A summary."), "m2", None, stored, session_id="s", usage=recorder)
    await make_title(FakeProvider(), "fake-1", "hello", usage=recorder)  # no model call

    assert [(c.purpose, c.model, c.input_tokens, c.output_tokens) for c in recorder.calls] == [
        ("title", "m1", 30, 4),
        ("summary", "m2", 30, 4),
    ]


# --- The store ---------------------------------------------------------------


async def test_totals_per_model_since_a_moment(frozen_clock: FrozenClock) -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        repo = UsageRepo(db)
        await repo.record(ModelCall("groq", "big", "turn", input_tokens=100, output_tokens=10))
        start = frozen_clock.current
        frozen_clock.advance(60)
        await repo.record(ModelCall("groq", "big", "agent", input_tokens=50, output_tokens=5))
        await repo.record(ModelCall("groq", "big", "title", ok=False))
        await repo.record(ModelCall("mistral", "small", "turn", input_tokens=7, output_tokens=1))

        recent = await repo.totals(start + timedelta(seconds=1))
        everything = await repo.totals(start - timedelta(days=1))

    assert [(t.provider, t.model, t.requests, t.failed, t.input_tokens) for t in recent] == [
        ("groq", "big", 2, 1, 50),
        ("mistral", "small", 1, 0, 7),
    ]
    assert everything[0].requests == 3


# --- Through the service and the command -------------------------------------


def test_the_service_records_turns_and_zero_usage_shows_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TestClient(create_app(Config(), FakeProvider())) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            ws.send_json({"type": "message.send", "id": "m1", "text": "hi", "source": "desktop"})
            while ws.receive_json()["type"] != "turn.finished":
                pass

    assert cli.main(["usage"]) == 0
    out = capsys.readouterr().out
    assert out == (
        "Today (since 00:00 UTC):\n"
        "  fake / fake-1: 1 requests (0 failed), 1 tokens in, 2 out\n"
        "Last 30 days:\n"
        "  fake / fake-1: 1 requests (0 failed), 1 tokens in, 2 out\n"
    )


def test_zero_usage_without_a_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["usage"]) == 0
    assert capsys.readouterr().out.count("no model calls") == 2
