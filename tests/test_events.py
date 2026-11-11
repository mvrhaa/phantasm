"""Tests for zero.core.events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from zero.config import Config
from zero.core.events import EventBus, ReplyTo
from zero.core.turn import Turn
from zero.paths import db_path
from zero.protocol.messages import Envelope, Pong, SessionDeleted
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider
from zero.store.db import Database
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo


@dataclass
class FakeRecipient:
    id: str
    alive: bool = True
    received: list[Envelope] = field(default_factory=list)

    async def send(self, event: Envelope) -> bool:
        if not self.alive:
            return False
        self.received.append(event)
        return True


@dataclass
class FakeAudience:
    connections: list[FakeRecipient] = field(default_factory=list)


def bus_with(*ids: str) -> tuple[EventBus, list[FakeRecipient]]:
    recipients = [FakeRecipient(id=connection_id) for connection_id in ids]
    return EventBus(FakeAudience(recipients)), recipients


async def test_publish_reaches_every_connection() -> None:
    bus, (first, second) = bus_with("a", "b")

    await bus.publish(SessionDeleted(session_id="s1"))

    assert first.received == second.received == [SessionDeleted(session_id="s1")]


async def test_publish_with_nobody_connected_does_nothing() -> None:
    bus, _ = bus_with()

    await bus.publish(Pong())


async def test_only_the_requester_gets_reply_to() -> None:
    bus, (requester, other) = bus_with("a", "b")

    await bus.publish(SessionDeleted(session_id="s1"), reply=ReplyTo("a", "m1"))

    assert requester.received == [SessionDeleted(session_id="s1", reply_to="m1")]
    assert other.received == [SessionDeleted(session_id="s1")]


async def test_reply_without_message_id_changes_nothing() -> None:
    bus, (requester,) = bus_with("a")

    await bus.publish(Pong(), reply=ReplyTo("a", None))

    assert requester.received == [Pong()]


async def test_reply_to_a_connection_that_left_still_reaches_the_rest() -> None:
    bus, (other,) = bus_with("b")

    await bus.publish(Pong(), reply=ReplyTo("gone", "m1"))

    assert other.received == [Pong()]


async def test_a_dead_connection_does_not_stop_delivery() -> None:
    audience = FakeAudience([FakeRecipient("a", alive=False), FakeRecipient("b")])

    await EventBus(audience).publish(Pong())

    assert audience.connections[1].received == [Pong()]


async def test_the_published_event_is_not_modified() -> None:
    bus, _ = bus_with("a", "b")
    event = SessionDeleted(session_id="s1")

    await bus.publish(event, reply=ReplyTo("a", "m1"))

    assert event.reply_to is None


async def test_connections_joining_later_see_later_events() -> None:
    audience = FakeAudience([FakeRecipient("a")])
    bus = EventBus(audience)

    await bus.publish(Pong(reply_to="first"))
    audience.connections.append(FakeRecipient("b"))
    await bus.publish(Pong(reply_to="second"))

    assert [e.reply_to for e in audience.connections[0].received] == ["first", "second"]
    assert [e.reply_to for e in audience.connections[1].received] == ["second"]


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(db_path()) as database:
        await database.migrate()
        yield database


async def test_turn_publishes_on_the_bus_without_any_websocket(db: Database) -> None:
    bus, (requester, other) = bus_with("a", "b")
    sessions, messages = SessionsRepo(db), MessagesRepo(db)
    session = await sessions.create()
    turn = Turn(
        session_id=session.id,
        text="hi",
        source="desktop",
        client=None,
        config=Config(),
        provider=FakeProvider([[TextDelta(text="Hel"), TextDelta(text="lo")]]),
        sessions=sessions,
        messages=messages,
        events=bus,
        reply=ReplyTo("a", "m1"),
    )

    await turn.run()

    types = [event.type for event in other.received]
    assert types == ["turn.started", "text.delta", "text.delta", "turn.finished"]
    assert [e.type for e in requester.received] == types
    assert requester.received[0].reply_to == "m1"
    assert other.received[0].reply_to is None
    assert requester.received[1:] == other.received[1:]
