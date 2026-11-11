"""Tests for zero.server.connections."""

import json

from starlette.websockets import WebSocketDisconnect

from zero.protocol.messages import ClientInfo, Error, Pong
from zero.server.connections import Connection, ConnectionManager


class FakeSocket:
    """Records the text frames sent to it; can pretend to be closed."""

    def __init__(self, closed: bool = False) -> None:
        self.sent: list[str] = []
        self.closed = closed

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise WebSocketDisconnect(1006)
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True


def connect(manager: ConnectionManager, client_id: str | None = None, closed: bool = False):
    socket = FakeSocket(closed)
    connection = Connection(socket)
    if client_id is not None:
        connection.hello = ClientInfo(app="test", client_id=client_id)
    manager.add(connection)
    return connection, socket


async def test_send_writes_json_without_null_fields() -> None:
    socket = FakeSocket()

    delivered = await Connection(socket).send(Pong(reply_to="p1"))

    assert delivered is True
    assert json.loads(socket.sent[0]) == {"type": "pong", "reply_to": "p1"}


async def test_send_to_a_closed_socket_returns_false() -> None:
    assert await Connection(FakeSocket(closed=True)).send(Pong()) is False


def test_client_id_comes_from_hello() -> None:
    connection = Connection(FakeSocket())
    assert connection.client_id is None

    connection.hello = ClientInfo(app="test", client_id="c-1")

    assert connection.client_id == "c-1"


def test_connections_get_distinct_ids() -> None:
    assert Connection(FakeSocket()).id != Connection(FakeSocket()).id


def test_add_and_remove() -> None:
    manager = ConnectionManager()
    first, _ = connect(manager)
    second, _ = connect(manager)

    assert manager.connections == [first, second]

    manager.remove(first)
    manager.remove(first)

    assert manager.connections == [second]


async def test_broadcast_reaches_everyone() -> None:
    manager = ConnectionManager()
    _, first = connect(manager, "c-1")
    _, second = connect(manager)

    await manager.broadcast(Pong())

    assert first.sent == second.sent == ['{"type":"pong"}']


async def test_broadcast_skips_a_dead_connection() -> None:
    manager = ConnectionManager()
    _, dead = connect(manager, closed=True)
    _, alive = connect(manager)

    await manager.broadcast(Pong())

    assert dead.sent == []
    assert alive.sent == ['{"type":"pong"}']


async def test_send_to_reaches_only_the_named_client() -> None:
    manager = ConnectionManager()
    _, target = connect(manager, "c-1")
    _, other = connect(manager, "c-2")
    _, anonymous = connect(manager)

    delivered = await manager.send_to("c-1", Error(code="internal", message="x"))

    assert delivered is True
    assert len(target.sent) == 1
    assert other.sent == anonymous.sent == []


async def test_send_to_unknown_or_dead_client_returns_false() -> None:
    manager = ConnectionManager()
    connect(manager, "c-1", closed=True)

    assert await manager.send_to("c-1", Pong()) is False
    assert await manager.send_to("nobody", Pong()) is False


async def test_broadcast_can_skip_one_connection() -> None:
    manager = ConnectionManager()
    sender, sender_socket = connect(manager)
    _, other = connect(manager)

    await manager.broadcast(Pong(), exclude=sender)

    assert sender_socket.sent == []
    assert other.sent == ['{"type":"pong"}']
