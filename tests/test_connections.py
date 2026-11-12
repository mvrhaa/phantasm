"""Tests for zero.server.connections."""

import json

from starlette.websockets import WebSocketDisconnect

from zero.protocol.messages import ClientInfo, Pong
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
