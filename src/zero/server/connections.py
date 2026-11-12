"""Registry of connected clients."""

import logging
import uuid
from typing import Protocol

from starlette.websockets import WebSocketDisconnect

from zero.protocol.messages import ClientInfo, Envelope

logger = logging.getLogger(__name__)


class TextSocket(Protocol):
    """The part of a WebSocket a connection needs: sending text and closing."""

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class Connection:
    """One connected client: its socket and what it declared in ``hello``."""

    def __init__(self, websocket: TextSocket) -> None:
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.hello: ClientInfo | None = None
        self.closed = False

    @property
    def client_id(self) -> str | None:
        """The id the client introduced itself with, None before ``hello``."""
        return self.hello.client_id if self.hello else None

    async def send(self, event: Envelope) -> bool:
        """Send an event as JSON without null fields. False if the socket is gone."""
        try:
            await self.websocket.send_text(event.model_dump_json(exclude_none=True))
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("Dropped %s for closed connection %s", event.type, self.id)
            return False
        return True

    async def close(self, code: int, reason: str) -> None:
        """Close the socket from the server side; the read loop then stops."""
        self.closed = True
        try:
            await self.websocket.close(code=code, reason=reason)
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("Connection %s was already closed", self.id)


class ConnectionManager:
    """Keeps track of live connections and delivers events to them."""

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}

    @property
    def connections(self) -> list[Connection]:
        """All live connections, in the order they were added."""
        return list(self._connections.values())

    def get(self, connection_id: str) -> Connection | None:
        """The live connection with this id, or None."""
        return self._connections.get(connection_id)

    def add(self, connection: Connection) -> None:
        """Register a connection."""
        self._connections[connection.id] = connection

    def remove(self, connection: Connection) -> None:
        """Forget a connection. Does nothing if it is not registered."""
        self._connections.pop(connection.id, None)
