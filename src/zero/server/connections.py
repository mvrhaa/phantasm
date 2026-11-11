"""Registry of connected clients."""

import logging
import uuid
from typing import Protocol

from starlette.websockets import WebSocketDisconnect

from zero.protocol.messages import ClientInfo, Envelope

logger = logging.getLogger(__name__)


class TextSocket(Protocol):
    """The part of a WebSocket a connection needs: sending text frames."""

    async def send_text(self, data: str) -> None: ...


class Connection:
    """One connected client: its socket and what it declared in ``hello``."""

    def __init__(self, websocket: TextSocket) -> None:
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.hello: ClientInfo | None = None

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


class ConnectionManager:
    """Keeps track of live connections and delivers events to them."""

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}

    @property
    def connections(self) -> list[Connection]:
        """All live connections, in the order they were added."""
        return list(self._connections.values())

    def add(self, connection: Connection) -> None:
        """Register a connection."""
        self._connections[connection.id] = connection

    def remove(self, connection: Connection) -> None:
        """Forget a connection. Does nothing if it is not registered."""
        self._connections.pop(connection.id, None)

    async def broadcast(self, event: Envelope) -> None:
        """Send an event to every connection; a dead socket does not stop the rest."""
        for connection in self.connections:
            await connection.send(event)

    async def send_to(self, client_id: str, event: Envelope) -> bool:
        """Send an event to the connections of one client. False if none received it."""
        delivered = False
        for connection in self.connections:
            if connection.client_id == client_id:
                delivered = await connection.send(event) or delivered
        return delivered
