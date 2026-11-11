"""The event bus: how the brain tells clients what is happening.

Turns and handlers publish protocol events here and never touch sockets.
The bus delivers every event to every connected client, so a desktop and a
phone see the same conversation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from zero.protocol.messages import Envelope


class Recipient(Protocol):
    """One connected client as the bus sees it."""

    @property
    def id(self) -> str: ...

    async def send(self, event: Envelope) -> bool: ...


class Audience(Protocol):
    """Whoever is connected right now."""

    @property
    def connections(self) -> Sequence[Recipient]: ...


@dataclass(frozen=True)
class ReplyTo:
    """Which connection asked for this event, and the id of its request.

    Only that connection's copy of the event gets ``reply_to``; the others
    get the event without it.
    """

    connection_id: str
    message_id: str | None


class EventBus:
    """Delivers events to every connected client."""

    def __init__(self, audience: Audience) -> None:
        self._audience = audience

    async def publish(self, event: Envelope, *, reply: ReplyTo | None = None) -> None:
        """Send ``event`` to everyone; a dead connection does not stop the rest."""
        answer = (
            event.model_copy(update={"reply_to": reply.message_id})
            if reply is not None and reply.message_id is not None
            else event
        )
        for recipient in list(self._audience.connections):
            if reply is not None and recipient.id == reply.connection_id:
                await recipient.send(answer)
            else:
                await recipient.send(event)
