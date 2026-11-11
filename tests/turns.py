"""Helpers for running a turn directly, without a WebSocket."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from zero.config import Config
from zero.core.events import EventBus
from zero.core.stub_tools import StubToolRegistry
from zero.core.turn import Turn
from zero.protocol.messages import (
    ClientInfo,
    Envelope,
    TextPart,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from zero.providers.base import LLMProvider, Message
from zero.store.messages import MessagesRepo, StoredMessage
from zero.store.sessions import SessionsRepo

NO_DELAYS = (0.0, 0.0)


@dataclass
class RecordingClient:
    """Stands in for a connected client and keeps everything it receives."""

    id: str = "c-1"
    received: list[Envelope] = field(default_factory=list)

    async def send(self, event: Envelope) -> bool:
        self.received.append(event)
        return True


@dataclass
class Audience:
    """Everyone connected: here, a single recording client."""

    connections: list[RecordingClient]


@dataclass
class Harness:
    """A session in a real database and a client that watches its turns."""

    sessions: SessionsRepo
    messages: MessagesRepo
    session_id: str
    client: RecordingClient = field(default_factory=RecordingClient)

    def turn(
        self,
        provider: LLMProvider,
        tools: StubToolRegistry | None = None,
        *,
        max_steps: int | None = None,
        text: str = "hi",
        client: ClientInfo | None = None,
        retry_delays: Sequence[float] = NO_DELAYS,
    ) -> Turn:
        """Build a turn in this session; retries do not wait unless asked to."""
        config = Config()
        if max_steps is not None:
            config.agent.max_steps = max_steps
        return Turn(
            session_id=self.session_id,
            text=text,
            source="desktop",
            client=client,
            config=config,
            provider=provider,
            sessions=self.sessions,
            messages=self.messages,
            events=EventBus(Audience([self.client])),
            tools=tools,
            retry_delays=retry_delays,
        )

    async def run(
        self,
        provider: LLMProvider,
        tools: StubToolRegistry | None = None,
        *,
        max_steps: int | None = None,
        text: str = "hi",
        client: ClientInfo | None = None,
    ) -> Turn:
        """Build a turn and run it to the end."""
        turn = self.turn(provider, tools, max_steps=max_steps, text=text, client=client)
        await turn.run()
        return turn

    @property
    def types(self) -> list[str]:
        """Types of the events the client got, in order."""
        return [event.type for event in self.client.received]

    @property
    def deltas(self) -> list[str]:
        """Text chunks the client got."""
        return [e.delta for e in self.client.received if e.type == "text.delta"]  # type: ignore[attr-defined]

    @property
    def finished(self) -> TurnFinished:
        """The last event, which must be ``turn.finished``."""
        last = self.client.received[-1]
        assert isinstance(last, TurnFinished)
        return last

    def tool_events(self) -> list[ToolStarted | ToolFinished]:
        """``tool.started`` and ``tool.finished`` events, in order."""
        return [e for e in self.client.received if isinstance(e, ToolStarted | ToolFinished)]

    async def stored(self) -> list[StoredMessage]:
        """Everything saved in the session, oldest first."""
        return await self.messages.list(self.session_id, limit=100)


def text_of(message: Any) -> str:
    """The text parts of a message or a tool result, joined."""
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def roles(messages: Sequence[Message | StoredMessage]) -> str:
    """Roles of the messages as one string, e.g. ``"user assistant"``."""
    return " ".join(message.role for message in messages)


async def until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    """Let other tasks run until ``condition`` holds; fail after ``timeout`` seconds."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.001)
