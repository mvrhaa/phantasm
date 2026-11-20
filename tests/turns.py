"""Helpers for running a turn directly, without a WebSocket."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from phantasm.approvals.manager import Answer, ApprovalAsk, Approver, Decision
from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.core.events import EventBus
from phantasm.core.model_call import RETRY_DELAYS_S, UsageRecorder
from phantasm.core.tool_runner import ToolRunRecorder
from phantasm.core.turn import Turn
from phantasm.protocol.messages import (
    ClientInfo,
    Envelope,
    TextPart,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from phantasm.providers.base import LLMProvider, Message
from phantasm.store.messages import MessagesRepo, StoredMessage
from phantasm.store.sessions import SessionsRepo
from phantasm.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, text_result

NO_DELAYS = tuple(0.0 for _ in RETRY_DELAYS_S)

ToolFn = Callable[[Any, ToolContext], Awaitable[str]]


class NoInput(BaseModel):
    """Input of a tool that takes no arguments."""


class FuncTool(Tool[Any]):
    """A tool made from an async function that returns text; handy in tests. Runs unasked."""

    default_policy = "allow"

    def __init__(
        self,
        name: str,
        fn: ToolFn,
        input_model: type[BaseModel] = NoInput,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description or f"The {name} tool."
        self.input_model = input_model
        self._fn = fn

    async def run(self, params: Any, ctx: ToolContext) -> ToolResult:
        return text_result(await self._fn(params, ctx))


class FakeApprover:
    """Answers every approval at once with ``decision`` and remembers what was asked."""

    def __init__(self, decision: Decision = "allow", note: str | None = None) -> None:
        self.decision: Decision = decision
        self.note = note
        self.asked: list[ApprovalAsk] = []

    async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Answer:
        cancel.raise_if_cancelled()
        self.asked.append(ask)
        return Answer(self.decision, "c-1", self.note)


def registry(*tools: Tool[Any]) -> ToolRegistry:
    """A registry holding ``tools``."""
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


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
        tools: ToolRegistry | None = None,
        *,
        max_steps: int | None = None,
        text: str = "hi",
        client: ClientInfo | None = None,
        retry_delays: Sequence[float] = NO_DELAYS,
        approvals: Approver | None = None,
        tool_runs: ToolRunRecorder | None = None,
        usage: UsageRecorder | None = None,
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
            events=self.bus,
            tools=tools,
            retry_delays=retry_delays,
            approvals=approvals,
            tool_runs=tool_runs,
            usage=usage,
        )

    async def run(
        self,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        *,
        max_steps: int | None = None,
        text: str = "hi",
        client: ClientInfo | None = None,
        approvals: Approver | None = None,
        tool_runs: ToolRunRecorder | None = None,
    ) -> Turn:
        """Build a turn and run it to the end."""
        turn = self.turn(
            provider,
            tools,
            max_steps=max_steps,
            text=text,
            client=client,
            approvals=approvals,
            tool_runs=tool_runs,
        )
        await turn.run()
        return turn

    @property
    def bus(self) -> EventBus:
        """A bus that delivers events to this harness's client."""
        return EventBus(Audience([self.client]))

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
