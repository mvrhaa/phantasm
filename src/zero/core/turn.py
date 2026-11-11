"""The agent loop: one turn from a user message to the final answer."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from zero import clock
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.core.context import build_history, build_system_prompt
from zero.core.events import EventBus, ReplyTo
from zero.protocol.messages import (
    ClientInfo,
    ContentPart,
    TextPart,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
    Usage,
)
from zero.protocol.messages import TextDelta as TextDeltaEvent
from zero.providers.base import (
    LLMProvider,
    Message,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)
from zero.store.messages import MessagesRepo, Source
from zero.store.sessions import SessionsRepo

HISTORY_FETCH_LIMIT = 500

logger = logging.getLogger(__name__)


# --- Temporary tools -----------------------------------------------------------

StubToolFn = Callable[[dict[str, Any]], Awaitable[str]]
"""A stub tool: takes the call's input and returns text for the model."""


@dataclass
class StubToolResult:
    """What a tool call gave back to the model."""

    content: list[ContentPart]
    is_error: bool = False


def _error(text: str) -> StubToolResult:
    return StubToolResult(content=[TextPart(text=text)], is_error=True)


class StubToolRegistry:
    """A temporary tool registry for the agent loop; commit 35 replaces it.

    Failures never escape ``execute``: an unknown tool or an exception in a
    tool becomes an error result, so the model can read it and try again.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, StubToolFn]] = {}

    def register(
        self,
        name: str,
        fn: StubToolFn,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Add a tool. Raises ``ValueError`` if the name is already taken."""
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
        )
        self._tools[name] = (spec, fn)

    def specs(self) -> list[ToolSpec]:
        """Describe every tool for the model."""
        return [spec for spec, _ in self._tools.values()]

    async def execute(self, call: ToolCall) -> StubToolResult:
        """Run a call and return its result; errors come back as results too."""
        entry = self._tools.get(call.name)
        if entry is None:
            available = ", ".join(self._tools) or "none"
            return _error(f"Unknown tool {call.name!r}. Available tools: {available}.")
        _, fn = entry
        try:
            output = await fn(call.input)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, type(exc).__name__)
            return _error(f"Tool {call.name!r} failed: {str(exc) or type(exc).__name__}")
        return StubToolResult(content=[TextPart(text=output)])


# --- Turn ----------------------------------------------------------------------


@dataclass
class _Progress:
    """What a turn has produced so far; survives a failure in the middle of a step."""

    text: str = ""  # everything the model wrote during the turn, as streamed
    step_text: str = ""  # what it wrote in the current step and is not saved yet
    usage: Usage = field(default_factory=Usage)


def _text_content(text: str) -> list[ContentPart]:
    return [TextPart(text=text)] if text else []


class Turn:
    """One user message and everything the model does in response to it.

    The turn knows nothing about WebSockets: it reports progress by publishing
    protocol events on the bus. ``reply`` names the request that started it,
    so the sender's ``turn.started`` can point back at that request.
    """

    def __init__(
        self,
        *,
        session_id: str,
        text: str,
        source: Source,
        client: ClientInfo | None,
        config: Config,
        provider: LLMProvider,
        sessions: SessionsRepo,
        messages: MessagesRepo,
        events: EventBus,
        tools: StubToolRegistry | None = None,
        reply: ReplyTo | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.cancel = CancelToken()
        self._text = text
        self._source: Source = source
        self._client = client
        self._config = config
        self._provider = provider
        self._sessions = sessions
        self._messages = messages
        self._events = events
        self._tools = tools if tools is not None else StubToolRegistry()
        self._reply = reply

    async def run(self) -> None:
        """Run the turn to the end. Never raises: failures end the turn with an error."""
        await self._messages.append(
            self.session_id,
            "user",
            [TextPart(text=self._text)],
            turn_id=self.id,
            status="ok",
            source=self._source,
        )
        await self._events.publish(
            TurnStarted(session_id=self.session_id, turn_id=self.id, source=self._source),
            reply=self._reply,
        )

        progress = _Progress()
        hit_limit = False
        error: str | None = None
        try:
            hit_limit = await self._steps(progress)
        except ProviderError as exc:
            logger.warning("Turn %s failed: provider error (retryable=%s)", self.id, exc.retryable)
            error = exc.message
        except Exception:
            logger.exception("Turn %s failed", self.id)
            error = "internal error"

        status: Literal["ok", "error", "max_steps"] = (
            "error" if error is not None else "max_steps" if hit_limit else "ok"
        )
        await self._messages.append(
            self.session_id,
            "assistant",
            _text_content(progress.step_text),
            turn_id=self.id,
            status="error" if error is not None else "ok",
            source=self._source,
        )
        await self._sessions.touch(self.session_id)
        await self._events.publish(
            TurnFinished(
                session_id=self.session_id,
                turn_id=self.id,
                status=status,
                text=progress.text,
                usage=progress.usage,
                error=error,
            )
        )

    async def _steps(self, progress: _Progress) -> bool:
        """Call the model step by step until it answers without asking for tools.

        A turn makes at most ``max_steps`` model calls. If the model still
        wants tools when only one call is left, that call goes without tools so
        the model has to sum up. Returns True when the limit was reached.
        """
        prompt = await self._build_prompt()
        specs = self._tools.specs()
        max_steps = max(self._config.agent.max_steps, 1)

        for _ in range(max_steps - 1):
            calls = await self._stream_step(prompt, specs, progress)
            if not calls:
                return False
            await self._run_tools(prompt, calls, progress)

        ignored = await self._stream_step(prompt, [], progress)
        if ignored:
            logger.warning(
                "Turn %s: model asked for %d tools on the last step; ignored",
                self.id,
                len(ignored),
            )
        return True

    async def _stream_step(
        self, prompt: list[Message], tools: list[ToolSpec], progress: _Progress
    ) -> list[ToolCall]:
        """Stream one answer of the model, publishing its text; return its tool calls."""
        calls: list[ToolCall] = []
        stream = self._provider.stream(
            prompt, tools, self._config.provider.model, cancel=self.cancel
        )
        async for event in stream:
            if isinstance(event, TextDelta):
                progress.text += event.text
                progress.step_text += event.text
                await self._events.publish(
                    TextDeltaEvent(session_id=self.session_id, turn_id=self.id, delta=event.text)
                )
            elif isinstance(event, ToolCallEvent):
                calls.append(event.call)
            elif isinstance(event, UsageEvent):
                progress.usage.input_tokens += event.input_tokens
                progress.usage.output_tokens += event.output_tokens
        return calls

    async def _run_tools(
        self, prompt: list[Message], calls: list[ToolCall], progress: _Progress
    ) -> None:
        """Run the calls one by one, then add them with their results to the conversation.

        The assistant message and its tool results are saved only after every
        call is done, so the history never holds a call without its result.
        """
        results = []
        for call in calls:
            results.append(await self._run_tool(call))

        assistant = Message(
            role="assistant", content=_text_content(progress.step_text), tool_calls=calls
        )
        await self._messages.append(
            self.session_id,
            "assistant",
            assistant.content,
            turn_id=self.id,
            tool_calls=[call.model_dump() for call in calls],
            status="ok",
            source=self._source,
        )
        prompt.append(assistant)
        progress.step_text = ""

        for call, result in zip(calls, results, strict=True):
            await self._messages.append(
                self.session_id,
                "tool",
                result.content,
                turn_id=self.id,
                tool_call_id=call.id,
                is_error=result.is_error,
                status="ok",
                source=self._source,
            )
            prompt.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=call.id,
                    is_error=result.is_error,
                )
            )

    async def _run_tool(self, call: ToolCall) -> StubToolResult:
        """Run one call, telling clients when it starts and how it ends."""
        await self._events.publish(
            ToolStarted(
                session_id=self.session_id,
                turn_id=self.id,
                call_id=call.id,
                tool=call.name,
                input=call.input,
            )
        )
        result = await self._tools.execute(call)
        await self._events.publish(
            ToolFinished(
                session_id=self.session_id,
                turn_id=self.id,
                call_id=call.id,
                status="error" if result.is_error else "ok",
            )
        )
        return result

    async def _build_prompt(self) -> list[Message]:
        """System prompt followed by the history that fits the token budget."""
        system = build_system_prompt(self._client, self._source, now=clock.now())
        stored = await self._messages.list(self.session_id, limit=HISTORY_FETCH_LIMIT)
        history = build_history(stored, self._config.agent.history_token_budget)
        return [Message(role="system", content=[TextPart(text=system)]), *history]


class TurnManager:
    """Runs turns as background tasks and keeps track of the ones in progress."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[str, Turn] = {}

    @property
    def active(self) -> list[Turn]:
        """Turns that are still running."""
        return list(self._turns.values())

    def start(self, turn: Turn) -> None:
        """Start a turn in the background and forget it once it is done."""
        task = asyncio.create_task(turn.run(), name=f"turn-{turn.id}")
        self._tasks[turn.id] = task
        self._turns[turn.id] = turn
        task.add_done_callback(lambda _: self._forget(turn.id))

    def _forget(self, turn_id: str) -> None:
        self._tasks.pop(turn_id, None)
        self._turns.pop(turn_id, None)

    async def wait_idle(self) -> None:
        """Wait until every running turn has finished."""
        while self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def shutdown(self) -> None:
        """Stop all running turns; used when the application shuts down."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
