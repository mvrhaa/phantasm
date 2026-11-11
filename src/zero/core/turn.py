"""The agent loop: one turn from a user message to the final answer."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from zero import clock
from zero.config import Config
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.core.context import build_history, build_system_prompt
from zero.core.events import EventBus, ReplyTo
from zero.core.stub_tools import StubToolRegistry, StubToolResult, error_result
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
RETRY_DELAYS_S: tuple[float, ...] = (1.0, 4.0)
NOT_RUN = "Not run: the user cancelled the turn."

TurnStatus = Literal["ok", "error", "cancelled", "max_steps"]

logger = logging.getLogger(__name__)


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
    ``retry_delays`` are the pauses before each retry of a failed model call.
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
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.cancel = CancelToken()
        self.done = False  # set once everything is saved, just before turn.finished
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
        self._retry_delays = tuple(retry_delays)

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
        status: TurnStatus = "ok"
        error: str | None = None
        try:
            if await self._steps(progress):
                status = "max_steps"
        except TurnCancelled:
            logger.info("Turn %s cancelled", self.id)
            status = "cancelled"
        except ProviderError as exc:
            logger.warning("Turn %s failed: provider error (retryable=%s)", self.id, exc.retryable)
            status, error = "error", exc.message
        except Exception:
            logger.exception("Turn %s failed", self.id)
            status, error = "error", "internal error"

        await self._messages.append(
            self.session_id,
            "assistant",
            _text_content(progress.step_text),
            turn_id=self.id,
            status="ok" if status == "max_steps" else status,
            source=self._source,
        )
        await self._sessions.touch(self.session_id)
        self.done = True
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
            calls = await self._call_model(prompt, specs, progress)
            if not calls:
                return False
            await self._run_tools(prompt, calls, progress)

        ignored = await self._call_model(prompt, [], progress)
        if ignored:
            logger.warning(
                "Turn %s: model asked for %d tools on the last step; ignored",
                self.id,
                len(ignored),
            )
        return True

    async def _call_model(
        self, prompt: list[Message], tools: list[ToolSpec], progress: _Progress
    ) -> list[ToolCall]:
        """Stream one answer, retrying a retryable failure that came before any text.

        Text already sent to clients cannot be taken back, so a failure in the
        middle of an answer ends the turn instead of streaming the answer twice.
        """
        for delay in self._retry_delays:
            try:
                return await self._stream_step(prompt, tools, progress)
            except ProviderError as exc:
                if not exc.retryable or progress.step_text:
                    raise
                logger.warning(
                    "Turn %s: retryable provider error, trying again in %.1f s", self.id, delay
                )
                await self._pause(delay)
        return await self._stream_step(prompt, tools, progress)

    async def _stream_step(
        self, prompt: list[Message], tools: list[ToolSpec], progress: _Progress
    ) -> list[ToolCall]:
        """Stream one answer of the model, publishing its text; return its tool calls."""
        self.cancel.raise_if_cancelled()
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
        # A provider may end its stream quietly on cancellation instead of raising.
        self.cancel.raise_if_cancelled()
        return calls

    async def _run_tools(
        self, prompt: list[Message], calls: list[ToolCall], progress: _Progress
    ) -> None:
        """Run the calls one by one, then add them with their results to the conversation.

        The assistant message and its tool results are saved only after every
        call is done, so the history never holds a call without its result.
        Once the turn is cancelled, the calls that have not started are not run:
        they get a result saying so, and the turn stops after the step is saved.
        """
        results: list[StubToolResult] = []
        for call in calls:
            if self.cancel.cancelled:
                results.append(error_result(NOT_RUN))
            else:
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
        self.cancel.raise_if_cancelled()

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

    async def _pause(self, seconds: float) -> None:
        """Sleep, but stop with ``TurnCancelled`` as soon as the turn is cancelled."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.cancel.wait(), timeout=seconds)
        self.cancel.raise_if_cancelled()

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

    def get(self, turn_id: str) -> Turn | None:
        """The running turn with this id, or None."""
        return self._turns.get(turn_id)

    def busy(self, session_id: str) -> bool:
        """True while a turn of this session is unfinished.

        A turn counts as finished once it has saved everything, just before it
        announces ``turn.finished``, so a client may send the next message as
        soon as it sees that event.
        """
        return any(t.session_id == session_id and not t.done for t in self._turns.values())

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
