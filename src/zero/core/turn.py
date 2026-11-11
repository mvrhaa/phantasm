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
from zero.core.expression import EXPRESSION_SET, EXPRESSION_TOOL, ExpressionTool
from zero.protocol.messages import (
    ClientInfo,
    ContentPart,
    Expression,
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
from zero.tools.base import ToolContext, ToolRegistry, ToolResult, error_result, text_result

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


def _only_expression(calls: list[ToolCall], results: list[ToolResult]) -> bool:
    """True if every call of a step set the expression, and every one worked."""
    return all(
        call.name == EXPRESSION_TOOL and not result.is_error
        for call, result in zip(calls, results, strict=True)
    )


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
        tools: ToolRegistry | None = None,
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
        self._tools = tools if tools is not None else ToolRegistry()
        self._reply = reply
        self._retry_delays = tuple(retry_delays)
        character = client.character if client is not None else None
        self._expression = ExpressionTool(character) if character is not None else None
        self._tool_context = ToolContext(
            session_id=session_id,
            turn_id=self.id,
            client_id=client.client_id if client is not None else None,
            cancel=self.cancel,
            config=config,
        )

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

        # Providers reject empty assistant messages, so an empty answer is kept
        # only as a mark that the turn broke off.
        if progress.step_text or status in ("error", "cancelled"):
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
                expression=self._expression.final if self._expression is not None else None,
                usage=progress.usage,
                error=error,
            )
        )

    async def _steps(self, progress: _Progress) -> bool:
        """Call the model step by step until it answers without asking for tools.

        A step that has already written its answer and only sets the expression
        also ends the turn: calling the model again would just cost time.

        A turn makes at most ``max_steps`` model calls. If the model still
        wants tools when only one call is left, that call goes without tools so
        the model has to sum up. Returns True when the limit was reached.
        """
        prompt = await self._build_prompt()
        specs = self._tools.specs()
        if self._expression is not None:
            specs.append(self._expression.spec)
        max_steps = max(self._config.agent.max_steps, 1)

        for _ in range(max_steps - 1):
            calls = await self._call_model(prompt, specs, progress)
            if not calls:
                return False
            answered = bool(progress.step_text)
            results = await self._run_tools(prompt, calls, progress)
            if answered and _only_expression(calls, results):
                return False

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
    ) -> list[ToolResult]:
        """Run the calls one by one, add them with their results to the conversation.

        The assistant message and its tool results are saved only after every
        call is done, so the history never holds a call without its result.
        Once the turn is cancelled, the calls that have not started are not run:
        they get a result saying so, and the turn stops after the step is saved.
        """
        results: list[ToolResult] = []
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
        return results

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        """Run one call, telling clients when it starts and how it ends.

        ``set_expression`` is handled by the turn itself and is marked hidden,
        so clients do not show it as a separate tool line.
        """
        expression = self._expression if call.name == EXPRESSION_TOOL else None
        await self._events.publish(
            ToolStarted(
                session_id=self.session_id,
                turn_id=self.id,
                call_id=call.id,
                tool=call.name,
                input=call.input,
                hidden=expression is not None,
            )
        )
        if expression is not None:
            result = await self._set_expression(expression, call)
        else:
            result = await self._tools.execute(call, self._tool_context)
        await self._events.publish(
            ToolFinished(
                session_id=self.session_id,
                turn_id=self.id,
                call_id=call.id,
                status="error" if result.is_error else "ok",
            )
        )
        return result

    async def _set_expression(self, expression: ExpressionTool, call: ToolCall) -> ToolResult:
        """Change the character's expression and tell the clients at once."""
        try:
            state = expression.apply(call.input)
        except ValueError as exc:
            return error_result(str(exc))
        await self._events.publish(
            Expression(
                session_id=self.session_id, turn_id=self.id, pose=state.pose, mood=state.mood
            )
        )
        return text_result(EXPRESSION_SET)

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
