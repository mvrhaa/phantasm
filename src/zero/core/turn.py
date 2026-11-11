"""The agent loop: one turn from a user message to the final answer."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from zero import clock
from zero.approvals.manager import Approver
from zero.config import Config
from zero.core.attachments import prepare_for_model
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.core.context import build_history, build_system_prompt
from zero.core.events import EventBus, ReplyTo
from zero.core.expression import EXPRESSION_TOOL, ExpressionTool
from zero.core.tool_runner import ToolRunner, ToolRunRecorder
from zero.protocol.messages import (
    ClientInfo,
    ContentPart,
    FilePart,
    TextPart,
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
from zero.store.memory import Memory
from zero.store.messages import MessagesRepo, Source
from zero.store.sessions import SessionsRepo
from zero.tools.base import (
    ApprovalRules,
    ToolContext,
    ToolRegistry,
    ToolResult,
    error_result,
)
from zero.tools.client_actions import (
    COMPUTER_PREFIX,
    ActionCaller,
    ClientActionTool,
    action_tools,
)
from zero.tools.permissions import PathGuard

HISTORY_FETCH_LIMIT = 500
PROMPT_MEMORIES = 10  # how many memories go into the system prompt
RETRY_DELAYS_S: tuple[float, ...] = (1.0, 4.0)
NOT_RUN = "Not run: the user cancelled the turn."

TurnStatus = Literal["ok", "error", "cancelled", "max_steps"]


class MemorySource(Protocol):
    """Gives the memories for the system prompt."""

    async def top(self, n: int = 10) -> Sequence[Memory]: ...


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
    ``approvals`` asks the user about tool calls that need it; without it
    such calls are refused. ``approval_rules`` holds the "always allow" answers.
    The actions the client declared in ``hello`` join ``tools`` for this turn;
    ``action_caller`` reaches that client when the model calls one.
    ``memory`` gives the facts about the user that go into the system prompt.
    ``tool_runs`` keeps a record of the tool calls for the history.
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
        approvals: Approver | None = None,
        approval_rules: ApprovalRules | None = None,
        tool_runs: ToolRunRecorder | None = None,
        action_caller: ActionCaller | None = None,
        memory: MemorySource | None = None,
        attachments: Sequence[FilePart] = (),
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
        actions = (
            action_tools(client.actions, reply.connection_id, action_caller)
            if client is not None and reply is not None
            else []
        )
        base = tools if tools is not None else ToolRegistry()
        self._tools = base.extended(actions, reserved=[EXPRESSION_TOOL])
        self._used_computer = False  # a computer_* action raises the step limit
        self._reply = reply
        self._memory = memory
        self._attachments = list(attachments)
        self._retry_delays = tuple(retry_delays)
        character = client.character if client is not None else None
        self._expression = ExpressionTool(character) if character is not None else None
        self._tool_context = ToolContext(
            session_id=session_id,
            turn_id=self.id,
            client_id=client.client_id if client is not None else None,
            cancel=self.cancel,
            config=config,
            paths=PathGuard.from_config(config.permissions),
            approvals=approvals,
            approval_rules=approval_rules,
            events=events,
        )
        self._runner = ToolRunner(
            session_id=session_id,
            turn_id=self.id,
            events=events,
            tools=self._tools,
            context=self._tool_context,
            expression=self._expression,
            runs=tool_runs,
        )

    async def run(self) -> None:
        """Run the turn to the end. Never raises: failures end the turn with an error."""
        await self._messages.append(
            self.session_id,
            "user",
            [TextPart(text=self._text), *self._attachments],
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

        A turn makes at most ``max_steps`` model calls, or ``max_steps_computer``
        once it has used a ``computer_*`` client action. If the model still
        wants tools when only one call is left, that call goes without tools so
        the model has to sum up. Returns True when the limit was reached.
        """
        prompt = await self._build_prompt()
        specs = self._tools.specs()
        if self._expression is not None:
            specs.append(self._expression.spec)

        step = 1
        while step < self._step_limit():
            step += 1
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

    def _step_limit(self) -> int:
        agent = self._config.agent
        return max(agent.max_steps_computer if self._used_computer else agent.max_steps, 1)

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
                if call.name.startswith(COMPUTER_PREFIX) and isinstance(
                    self._tools.get(call.name), ClientActionTool
                ):
                    self._used_computer = True
                results.append(await self._runner.run(call))

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

    async def _pause(self, seconds: float) -> None:
        """Sleep, but stop with ``TurnCancelled`` as soon as the turn is cancelled."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.cancel.wait(), timeout=seconds)
        self.cancel.raise_if_cancelled()

    async def _build_prompt(self) -> list[Message]:
        """System prompt followed by the history that fits the token budget."""
        memories = await self._memory.top(PROMPT_MEMORIES) if self._memory is not None else []
        system = build_system_prompt(
            self._client, self._source, [memory.text for memory in memories], now=clock.now()
        )
        stored = await self._messages.list(self.session_id, limit=HISTORY_FETCH_LIMIT)
        history = build_history(stored, self._config.agent.history_token_budget)
        can_see = self._provider.supports_images(self._config.provider.model)
        prompt = [Message(role="system", content=[TextPart(text=system)]), *history]
        return await prepare_for_model(prompt, can_see)
