"""The agent loop: one turn from a user message to the final answer."""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from typing import Literal

from zero.approvals.manager import Approver
from zero.config import Config
from zero.core.attachments import images_to_files
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.core.context import IMAGE_OMITTED
from zero.core.events import EventBus, ReplyTo
from zero.core.expression import EXPRESSION_TOOL, ExpressionTool
from zero.core.model_call import RETRY_DELAYS_S, ModelCaller, Progress, UsageRecorder
from zero.core.prompting import MemorySource, PromptBuilder
from zero.core.tool_runner import ToolRunner, ToolRunRecorder
from zero.protocol.messages import (
    AgentRef,
    ClientInfo,
    ContentPart,
    FilePart,
    ImagePart,
    MessageCommand,
    TextPart,
    TurnFinished,
    TurnStarted,
    TurnSteered,
)
from zero.providers.base import (
    LLMProvider,
    Message,
    ProviderError,
    ToolCall,
)
from zero.providers.registry import small_model
from zero.store.journal import JournalRepo
from zero.store.messages import MessagesRepo, Source
from zero.store.session_memory import SessionMemoryRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
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

NOT_RUN = "Not run: the user cancelled the turn."
IMAGE_UNSEEN = "[an image; this model cannot see images]"
MAX_TOOL_IMAGES = 5

TurnStatus = Literal["ok", "error", "cancelled", "max_steps"]


logger = logging.getLogger(__name__)


def _text_content(text: str) -> list[ContentPart]:
    return [TextPart(text=text)] if text else []


def _only_expression(calls: list[ToolCall], results: list[ToolResult]) -> bool:
    """True if every call of a step set the expression, and every one worked."""
    return all(
        call.name == EXPRESSION_TOOL and not result.is_error
        for call, result in zip(calls, results, strict=True)
    )


def _keep_latest_images(prompt: list[Message], keep: int = MAX_TOOL_IMAGES) -> None:
    """Leave only the latest ``keep`` images of tool results in the prompt.

    A turn that looks at the screen step after step would otherwise send
    every screenshot again with every request.
    """
    left = keep
    for index in range(len(prompt) - 1, -1, -1):
        message = prompt[index]
        if message.role != "tool" or not any(isinstance(p, ImagePart) for p in message.content):
            continue
        content: list[ContentPart] = []
        for part in reversed(message.content):
            if isinstance(part, ImagePart):
                if left > 0:
                    left -= 1
                else:
                    part = TextPart(text=IMAGE_OMITTED)
            content.append(part)
        prompt[index] = message.model_copy(update={"content": content[::-1]})


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
    ``memory`` gives the facts about the user that go into the system prompt;
    ``summaries`` lets a long history be compacted into a summary.
    ``tool_runs`` keeps a record of the tool calls for the history, ``usage`` of the model calls.
    ``agent`` is set when the turn is an agent's work: its prompt gets the agent's role.
    ``command`` came with the user's message: its instructions join this turn's
    system prompt, and it can take the tools away or run the turn on the small model.
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
        session_memory: SessionMemoryRepo | None = None,
        journal: JournalRepo | None = None,
        summaries: SummariesRepo | None = None,
        agent: AgentRef | None = None,
        attachments: Sequence[FilePart] = (),
        reply: ReplyTo | None = None,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        usage: UsageRecorder | None = None,
        command: MessageCommand | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.cancel = CancelToken()
        self.done = False
        self.finished: TurnFinished | None = None
        self.agent = agent
        if agent is not None:
            events = events.for_agent(agent)
        self._text = text
        self._command = command
        has_tools = command is None or command.tools == "all"
        model = (
            small_model(config.provider)
            if command is not None and command.model == "small"
            else config.provider.model
        )
        self._source: Source = source
        self._client = client
        self._config = config
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
        self._used_computer = False
        self._timed_out = False
        self._steers: list[tuple[str, ReplyTo | None]] = []
        self._closed = False
        self._can_see_images = provider.supports_images(model)
        self._reply = reply
        self._prompts = PromptBuilder(
            session_id=session_id,
            client=client,
            source=source,
            config=config,
            provider=provider,
            messages=messages,
            events=events,
            memory=memory,
            session_memory=session_memory,
            journal=journal,
            summaries=summaries,
            agent_name=agent.name if agent is not None else None,
            usage=usage,
            cancel=self.cancel,
            command=command.instructions if command is not None else None,
            model=model,
            tools=has_tools,
        )
        self._attachments = list(attachments)
        character = client.character if client is not None else None
        self._expression = (
            ExpressionTool(character)
            if character is not None and character.expressive and has_tools
            else None
        )
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
            agent=agent,
        )
        self._model = ModelCaller(
            provider=provider,
            model=model,
            purpose="agent" if agent is not None else "turn",
            session_id=session_id,
            turn_id=self.id,
            events=events,
            cancel=self.cancel,
            retry_delays=retry_delays,
            usage=usage,
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
        """Run the turn to the end. Never raises: failures end the turn with an error.

        A turn that runs longer than ``turn_timeout_s`` is stopped like a cancelled
        one (its tools and approvals end with it) and finishes with an error.
        """
        progress = Progress()
        status: TurnStatus = "ok"
        error: str | None = None
        limit = self._config.agent.turn_timeout_s
        timer = asyncio.get_running_loop().call_later(limit, self._time_out)
        try:
            text = [TextPart(text=self._text)] if self._text or not self._attachments else []
            await self._messages.append(
                self.session_id,
                "user",
                [*text, *self._attachments],
                turn_id=self.id,
                status="ok",
                source=self._source,
                command=self._command.name if self._command is not None else None,
            )
            await self._events.publish(
                TurnStarted(session_id=self.session_id, turn_id=self.id, source=self._source),
                reply=self._reply,
            )
            if await self._steps(progress):
                status = "max_steps"
        except TurnCancelled:
            if self._timed_out:
                logger.warning("Turn %s stopped after %s s", self.id, limit)
                status, error = "error", f"the turn ran longer than {limit} s and was stopped"
            else:
                logger.info("Turn %s cancelled", self.id)
                status = "cancelled"
        except ProviderError as exc:
            logger.warning("Turn %s failed: provider error (retryable=%s)", self.id, exc.retryable)
            status, error = "error", exc.message
        except Exception:
            logger.exception("Turn %s failed", self.id)
            status, error = "error", "internal error"

        finally:
            timer.cancel()
            self._closed = True

        try:
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
        except Exception:
            logger.exception("Turn %s could not save its end", self.id)
            if status in ("ok", "max_steps"):
                status, error = "error", "internal error"
        self.done = True
        self.finished = TurnFinished(
            session_id=self.session_id,
            turn_id=self.id,
            status=status,
            text=progress.text,
            expression=self._expression.final if self._expression is not None else None,
            usage=progress.usage,
            error=error,
        )
        await self._events.publish(self.finished)

    def steer(self, text: str, reply: ReplyTo | None = None) -> bool:
        """Add the user's words to this turn; the model gets them before its next step.

        False once the turn is ending: then the words should start a new turn.
        """
        if self._closed or self.done:
            return False
        self._steers.append((text, reply))
        return True

    async def _take_steers(self, prompt: list[Message]) -> None:
        """Put the words the user added while the model worked into the conversation."""
        steers, self._steers = self._steers, []
        for text, reply in steers:
            content: list[ContentPart] = [TextPart(text=text)]
            await self._messages.append(
                self.session_id, "user", content, turn_id=self.id, status="ok", source=self._source
            )
            prompt.append(Message(role="user", content=content))
            await self._events.publish(
                TurnSteered(session_id=self.session_id, turn_id=self.id, text=text), reply=reply
            )

    async def _keep_answer(self, prompt: list[Message], progress: Progress) -> None:
        """The model answered, but the user added something meanwhile: keep the answer
        in the conversation and let the model go on."""
        if progress.step_text:
            content = _text_content(progress.step_text)
            await self._messages.append(
                self.session_id,
                "assistant",
                content,
                turn_id=self.id,
                status="ok",
                source=self._source,
            )
            prompt.append(Message(role="assistant", content=content))
            progress.step_text = ""

    def _time_out(self) -> None:
        self._timed_out = True
        self.cancel.cancel()

    async def _steps(self, progress: Progress) -> bool:
        """Call the model step by step until it answers without asking for tools.

        A step that has already written its answer and only sets the expression
        also ends the turn: calling the model again would just cost time.

        A turn makes at most ``max_steps`` model calls, or ``max_steps_computer``
        once it has used a ``computer_*`` client action. If the model still
        wants tools when only one call is left, that call goes without tools so
        the model has to sum up. Returns True when the limit was reached.
        """
        prompt = await self._prompts.build()
        specs = []
        if self._command is None or self._command.tools == "all":
            specs = self._tools.specs()
            if self._expression is not None:
                specs.append(self._expression.spec)

        step = 1
        while step < self._step_limit():
            step += 1
            await self._take_steers(prompt)
            calls = await self._model.answer(prompt, specs, progress)
            if not calls:
                if not self._steers:
                    self._closed = True
                    return False
                await self._keep_answer(prompt, progress)
                continue
            answered = bool(progress.step_text)
            results = await self._run_tools(prompt, calls, progress)
            if answered and _only_expression(calls, results) and not self._steers:
                self._closed = True
                return False

        await self._take_steers(prompt)
        self._closed = True
        ignored = await self._model.answer(prompt, [], progress)
        if ignored:
            logger.warning(
                "Turn %s: model asked for %d tools on the last step; ignored",
                self.id,
                len(ignored),
            )
        return True

    def _for_model(self, content: list[ContentPart]) -> list[ContentPart]:
        """A tool's result as the model gets it: no images for a model that cannot see them."""
        if self._can_see_images:
            return content
        return [
            TextPart(text=IMAGE_UNSEEN) if isinstance(part, ImagePart) else part for part in content
        ]

    def _step_limit(self) -> int:
        agent = self._config.agent
        return max(agent.max_steps_computer if self._used_computer else agent.max_steps, 1)

    async def _run_tools(
        self, prompt: list[Message], calls: list[ToolCall], progress: Progress
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
            tool_calls=[call.model_dump(exclude_none=True) for call in calls],
            status="ok",
            source=self._source,
        )
        prompt.append(assistant)
        progress.step_text = ""

        for call, result in zip(calls, results, strict=True):
            await self._messages.append(
                self.session_id,
                "tool",
                await images_to_files(result.content),
                turn_id=self.id,
                tool_call_id=call.id,
                is_error=result.is_error,
                status="ok",
                source=self._source,
            )
            prompt.append(
                Message(
                    role="tool",
                    content=self._for_model(result.content),
                    tool_call_id=call.id,
                    is_error=result.is_error,
                )
            )
        _keep_latest_images(prompt)
        self.cancel.raise_if_cancelled()
        return results
