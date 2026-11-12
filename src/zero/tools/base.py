"""The tool interface: ``Tool``, its result and context, and the registry that runs tools."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from zero.approvals.manager import ApprovalAsk, Approver
from zero.config import Config, Policy
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.protocol.messages import (
    AgentRef,
    ContentPart,
    Envelope,
    FilePart,
    ImagePart,
    TextPart,
)
from zero.providers.base import ToolCall, ToolSpec
from zero.secrets import mask_secrets
from zero.tools.permissions import PathGuard, decide

MAX_OUTPUT_CHARS = 30_000
PREVIEW_CHARS = 300
CANCELLED = "Cancelled: the user stopped the turn."
NEEDS_APPROVAL = "so this call needs the user's approval, and no one can be asked here."
DENIED = "The user denied this call."

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """What a tool gives back to the model."""

    content: list[ContentPart] = field(default_factory=list)
    is_error: bool = False
    summary: str = ""  # a few words for the UI: "exit 0", "+12 -3", "5 matches"
    denied: bool = False  # the user said no, so the tool did not run
    cancelled: bool = False  # the turn was cancelled before the call could finish


def output_preview(result: ToolResult, limit: int = PREVIEW_CHARS) -> str:
    """The start of a result for the UI: its text, or what kind of content it holds."""
    text = " ".join(
        " ".join(part.text.split()) for part in result.content if isinstance(part, TextPart)
    ).strip()
    if not text:
        kinds = [
            "[image]" if isinstance(part, ImagePart) else f"[file {part.name}]"
            for part in result.content
            if isinstance(part, ImagePart | FilePart)
        ]
        text = " ".join(kinds)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def text_result(text: str, *, summary: str = "") -> ToolResult:
    """A successful result made of one piece of text."""
    return ToolResult(content=[TextPart(text=text)], summary=summary)


def error_result(text: str) -> ToolResult:
    """A result that tells the model the call did not work and why."""
    return ToolResult(content=[TextPart(text=text)], is_error=True)


def denied_result(note: str | None = None) -> ToolResult:
    """The user refused the call; their note, if any, goes to the model too."""
    text = f"{DENIED} Their note: {note}" if note else DENIED
    return ToolResult(content=[TextPart(text=text)], is_error=True, denied=True)


def cancelled_result() -> ToolResult:
    """The turn was cancelled while the call ran or waited for approval."""
    return ToolResult(content=[TextPart(text=CANCELLED)], is_error=True, cancelled=True)


class EventPublisher(Protocol):
    """Sends an event to every connected client."""

    async def publish(self, event: Envelope) -> None: ...


class ApprovalRules(Protocol):
    """Where the user's "always allow" answers are kept."""

    async def always_allowed(self) -> set[str]: ...

    async def allow_always(self, tool: str) -> object: ...


@dataclass(frozen=True)
class Preview:
    """What a tool says about a call before it runs; see ``Tool.preview``."""

    refusal: str | None = None  # the call cannot run; the model gets this and no one is asked
    approval_reason: str | None = None  # ask the user even if bypass or a saved rule says allow
    diff: str | None = None  # shown in the approval card, for changes to files
    label: str | None = None  # a better line for the approval card than ``Tool.label`` gives


@dataclass(frozen=True)
class ToolContext:
    """What a running tool may use besides its input."""

    session_id: str
    turn_id: str
    client_id: str | None  # None when no client started the turn, e.g. a scheduled one
    cancel: CancelToken
    config: Config
    paths: PathGuard  # every path a tool touches goes through ``paths.check``
    approvals: Approver | None = None  # None when there is no one to ask
    approval_rules: ApprovalRules | None = None  # None: no saved rules apply
    events: EventPublisher | None = None  # for tools that tell clients about a change
    agent: AgentRef | None = None  # set when the turn is an agent's


class Tool[InputT: BaseModel](ABC):
    """A tool the model can call.

    A subclass sets ``name``, ``description`` and ``input_model`` and
    implements ``run``. The registry validates the input against
    ``input_model`` before ``run`` sees it, so ``run`` gets a ready model.
    """

    name: str
    description: str
    input_model: type[InputT]
    default_policy: Policy = "ask"
    config_tightens_only = False  # True: the config may make the policy stricter, never looser

    def label(self, params: InputT) -> str:
        """A short line for the UI, such as ``$ ls -la`` or ``~/notes.md``."""
        return self.name

    async def preview(self, params: InputT, ctx: ToolContext) -> Preview:
        """Look at a call before the registry decides whether it may run.

        The default says nothing. A tool can refuse a call that cannot work
        (no one is asked then), demand the user's approval for a call riskier
        than it usually is, and give a diff to show in the approval card.
        """
        return Preview()

    def spec(self) -> ToolSpec:
        """The tool as the model sees it; the schema comes from ``input_model``."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    @abstractmethod
    async def run(self, params: InputT, ctx: ToolContext) -> ToolResult:
        """Do the work. A failure may be returned as ``error_result`` or raised."""


def truncate_text(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and the tail of a long text and say how much was cut out."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n[… {len(text) - limit} chars truncated …]\n{text[-tail:]}"


def _truncated(result: ToolResult) -> ToolResult:
    """The result cut to size, with any API key from the .env file masked."""
    content: list[ContentPart] = [
        TextPart(text=truncate_text(mask_secrets(part.text)))
        if isinstance(part, TextPart)
        else part
        for part in result.content
    ]
    return replace(result, content=content)


def _describe(exc: ValidationError) -> str:
    """Name each invalid field and what is wrong with it, without echoing values."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)


async def _remember(tool: str, rules: ApprovalRules | None) -> None:
    """Save an "always allow" answer. The call itself runs even if saving fails."""
    if rules is None:
        return
    try:
        await rules.allow_always(tool)
    except Exception:
        logger.exception("Could not save the always-allow rule for %s", tool)


class ToolRegistry:
    """The tools the model may call, found by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        """Add a tool. Raises ``ValueError`` if its name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def extended(self, tools: Iterable[Tool[Any]], reserved: Iterable[str] = ()) -> "ToolRegistry":
        """A new registry with this one's tools and then ``tools``.

        A tool whose name is already taken, here or in ``reserved``, is left
        out with a warning: an extra tool never replaces one already there.
        """
        result = ToolRegistry()
        result._tools = dict(self._tools)
        taken = set(reserved)
        for tool in tools:
            if tool.name in result._tools or tool.name in taken:
                logger.warning("Tool %r is already taken; the new one is left out", tool.name)
                continue
            result._tools[tool.name] = tool
        return result

    def without(self, prefix: str) -> "ToolRegistry":
        """A new registry with every tool except those whose name starts with ``prefix``."""
        registry = ToolRegistry()
        for name, tool in self._tools.items():
            if not name.startswith(prefix):
                registry.register(tool)
        return registry

    def get(self, name: str) -> Tool[Any] | None:
        """The tool with this name, or None."""
        return self._tools.get(name)

    def label(self, call: ToolCall) -> str:
        """The UI line of a call, from the tool's ``label``; the tool name if the input is bad."""
        tool = self._tools.get(call.name)
        if tool is None:
            return call.name
        try:
            return tool.label(tool.input_model.model_validate(call.input))
        except ValidationError:
            return call.name

    def specs(self) -> list[ToolSpec]:
        """Describe every tool for the model, in the order they were registered."""
        return [tool.spec() for tool in self._tools.values()]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Validate the input, check the call is permitted, run it and cut the output to size.

        Whether the call may run is decided by ``decide`` from the tool's
        policy, the config and the tool's own ``preview`` of the call. Nothing
        escapes: an unknown tool, invalid input, a call that is not permitted or
        a failing tool gives an error result that the model can read and react to.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return error_result(f"Unknown tool {call.name!r}. Available tools: {available}.")
        try:
            params = tool.input_model.model_validate(call.input)
        except ValidationError as exc:
            return error_result(
                f"Invalid input for {call.name!r}: {_describe(exc)}. Fix it and call again."
            )
        try:
            preview = await tool.preview(params, ctx)
            if preview.refusal is not None:
                return error_result(preview.refusal)
            reason = preview.approval_reason
            rules = ctx.approval_rules
            policy = decide(
                tool.name,
                tool.default_policy,
                ctx.config.permissions,
                await rules.always_allowed() if rules is not None else (),
                needs_approval=reason is not None,
                config_tightens_only=tool.config_tightens_only,
            )
            if policy == "deny":
                return error_result(f"{tool.name} is turned off in the settings.")
            if policy == "ask":
                why = reason or f"{tool.name} is set to ask the user first"
                if ctx.approvals is None:
                    return error_result(f"{why}, {NEEDS_APPROVAL}")
                ask = ApprovalAsk(
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    tool=tool.name,
                    label=preview.label or tool.label(params),
                    input=call.input,
                    reason=why,
                    diff=preview.diff,
                    agent=ctx.agent,
                )
                answer = await ctx.approvals.request(ask, ctx.cancel)
                if answer.decision == "deny":
                    return denied_result(answer.note)
                if answer.decision == "always":
                    await _remember(tool.name, rules)
            result = await tool.run(params, ctx)
        except TurnCancelled:
            return cancelled_result()
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, type(exc).__name__)
            reason = mask_secrets(str(exc)) or type(exc).__name__
            return error_result(f"Tool {call.name!r} failed: {reason}")
        return _truncated(result)
