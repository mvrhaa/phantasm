"""The tool interface: ``Tool``, its result and context, and the registry that runs tools."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ValidationError

from zero.approvals.manager import ApprovalAsk, Approver
from zero.config import Config, Policy
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.protocol.messages import ContentPart, TextPart
from zero.providers.base import ToolCall, ToolSpec
from zero.tools.permissions import PathGuard, decide

MAX_OUTPUT_CHARS = 30_000
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

    def label(self, params: InputT) -> str:
        """A short line for the UI, such as ``$ ls -la`` or ``~/notes.md``."""
        return self.name

    async def approval_reason(self, params: InputT, ctx: ToolContext) -> str | None:
        """Why this particular call needs the user's approval whatever the rules say.

        None for an ordinary call. A tool returns a reason for calls riskier
        than it usually is, such as reading a file that may hold secrets; even
        ``bypass_approvals`` does not skip such a call.
        """
        return None

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
    content: list[ContentPart] = [
        TextPart(text=truncate_text(part.text)) if isinstance(part, TextPart) else part
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


class ToolRegistry:
    """The tools the model may call, found by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        """Add a tool. Raises ``ValueError`` if its name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        """The tool with this name, or None."""
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """Describe every tool for the model, in the order they were registered."""
        return [tool.spec() for tool in self._tools.values()]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Validate the input, check the call is permitted, run it and cut the output to size.

        Whether the call may run is decided by ``decide`` from the tool's
        policy, the config and the tool's own ``approval_reason``. Nothing
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
            reason = await tool.approval_reason(params, ctx)
            policy = decide(
                tool.name,
                tool.default_policy,
                ctx.config.permissions,
                needs_approval=reason is not None,
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
                    label=tool.label(params),
                    input=call.input,
                    reason=why,
                )
                answer = await ctx.approvals.request(ask, ctx.cancel)
                if answer.decision == "deny":
                    return denied_result(answer.note)
            result = await tool.run(params, ctx)
        except TurnCancelled:
            return cancelled_result()
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, type(exc).__name__)
            return error_result(f"Tool {call.name!r} failed: {str(exc) or type(exc).__name__}")
        return _truncated(result)
