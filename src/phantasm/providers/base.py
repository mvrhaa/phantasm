"""Provider-independent message and stream types.

The brain speaks only this format. Every provider adapter translates it to
its own API and back.
"""

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from phantasm.protocol.messages import ContentPart


class ToolCall(BaseModel):
    """A request from the model to run a tool.

    ``extra`` is what the provider attached to the call and wants back with it:
    Gemini 3 refuses the next step without the call's thought signature.
    """

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    """A tool as it is described to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Message(BaseModel):
    """One message of a conversation sent to the model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentPart] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False

    @model_validator(mode="after")
    def _fields_match_role(self) -> Self:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only an assistant message can carry tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message needs tool_call_id")
        if self.role != "tool" and (self.tool_call_id is not None or self.is_error):
            raise ValueError("tool_call_id and is_error belong to tool messages only")
        return self


class TextDelta(BaseModel):
    """The next chunk of the model's text."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallEvent(BaseModel):
    """A complete tool call; the adapter assembles its arguments from deltas."""

    type: Literal["tool_call"] = "tool_call"
    call: ToolCall


class UsageEvent(BaseModel):
    """Tokens spent by this request."""

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0


class StopEvent(BaseModel):
    """The stream ended and why."""

    type: Literal["stop"] = "stop"
    reason: Literal["end", "tool_use", "max_tokens", "error"]


StreamEvent = Annotated[
    TextDelta | ToolCallEvent | UsageEvent | StopEvent, Field(discriminator="type")
]
"""Anything a provider yields while streaming an answer."""


class ProviderError(Exception):
    """A provider request failed. ``retryable`` tells whether trying again may help.

    ``retry_after`` is how many seconds the provider asked to wait first, if it said.
    """

    def __init__(
        self, message: str, *, retryable: bool = False, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after


class CancelSignal(Protocol):
    """What a provider needs to notice that the turn was cancelled."""

    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...

    async def wait(self) -> None:
        """Return once the turn is cancelled; providers race their network waits against it."""
        ...


def writes_prose(provider: object) -> bool:
    """False for a provider that cannot write a title or a summary (it says so itself)."""
    return bool(getattr(provider, "writes_prose", True))


@runtime_checkable
class LLMProvider(Protocol):
    """The interface every provider adapter implements."""

    name: str

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        """Yield the model's answer piece by piece.

        Implemented as an async generator. Failures are raised as
        ``ProviderError``; the stream stops early once ``cancel`` is set.
        """
        ...

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        """Return the prompt size in tokens, or None if the provider cannot tell."""
        ...

    def supports_images(self, model: str) -> bool:
        """Tell whether the model accepts image content parts."""
        ...
