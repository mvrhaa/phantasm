"""Provider-independent message and stream types.

The brain speaks only this format. Every provider adapter translates it to
its own API and back.
"""

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from zero.protocol.messages import ContentPart


class ToolCall(BaseModel):
    """A request from the model to run a tool."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """A tool as it is described to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema


class Message(BaseModel):
    """One message of a conversation sent to the model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentPart] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    is_error: bool = False  # tool only

    @model_validator(mode="after")
    def _fields_match_role(self) -> Self:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only an assistant message can carry tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message needs tool_call_id")
        if self.role != "tool" and (self.tool_call_id is not None or self.is_error):
            raise ValueError("tool_call_id and is_error belong to tool messages only")
        return self


# --- Stream events -------------------------------------------------------------


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


# --- Provider interface --------------------------------------------------------


class ProviderError(Exception):
    """A provider request failed. ``retryable`` tells whether trying again may help."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class CancelSignal(Protocol):
    """What a provider needs to notice that the turn was cancelled."""

    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...

    async def wait(self) -> None:
        """Return once the turn is cancelled; providers race their network waits against it."""
        ...


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
