"""Tests for the provider-independent types."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from phantasm.protocol.messages import ImagePart, TextPart
from phantasm.providers.base import (
    CancelSignal,
    LLMProvider,
    Message,
    ProviderError,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)

stream_event = TypeAdapter(StreamEvent)

CALL = ToolCall(id="c1", name="read_file", input={"path": "~/notes.md"})


def test_user_message_with_text_and_image() -> None:
    message = Message(
        role="user",
        content=[TextPart(text="look"), ImagePart(mime="image/png", data_b64="AAAA")],
    )

    assert message.tool_calls == []
    assert message.tool_call_id is None
    assert message.is_error is False
    assert Message.model_validate_json(message.model_dump_json()) == message


def test_assistant_message_with_tool_calls() -> None:
    message = Message(role="assistant", tool_calls=[CALL])

    assert message.content == []
    assert Message.model_validate_json(message.model_dump_json()) == message


def test_tool_result_message() -> None:
    message = Message(
        role="tool", content=[TextPart(text="no such file")], tool_call_id="c1", is_error=True
    )

    assert Message.model_validate_json(message.model_dump_json()) == message


def test_system_message() -> None:
    assert Message(role="system", content=[TextPart(text="rules")]).role == "system"


@pytest.mark.parametrize(
    "data",
    [
        {"role": "user", "tool_calls": [{"id": "c1", "name": "read_file"}]},
        {"role": "tool", "content": [{"type": "text", "text": "x"}]},
        {"role": "user", "tool_call_id": "c1"},
        {"role": "assistant", "is_error": True},
        {"role": "robot"},
        {},
    ],
    ids=[
        "tool-calls-from-user",
        "tool-without-call-id",
        "call-id-on-user",
        "is-error-on-assistant",
        "unknown-role",
        "no-role",
    ],
)
def test_invalid_message_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate(data)


def test_tool_call_input_defaults_to_empty() -> None:
    assert ToolCall(id="c1", name="set_expression").input == {}


def test_tool_spec_carries_a_json_schema() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    spec = ToolSpec(name="read_file", description="Read a file", input_schema=schema)

    assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.input_schema == schema


@pytest.mark.parametrize(
    "event",
    [
        TextDelta(text="Hel"),
        ToolCallEvent(call=CALL),
        UsageEvent(input_tokens=10, output_tokens=3),
        StopEvent(reason="end"),
        StopEvent(reason="tool_use"),
    ],
    ids=lambda event: event.type,
)
def test_stream_event_survives_a_round_trip(event: BaseModel) -> None:
    restored = stream_event.validate_json(event.model_dump_json())

    assert type(restored) is type(event)
    assert restored == event


def test_stream_event_types() -> None:
    assert [
        TextDelta(text="").type,
        ToolCallEvent(call=CALL).type,
        UsageEvent().type,
        StopEvent(reason="end").type,
    ] == ["text_delta", "tool_call", "usage", "stop"]


@pytest.mark.parametrize("reason", ["end", "tool_use", "max_tokens", "error"])
def test_stop_reasons_from_the_spec(reason: str) -> None:
    assert stream_event.validate_python({"type": "stop", "reason": reason}).reason == reason


@pytest.mark.parametrize(
    "data",
    [
        {"type": "stop", "reason": "bored"},
        {"type": "text_delta"},
        {"type": "tool_call", "call": {"name": "read_file"}},
        {"type": "thinking", "text": "hmm"},
        {"text": "no type"},
    ],
    ids=["stop-reason", "delta-without-text", "call-without-id", "unknown-type", "no-type"],
)
def test_invalid_stream_event_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        stream_event.validate_python(data)


def test_usage_defaults_to_zero() -> None:
    assert (UsageEvent().input_tokens, UsageEvent().output_tokens) == (0, 0)


class NeverCancelled:
    cancelled = False

    def raise_if_cancelled(self) -> None:
        return None


class TinyProvider:
    """The smallest class that satisfies LLMProvider."""

    name = "tiny"

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text=f"{len(messages)} messages for {model}")
        yield StopEvent(reason="end")

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        return None

    def supports_images(self, model: str) -> bool:
        return False


async def test_a_conforming_class_can_be_used_as_a_provider() -> None:
    provider: LLMProvider = TinyProvider()
    messages = [Message(role="user", content=[TextPart(text="hi")])]

    events = [e async for e in provider.stream(messages, [], "m-1", cancel=NeverCancelled())]

    assert isinstance(provider, LLMProvider)
    assert events == [TextDelta(text="1 messages for m-1"), StopEvent(reason="end")]
    assert await provider.count_tokens(messages, []) is None
    assert provider.supports_images("m-1") is False


def test_an_unrelated_class_is_not_a_provider() -> None:
    class NotAProvider:
        name = "nope"

    assert not isinstance(NotAProvider(), LLMProvider)


def test_provider_error_is_not_retryable_by_default() -> None:
    error = ProviderError("bad request")

    assert isinstance(error, Exception)
    assert (error.message, error.retryable, str(error)) == ("bad request", False, "bad request")


def test_provider_error_can_be_retryable() -> None:
    with pytest.raises(ProviderError) as excinfo:
        raise ProviderError("server overloaded", retryable=True)

    assert excinfo.value.retryable is True
