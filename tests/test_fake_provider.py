"""Tests for the scripted FakeProvider."""

import pytest

from zero.core.cancellation import CancelToken, TurnCancelled
from zero.protocol.messages import TextPart
from zero.providers.base import (
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
from zero.providers.fake import FakeProvider

USER = Message(role="user", content=[TextPart(text="hi")])
TOOL = ToolSpec(name="read_file", description="Read a file", input_schema={"type": "object"})
CALL = ToolCall(id="c1", name="read_file", input={"path": "a.txt"})


async def play(
    provider: FakeProvider,
    messages: list[Message] | None = None,
    tools: list[ToolSpec] | None = None,
    cancel: CancelToken | None = None,
    **kwargs: int,
) -> list[StreamEvent]:
    stream = provider.stream(
        messages if messages is not None else [USER],
        tools or [],
        "fake-1",
        cancel=cancel or CancelToken(),
        **kwargs,
    )
    return [event async for event in stream]


def test_fake_provider_satisfies_the_interface() -> None:
    provider: LLMProvider = FakeProvider()

    assert isinstance(provider, LLMProvider)
    assert provider.name == "fake"


async def test_step_is_yielded_in_order() -> None:
    step = [
        TextDelta(text="Hel"),
        TextDelta(text="lo"),
        UsageEvent(input_tokens=5, output_tokens=2),
    ]

    events = await play(FakeProvider([[*step, StopEvent(reason="end")]]))

    assert events == [*step, StopEvent(reason="end")]


async def test_each_call_plays_the_next_step() -> None:
    provider = FakeProvider([[TextDelta(text="first")], [TextDelta(text="second")]])

    first = await play(provider)
    second = await play(provider)

    assert first[0] == TextDelta(text="first")
    assert second[0] == TextDelta(text="second")
    assert provider.steps_left == 0


async def test_text_step_gets_an_end_stop() -> None:
    events = await play(FakeProvider([[TextDelta(text="hi")]]))

    assert events[-1] == StopEvent(reason="end")


async def test_tool_step_gets_a_tool_use_stop() -> None:
    events = await play(FakeProvider([[TextDelta(text="let me look"), ToolCallEvent(call=CALL)]]))

    assert events == [
        TextDelta(text="let me look"),
        ToolCallEvent(call=CALL),
        StopEvent(reason="tool_use"),
    ]


async def test_explicit_stop_is_kept() -> None:
    events = await play(FakeProvider([[TextDelta(text="cut"), StopEvent(reason="max_tokens")]]))

    assert events == [TextDelta(text="cut"), StopEvent(reason="max_tokens")]


async def test_empty_step_is_just_a_stop() -> None:
    assert await play(FakeProvider([[]])) == [StopEvent(reason="end")]


async def test_calls_record_what_was_sent() -> None:
    provider = FakeProvider([[TextDelta(text="a")], [TextDelta(text="b")]])
    second_messages = [USER, Message(role="assistant", content=[TextPart(text="a")]), USER]

    await play(provider, [USER], [TOOL], max_tokens=100)
    await play(provider, second_messages)

    assert len(provider.calls) == 2
    assert provider.calls[0].messages == [USER]
    assert provider.calls[0].tools == [TOOL]
    assert (provider.calls[0].model, provider.calls[0].max_tokens) == ("fake-1", 100)
    assert provider.calls[1].messages == second_messages
    assert provider.calls[1].tools == []
    assert provider.calls[1].max_tokens == 4096


async def test_recorded_messages_are_snapshots() -> None:
    provider = FakeProvider([[TextDelta(text="a")]])
    messages = [Message(role="user", content=[TextPart(text="original")])]

    await play(provider, messages)
    messages.append(USER)
    messages[0].content[0] = TextPart(text="changed later")

    assert provider.calls[0].messages == [Message(role="user", content=[TextPart(text="original")])]


async def test_running_past_the_script_is_a_provider_error() -> None:
    provider = FakeProvider([[TextDelta(text="only step")]])
    await play(provider)

    with pytest.raises(ProviderError, match="no step 2") as excinfo:
        await play(provider)

    assert excinfo.value.retryable is False
    assert len(provider.calls) == 2


async def test_provider_without_a_script_has_nothing_to_play() -> None:
    with pytest.raises(ProviderError):
        await play(FakeProvider())


async def test_cancel_before_the_stream_yields_nothing() -> None:
    cancel = CancelToken()
    cancel.cancel()
    provider = FakeProvider([[TextDelta(text="never")]])

    with pytest.raises(TurnCancelled):
        await play(provider, cancel=cancel)

    assert len(provider.calls) == 1


async def test_cancel_between_events_stops_the_stream() -> None:
    cancel = CancelToken()
    provider = FakeProvider(
        [[TextDelta(text="one"), TextDelta(text="two"), TextDelta(text="three")]]
    )
    received: list[StreamEvent] = []

    with pytest.raises(TurnCancelled):
        async for event in provider.stream([USER], [], "fake-1", cancel=cancel):
            received.append(event)
            if len(received) == 2:
                cancel.cancel()

    assert received == [TextDelta(text="one"), TextDelta(text="two")]


async def test_script_is_copied_at_construction() -> None:
    script = [[TextDelta(text="original")]]
    provider = FakeProvider(script)

    script[0].append(TextDelta(text="added later"))
    script.append([TextDelta(text="extra step")])

    assert await play(provider) == [TextDelta(text="original"), StopEvent(reason="end")]
    assert provider.steps_left == 0


async def test_fake_cannot_count_tokens_and_accepts_images() -> None:
    provider = FakeProvider()

    assert await provider.count_tokens([USER], [TOOL]) is None
    assert provider.supports_images("fake-1") is True
