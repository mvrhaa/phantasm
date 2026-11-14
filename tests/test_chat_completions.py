"""Tests of the /chat/completions provider. No network: httpx gets a MockTransport.

The streams imitate what Google sent for Gemma 4 when it was tried live:
thinking in marked chunks, the closing tag unmarked, ``stop`` after tool calls,
token counts in every chunk.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from tests.conftest import FrozenClock
from zero.core.cancellation import CancelToken
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
from zero.providers.chat_completions import ChatCompletionsProvider
from zero.providers.presets import PRESETS
from zero.secrets import env_file

KEY = "AIza-test-0123456789abcdef"
MESSAGES = [
    Message(role="system", content=[TextPart(text="You are Zero.")]),
    Message(role="user", content=[TextPart(text="Какая погода в Праге?")]),
]
WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather in a city.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)

Handler = Callable[[httpx.Request], Any]


@pytest.fixture(autouse=True)
def keys() -> None:
    path = env_file()
    path.write_text(f"GEMINI_API_KEY={KEY}\nMISTRAL_API_KEY=mistral-key-0123456789\n")
    path.chmod(0o600)
    os.utime(path)


def provider(handler: Handler, preset: str = "google", **kwargs: Any) -> ChatCompletionsProvider:
    """A provider answered by ``handler``; no pause between requests unless asked for."""
    kwargs.setdefault("min_interval_s", 0)
    return ChatCompletionsProvider(
        PRESETS[preset], transport=httpx.MockTransport(handler), **kwargs
    )


def sse(*chunks: dict[str, Any] | str) -> bytes:
    """Server-sent events; a string starting with ":" is a comment line, sent as is."""
    lines = [
        f"{c}\n\n"
        if isinstance(c, str) and c.startswith(":")
        else f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n"
        for c in chunks
    ]
    return "".join(lines).encode()


def chunk(
    content: str | None = None,
    *,
    thought: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if thought:
        delta["extra_content"] = {"google": {"thought": True}}
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    result: dict[str, Any] = {"object": "chat.completion.chunk", "choices": [choice]}
    if usage:
        result["usage"] = usage
    return result


def answering(body: bytes, status: int = 200) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return handle


async def collect(
    p: ChatCompletionsProvider,
    *,
    tools: list[ToolSpec] | None = None,
    cancel: CancelToken | None = None,
    max_tokens: int = 4096,
) -> list[StreamEvent]:
    stream = p.stream(
        MESSAGES,
        tools or [],
        "gemma-4-31b-it",
        max_tokens=max_tokens,
        cancel=cancel or CancelToken(),
    )
    return [event async for event in stream]


async def failure(p: ChatCompletionsProvider) -> ProviderError:
    with pytest.raises(ProviderError) as excinfo:
        await collect(p)
    return excinfo.value


async def test_the_request_to_google() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(chunk("hi", finish="stop"), "[DONE]"))

    await collect(provider(handle), tools=[WEATHER], max_tokens=1000)

    (request,) = seen
    body = json.loads(request.content)
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["user-agent"].startswith("Zero/")
    assert body["model"] == "gemma-4-31b-it"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 1000 + 4096
    assert body["messages"][1] == {"role": "user", "content": "Какая погода в Праге?"}
    assert body["tools"][0]["function"]["name"] == "get_weather"


async def test_mistral_gets_no_stream_options_and_no_tools_list_when_there_are_none() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(chunk("hi", finish="stop"), "[DONE]"))

    await collect(provider(handle, "mistral"), max_tokens=1000)

    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.mistral.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer mistral-key-0123456789"
    assert "stream_options" not in body and "tools" not in body
    assert body["max_tokens"] == 1000


async def test_a_local_ollama_needs_no_key() -> None:
    env_file().unlink()
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(chunk("hi", finish="stop"), "[DONE]"))

    await collect(provider(handle, "ollama"))

    assert str(seen[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in seen[0].headers


async def test_the_key_is_read_from_env_for_every_request() -> None:
    keys_seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        keys_seen.append(request.headers["authorization"])
        return httpx.Response(200, content=sse(chunk("hi", finish="stop"), "[DONE]"))

    p = provider(handle)
    await collect(p)
    env_file().write_text("GEMINI_API_KEY=a-brand-new-key-0000\n")
    await collect(p)

    assert keys_seen == [f"Bearer {KEY}", "Bearer a-brand-new-key-0000"]


async def test_a_missing_key_is_named_and_nothing_is_sent() -> None:
    env_file().unlink()
    sent: list[httpx.Request] = []

    error = await failure(provider(lambda request: sent.append(request)))

    assert error.message == f"GEMINI_API_KEY is missing in {env_file()}"
    assert not error.retryable and not sent


def test_images_follow_the_preset_unless_the_config_says_otherwise() -> None:
    def never(request: httpx.Request) -> None:
        raise AssertionError

    assert provider(never).supports_images("gemma-4-31b-it")
    assert not provider(never, images=False).supports_images("gemma-4-31b-it")
    assert not provider(never, "ollama").supports_images("anything")
    assert isinstance(provider(never), LLMProvider)


async def test_text_comes_without_the_thinking() -> None:
    body = sse(
        chunk(
            "<thought>The user asks", thought=True, usage={"prompt_tokens": 30, "total_tokens": 40}
        ),
        chunk(" about the weather.", thought=True),
        chunk("</thought>"),
        chunk("Сейчас в Праге"),
        ": keep-alive comment",
        chunk(" +4°C.", finish="stop"),
        chunk(usage={"prompt_tokens": 30, "completion_tokens": 18, "total_tokens": 305}),
        "[DONE]",
    )

    events = await collect(provider(answering(body)))

    assert events == [
        TextDelta(text="Сейчас в Праге"),
        TextDelta(text=" +4°C."),
        UsageEvent(input_tokens=30, output_tokens=275),
        StopEvent(reason="end"),
    ]


async def test_marked_thinking_is_dropped_even_without_tags() -> None:
    body = sse(chunk("planning", thought=True), chunk("Answer", finish="stop"))

    events = await collect(provider(answering(body)))

    assert events == [TextDelta(text="Answer"), StopEvent(reason="end")]


async def test_a_tool_call_ends_the_step_even_when_google_says_stop() -> None:
    call = {
        "id": "call_14755307676125612924",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Prague"}'},
        "extra_content": {"google": {"thought_signature": "c2lnbmF0dXJl"}},
    }
    body = sse(chunk("<thought>use the tool</thought>"), chunk(tool_calls=[call], finish="stop"))

    events = await collect(provider(answering(body)), tools=[WEATHER])

    expected = ToolCall(
        id="call_14755307676125612924",
        name="get_weather",
        input={"city": "Prague"},
        extra={"google": {"thought_signature": "c2lnbmF0dXJl"}},
    )
    assert events == [ToolCallEvent(call=expected), StopEvent(reason="tool_use")]


SIGNED_STEP = [
    *MESSAGES,
    Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="get_weather",
                input={"city": "Prague"},
                extra={"google": {"thought_signature": "c2lnbmF0dXJl"}},
            )
        ],
    ),
    Message(role="tool", tool_call_id="call_1", content=[TextPart(text="+3, rain")]),
]


async def _sent_tool_call(preset: str) -> dict[str, Any]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(chunk("Rain.", finish="stop"), "[DONE]"))

    p = provider(handle, preset)
    async for _ in p.stream(
        SIGNED_STEP, [WEATHER], "gemini-3.5-flash-lite", max_tokens=1000, cancel=CancelToken()
    ):
        pass
    (request,) = seen
    (assistant,) = [m for m in json.loads(request.content)["messages"] if m.get("tool_calls")]
    return assistant["tool_calls"][0]


async def test_google_gets_its_thought_signatures_back() -> None:
    call = await _sent_tool_call("google")

    assert call["extra_content"] == {"google": {"thought_signature": "c2lnbmF0dXJl"}}


async def test_other_providers_never_get_the_extra_of_a_call() -> None:
    call = await _sent_tool_call("mistral")

    assert "extra_content" not in call


async def test_fragments_of_calls_across_chunks() -> None:
    body = sse(
        chunk(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "get_weather"}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"city":'}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": '"Kyiv"}'}}]),
        chunk(finish="tool_calls"),
        "[DONE]",
    )

    events = await collect(provider(answering(body, 200), "mistral"), tools=[WEATHER])

    assert events[0] == ToolCallEvent(
        call=ToolCall(id="c1", name="get_weather", input={"city": "Kyiv"})
    )


async def test_an_answer_cut_by_max_tokens() -> None:
    events = await collect(provider(answering(sse(chunk("Long", finish="length")))))

    assert events[-1] == StopEvent(reason="max_tokens")


async def test_a_malformed_tool_call_is_tried_again_after_its_tokens_are_counted() -> None:
    body = sse(
        chunk(
            finish="function_call_filter: MALFORMED_FUNCTION_CALL",
            usage={"prompt_tokens": 30, "completion_tokens": 18, "total_tokens": 305},
        )
    )
    events: list[StreamEvent] = []

    with pytest.raises(ProviderError) as excinfo:
        stream = provider(answering(body)).stream(MESSAGES, [], "m", cancel=CancelToken())
        async for event in stream:
            events.append(event)

    assert excinfo.value.retryable
    assert excinfo.value.retry_after == 0
    assert events == [UsageEvent(input_tokens=30, output_tokens=275)]


async def test_a_content_filter_is_final() -> None:
    error = await failure(provider(answering(sse(chunk(finish="content_filter")))))

    assert not error.retryable and "content filter" in error.message


GOOGLE_500 = (
    b'[{"error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}}]'
)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (403, False), (404, False), (429, True), (500, True), (503, True)],
)
async def test_http_errors(status: int, retryable: bool) -> None:
    error = await failure(provider(answering(GOOGLE_500, status)))

    assert error.retryable is retryable
    assert error.message == f"google answered {status}: Internal error encountered."


async def test_odd_error_bodies() -> None:
    plain = await failure(provider(answering(b"Bad Gateway\n", 502)))
    mistral = await failure(provider(answering(b'{"detail": [{"msg": "field required"}]}', 422)))

    assert plain.message == "google answered 502: Bad Gateway"
    assert mistral.message == 'google answered 422: {"detail": [{"msg": "field required"}]}'


async def test_an_error_that_quotes_the_key_does_not_show_it() -> None:
    body = json.dumps({"error": {"message": f"API key {KEY} not valid"}}).encode()

    error = await failure(provider(answering(body, 400)))

    assert KEY not in error.message
    assert error.message == "google answered 400: API key *** not valid"


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (httpx.ConnectError("no route"), "could not reach google: ConnectError"),
        (httpx.ReadTimeout("slow"), "google did not answer in time"),
    ],
)
async def test_network_failures_are_tried_again(exception: Exception, message: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exception

    error = await failure(provider(handle))

    assert error.retryable and error.message == message


@pytest.mark.parametrize(
    "body",
    [
        sse(chunk("partial"), {"error": {"message": "overloaded"}}),
        sse("{broken json"),
        sse(chunk(tool_calls=[{"id": "c1", "function": {"name": "a", "arguments": "{"}}])),
    ],
)
async def test_broken_streams_are_tried_again(body: bytes) -> None:
    assert (await failure(provider(answering(body)))).retryable


async def test_cancel_while_the_model_is_still_thinking() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError

    cancel = CancelToken()
    asyncio.get_running_loop().call_later(0.05, cancel.cancel)

    events = await asyncio.wait_for(collect(provider(handle), cancel=cancel), timeout=2)

    assert events == []


class Hanging(httpx.AsyncByteStream):
    """A stream that sends its first chunk and then goes silent."""

    def __init__(self, first: bytes) -> None:
        self.first = first

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        await asyncio.Event().wait()


async def test_cancel_in_the_middle_of_an_answer() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Hanging(sse(chunk("Hello"))))

    cancel = CancelToken()
    events: list[StreamEvent] = []

    async def read() -> None:
        async for event in provider(handle).stream(MESSAGES, [], "m", cancel=cancel):
            events.append(event)
            cancel.cancel()

    await asyncio.wait_for(read(), timeout=2)

    assert events == [TextDelta(text="Hello")]


def test_each_preset_keeps_its_own_pace() -> None:
    paces = {name: preset.min_interval_s for name, preset in PRESETS.items()}

    assert paces == {
        "google": 2.0,
        "openrouter": 3.0,
        "mistral": 1.0,
        "ollama": 0.0,
    }


async def test_requests_start_an_interval_apart() -> None:
    started: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        started.append(asyncio.get_running_loop().time())
        return httpx.Response(200, content=sse(chunk("hi", finish="stop")))

    p = provider(handle, min_interval_s=0.1)
    await asyncio.gather(collect(p), collect(p), collect(p))

    assert len(started) == 3
    assert started[1] - started[0] >= 0.09 and started[2] - started[1] >= 0.09


async def test_waiting_for_a_slot_ends_on_cancel() -> None:
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=sse(chunk("hi", finish="stop")))

    p = provider(handle, min_interval_s=60)
    await collect(p)
    cancel = CancelToken()
    asyncio.get_running_loop().call_later(0.05, cancel.cancel)

    events = await asyncio.wait_for(collect(p, cancel=cancel), timeout=2)

    assert events == [] and len(sent) == 1


def rate_limited(body: bytes, headers: dict[str, str] | None = None) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=body, headers=headers)

    return handle


GOOGLE_429 = json.dumps(
    [
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota. Please retry in 17.4s.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
                ],
            }
        }
    ]
).encode()


async def test_google_says_how_long_to_wait_inside_the_error() -> None:
    error = await failure(provider(rate_limited(GOOGLE_429)))

    assert error.retryable and error.retry_after == 17.0
    assert error.message == (
        "google answered 429: You exceeded your current quota. Please retry in 17.4s."
    )


async def test_the_retry_after_header_in_seconds_or_as_a_date(frozen_clock: FrozenClock) -> None:
    in_seconds = await failure(provider(rate_limited(b"{}", {"Retry-After": "7"})))
    later = format_datetime(frozen_clock.current + timedelta(seconds=30), usegmt=True)
    as_date = await failure(provider(rate_limited(b"{}", {"Retry-After": later})))
    nonsense = await failure(provider(rate_limited(b"{}", {"Retry-After": "soon"})))
    silent = await failure(provider(rate_limited(b"{}")))

    assert (in_seconds.retry_after, as_date.retry_after) == (7.0, 30.0)
    assert (nonsense.retry_after, silent.retry_after) == (None, None)
