"""A provider for every API of the ``/chat/completions`` format.

Google (Gemma), OpenRouter, Mistral and a local Ollama take the same request
and stream the answer back as server-sent events; what differs between them
lives in ``presets.py``. The API key is read from .env for every request and
goes nowhere but the request header: errors name the key, never its value.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable
from typing import Any

import httpx

from zero import __version__
from zero.providers.base import (
    CancelSignal,
    Message,
    ProviderError,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)
from zero.providers.chat_format import (
    ThoughtFilter,
    ToolCallAssembler,
    wire_messages,
    wire_tools,
)
from zero.providers.presets import Preset
from zero.secrets import api_key, env_file, mask_secrets

TIMEOUT = httpx.Timeout(120.0, connect=15.0)  # read: the longest silence within an answer
USER_AGENT = f"Zero/{__version__} (local assistant)"
ERROR_TEXT_CHARS = 300
RETRYABLE_STATUSES = frozenset({408, 425, 429})  # and every 5xx
MALFORMED_CALL = "MALFORMED_FUNCTION_CALL"  # Google's finish reason for a broken tool call

_CANCELLED = object()
_END = object()


class ChatCompletionsProvider:
    """Talks to one provider of the ``/chat/completions`` format, chosen by its preset.

    ``images`` overrides whether the model takes images; ``transport`` lets
    tests answer instead of the network.
    """

    def __init__(
        self,
        preset: Preset,
        *,
        images: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.preset = preset
        self.name = preset.name
        self._images = preset.images if images is None else images
        self._transport = transport

    def supports_images(self, model: str) -> bool:
        """Tell whether the model takes images; set per preset, not per model."""
        return self._images

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None:
        """These APIs cannot count a prompt before it is sent."""
        return None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelSignal,
    ) -> AsyncIterator[StreamEvent]:
        """Send the conversation and yield the answer as it comes.

        On cancellation the stream ends quietly at once, even in the middle
        of waiting for the network.
        """
        headers = self._headers()
        body = self._body(messages, tools, model, max_tokens)
        waiter = asyncio.ensure_future(cancel.wait())
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=TIMEOUT) as client:
                request = client.build_request(
                    "POST", f"{self.preset.base_url}/chat/completions", json=body, headers=headers
                )
                response = await _unless_cancelled(client.send(request, stream=True), waiter)
                if response is _CANCELLED:
                    return
                try:
                    if response.status_code != 200:
                        raw = await _unless_cancelled(response.aread(), waiter)
                        if raw is _CANCELLED:
                            return
                        raise _status_error(self.name, response.status_code, raw)
                    async for event in self._events(response, waiter):
                        yield event
                finally:
                    await response.aclose()
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name} did not answer in time", retryable=True) from None
        except httpx.HTTPError as exc:
            message = f"could not reach {self.name}: {type(exc).__name__}"
            raise ProviderError(message, retryable=True) from None
        finally:
            waiter.cancel()

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if self.preset.key_name:
            key = api_key(self.name)
            if key is None:
                raise ProviderError(f"{self.preset.key_name} is missing in {env_file()}")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(
        self, messages: list[Message], tools: list[ToolSpec], model: str, max_tokens: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": wire_messages(messages),
            "stream": True,
            "max_tokens": max_tokens + self.preset.thinking_tokens,
        }
        if self.preset.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = wire_tools(tools)
        return body

    async def _events(
        self, response: httpx.Response, waiter: Awaitable[None]
    ) -> AsyncIterator[StreamEvent]:
        """Read the server-sent events of one answer into Zero's stream events."""
        thoughts = ThoughtFilter()
        calls = ToolCallAssembler()
        usage: dict[str, Any] | None = None
        finish: str | None = None
        lines = response.aiter_lines()
        while True:
            line = await _unless_cancelled(_next_line(lines), waiter)
            if line is _CANCELLED:
                return
            chunk = _END if line is _END else _parse(self.name, line)
            if chunk is _END:
                break
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk:
                raise ProviderError(
                    f"{self.name} failed mid-answer: {_error_text(chunk)}", retryable=True
                )
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = thoughts.feed(_delta_text(delta), thought=_is_thought(delta))
                if text:
                    yield TextDelta(text=text)
                for piece in delta.get("tool_calls") or []:
                    calls.add(piece)
                finish = choice.get("finish_reason") or finish
        tail = thoughts.flush()
        if tail:
            yield TextDelta(text=tail)
        if usage:
            yield _usage(usage)
        _check_finish(self.name, finish)
        tool_calls = calls.calls()
        for call in tool_calls:
            yield ToolCallEvent(call=call)
        # The calls decide, whatever finish_reason says: Google ends them with "stop".
        reason = "tool_use" if tool_calls else "max_tokens" if finish == "length" else "end"
        yield StopEvent(reason=reason)


async def _unless_cancelled(awaitable: Awaitable[Any], waiter: Awaitable[None]) -> Any:
    """Await ``awaitable`` unless the turn is cancelled first; then give ``_CANCELLED``."""
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    return _CANCELLED


async def _next_line(lines: AsyncIterator[str]) -> Any:
    try:
        return await anext(lines)
    except StopAsyncIteration:
        return _END


def _parse(provider: str, line: str) -> Any:
    """One line of the stream: a chunk, ``_END`` for ``[DONE]``, or None for anything else."""
    if not line.startswith("data:"):
        return None  # blank lines between events, ": keep-alive" comments
    data = line[5:].strip()
    if data == "[DONE]":
        return _END
    if not data:
        return None
    try:
        chunk = json.loads(data)
    except ValueError:
        raise ProviderError(f"{provider} sent a broken chunk", retryable=True) from None
    if isinstance(chunk, list):  # Google wraps errors in a list
        chunk = chunk[0] if chunk else None
    return chunk


def _delta_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Mistral may send content as a list of chunks
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return ""


def _is_thought(delta: dict[str, Any]) -> bool:
    extra = delta.get("extra_content") or {}
    return bool((extra.get("google") or {}).get("thought"))


def _usage(usage: dict[str, Any]) -> UsageEvent:
    """Tokens of the whole answer; the model's thinking counts as output."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)
    return UsageEvent(input_tokens=prompt, output_tokens=max(completion, total - prompt))


def _check_finish(provider: str, finish: str | None) -> None:
    """Raise if the answer ended in a failure: a broken tool call or a provider-side
    error is worth another try, a content filter is not."""
    if finish and (MALFORMED_CALL in finish or finish == "error"):
        raise ProviderError(f"{provider} could not finish the answer ({finish})", retryable=True)
    if finish == "content_filter":
        raise ProviderError(f"{provider} blocked the answer with its content filter")


def _error_text(data: Any) -> str:
    """The provider's own words about a failure, short and without secrets."""
    text = _find_message(data) or json.dumps(data, ensure_ascii=False)
    return " ".join(mask_secrets(text).split())[:ERROR_TEXT_CHARS]


def _find_message(data: Any) -> str | None:
    if isinstance(data, list):
        return _find_message(data[0]) if data else None
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            found = value if isinstance(value, str) else _find_message(value)
            if found:
                return found
    return None


def _status_error(provider: str, status: int, raw: bytes) -> ProviderError:
    try:
        data: Any = json.loads(raw)
    except ValueError:
        data = raw.decode("utf-8", "replace")
    text = _error_text(data) if not isinstance(data, str) else _error_text({"message": data})
    retryable = status in RETRYABLE_STATUSES or status >= 500
    return ProviderError(f"{provider} answered {status}: {text}", retryable=retryable)
