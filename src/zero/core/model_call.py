"""Calling the model: one streamed answer, retries, and a record of every call.

Every request to a model is written down (provider, model, what it was for,
tokens, whether it worked), because free plans limit requests and tokens per
day and Zero has to know how much it has used.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from zero.core.cancellation import CancelToken
from zero.core.events import EventBus
from zero.protocol.messages import TextDelta as TextDeltaEvent
from zero.protocol.messages import Usage
from zero.providers.base import (
    LLMProvider,
    Message,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    UsageEvent,
)
from zero.store.usage import ModelCall, Purpose

RETRY_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)  # free plans fail often
MAX_RETRY_AFTER_S = 60.0  # a provider that wants a longer pause gets the error instead

logger = logging.getLogger(__name__)


class UsageRecorder(Protocol):
    """Keeps the record of model calls."""

    async def record(self, call: ModelCall) -> None: ...


async def metered(
    stream: AsyncIterator[StreamEvent], call: ModelCall, recorder: UsageRecorder | None
) -> AsyncIterator[StreamEvent]:
    """Pass a provider's stream through, counting its tokens; record the call at the end.

    The call is recorded however the stream ends: finished, failed or cancelled.
    A failure to record is logged and never breaks the stream.
    """
    try:
        async for event in stream:
            if isinstance(event, UsageEvent):
                call.input_tokens += event.input_tokens
                call.output_tokens += event.output_tokens
            yield event
    except BaseException:
        call.ok = False
        raise
    finally:
        if recorder is not None:
            try:
                await recorder.record(call)
            except Exception:
                logger.exception("Could not record a call to %s", call.provider)


@dataclass
class Progress:
    """What a turn has produced so far; survives a failure in the middle of a step."""

    text: str = ""  # everything the model wrote during the turn, as streamed
    step_text: str = ""  # what it wrote in the current step and is not saved yet
    usage: Usage = field(default_factory=Usage)


class ModelCaller:
    """Gets the answers of one turn from the model.

    Text is published as it streams. A retryable failure is retried after
    each of ``retry_delays``, but only before any text went out: text sent to
    clients cannot be taken back. When the provider says how long to wait,
    that wait is used instead, up to ``MAX_RETRY_AFTER_S``; a longer one ends
    the turn with the error rather than leaving it hanging.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        purpose: Purpose,
        session_id: str,
        turn_id: str,
        events: EventBus,
        cancel: CancelToken,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        usage: UsageRecorder | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._purpose: Purpose = purpose
        self._session_id = session_id
        self._turn_id = turn_id
        self._events = events
        self._cancel = cancel
        self._retry_delays = tuple(retry_delays)
        self._usage = usage

    async def answer(
        self, prompt: list[Message], tools: list[ToolSpec], progress: Progress
    ) -> list[ToolCall]:
        """Stream one answer, retrying a retryable failure that came before any text."""
        for delay in self._retry_delays:
            try:
                return await self._stream(prompt, tools, progress)
            except ProviderError as exc:
                if not exc.retryable or progress.step_text:
                    raise
                wait = delay if exc.retry_after is None else exc.retry_after
                if wait > MAX_RETRY_AFTER_S:
                    raise
                logger.warning(
                    "Turn %s: trying again in %.1f s after: %s", self._turn_id, wait, exc.message
                )
                await self._pause(wait)
        return await self._stream(prompt, tools, progress)

    async def _stream(
        self, prompt: list[Message], tools: list[ToolSpec], progress: Progress
    ) -> list[ToolCall]:
        """Stream one answer of the model, publishing its text; return its tool calls."""
        self._cancel.raise_if_cancelled()
        calls: list[ToolCall] = []
        call = ModelCall(self._provider.name, self._model, self._purpose, self._session_id)
        stream = metered(
            self._provider.stream(prompt, tools, self._model, cancel=self._cancel),
            call,
            self._usage,
        )
        async for event in stream:
            if isinstance(event, TextDelta):
                progress.text += event.text
                progress.step_text += event.text
                await self._events.publish(
                    TextDeltaEvent(
                        session_id=self._session_id, turn_id=self._turn_id, delta=event.text
                    )
                )
            elif isinstance(event, ToolCallEvent):
                calls.append(event.call)
            elif isinstance(event, UsageEvent):
                progress.usage.input_tokens += event.input_tokens
                progress.usage.output_tokens += event.output_tokens
        # A provider may end its stream quietly on cancellation instead of raising.
        self._cancel.raise_if_cancelled()
        return calls

    async def _pause(self, seconds: float) -> None:
        """Sleep, but stop with ``TurnCancelled`` as soon as the turn is cancelled."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._cancel.wait(), timeout=seconds)
        self._cancel.raise_if_cancelled()
