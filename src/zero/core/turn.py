"""The agent loop: one turn from a user message to the final answer."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

from zero import clock
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.core.context import build_history, build_system_prompt
from zero.protocol.messages import (
    ClientInfo,
    Envelope,
    TextPart,
    TurnFinished,
    TurnStarted,
    Usage,
)
from zero.protocol.messages import TextDelta as TextDeltaEvent
from zero.providers.base import LLMProvider, Message, ProviderError, TextDelta, UsageEvent
from zero.store.messages import MessagesRepo, Source
from zero.store.sessions import SessionsRepo

HISTORY_FETCH_LIMIT = 500

logger = logging.getLogger(__name__)

Emit = Callable[[Envelope], Awaitable[None]]


class Turn:
    """One user message and everything the model does in response to it.

    The turn knows nothing about WebSockets: it reports progress by calling
    ``emit`` with protocol events.
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
        emit: Emit,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.cancel = CancelToken()
        self._text = text
        self._source: Source = source
        self._client = client
        self._config = config
        self._provider = provider
        self._sessions = sessions
        self._messages = messages
        self._emit = emit

    async def run(self) -> None:
        """Run the turn to the end. Never raises: failures end the turn with an error."""
        await self._messages.append(
            self.session_id,
            "user",
            [TextPart(text=self._text)],
            turn_id=self.id,
            status="ok",
            source=self._source,
        )
        await self._emit(
            TurnStarted(session_id=self.session_id, turn_id=self.id, source=self._source)
        )

        text = ""
        usage = Usage()
        error: str | None = None
        try:
            prompt = await self._build_prompt()
            stream = self._provider.stream(
                prompt, [], self._config.provider.model, cancel=self.cancel
            )
            async for event in stream:
                if isinstance(event, TextDelta):
                    text += event.text
                    await self._emit(
                        TextDeltaEvent(
                            session_id=self.session_id, turn_id=self.id, delta=event.text
                        )
                    )
                elif isinstance(event, UsageEvent):
                    usage.input_tokens += event.input_tokens
                    usage.output_tokens += event.output_tokens
        except ProviderError as exc:
            logger.warning("Turn %s failed: provider error (retryable=%s)", self.id, exc.retryable)
            error = exc.message
        except Exception:
            logger.exception("Turn %s failed", self.id)
            error = "internal error"

        status: Literal["ok", "error"] = "error" if error is not None else "ok"
        await self._messages.append(
            self.session_id,
            "assistant",
            [TextPart(text=text)] if text else [],
            turn_id=self.id,
            status=status,
            source=self._source,
        )
        await self._sessions.touch(self.session_id)
        await self._emit(
            TurnFinished(
                session_id=self.session_id,
                turn_id=self.id,
                status=status,
                text=text,
                usage=usage,
                error=error,
            )
        )

    async def _build_prompt(self) -> list[Message]:
        """System prompt followed by the history that fits the token budget."""
        system = build_system_prompt(self._client, self._source, now=clock.now())
        stored = await self._messages.list(self.session_id, limit=HISTORY_FETCH_LIMIT)
        history = build_history(stored, self._config.agent.history_token_budget)
        return [Message(role="system", content=[TextPart(text=system)]), *history]


class TurnManager:
    """Runs turns as background tasks and keeps track of the ones in progress."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[str, Turn] = {}

    @property
    def active(self) -> list[Turn]:
        """Turns that are still running."""
        return list(self._turns.values())

    def start(self, turn: Turn) -> None:
        """Start a turn in the background and forget it once it is done."""
        task = asyncio.create_task(turn.run(), name=f"turn-{turn.id}")
        self._tasks[turn.id] = task
        self._turns[turn.id] = turn
        task.add_done_callback(lambda _: self._forget(turn.id))

    def _forget(self, turn_id: str) -> None:
        self._tasks.pop(turn_id, None)
        self._turns.pop(turn_id, None)

    async def wait_idle(self) -> None:
        """Wait until every running turn has finished."""
        while self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def shutdown(self) -> None:
        """Stop all running turns; used when the application shuts down."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
