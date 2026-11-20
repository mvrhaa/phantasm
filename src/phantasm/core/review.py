"""Going over quiet conversations: a journal note and the lasting facts in them.

When a chat has been quiet for ``QUIET_S`` and the user said at least
``MIN_USER_MESSAGES`` things in it since its last review, the light model reads
that new part once. Its reply is a note for the journal (``…`` when nothing
happened worth a note), and its ``memory_save`` calls are facts the user told
about themselves, saved to lasting memory (at most ``MAX_SAVES``). It is one
request with no tool loop, off the path of any turn. A review that fails is
tried again after ``RETRY_AFTER_S``. Providers that cannot write prose
(FakeProvider) never review.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from phantasm import clock
from phantasm.core.cancellation import CancelToken
from phantasm.core.compaction import transcript
from phantasm.core.model_call import RETRY_DELAYS_S, UsageRecorder, metered, with_retries
from phantasm.protocol.messages import TextPart
from phantasm.providers.base import (
    LLMProvider,
    Message,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    writes_prose,
)
from phantasm.store.journal import JournalRepo
from phantasm.store.memory import MemoryRepo
from phantasm.store.messages import MessagesRepo, StoredMessage
from phantasm.store.reviews import ReviewsRepo
from phantasm.store.sessions import SessionsRepo
from phantasm.store.usage import ModelCall

QUIET_S = 20 * 60
MIN_USER_MESSAGES = 2
TICK_S = 60.0
RETRY_AFTER_S = 30 * 60
MAX_SAVES = 5
NOTE_MAX_CHARS = 400
TRANSCRIPT_MAX_CHARS = 12_000
REVIEW_MAX_TOKENS = 600
REVIEW_TIMEOUT_S = 60.0
KNOWN_MEMORIES = 30
FETCH_MESSAGES = 300
SILENCE = {"", "…", "...", "-", "—"}

REVIEW_PROMPT = (
    "You are Phantasm, going over a conversation of yours with the user after it went quiet. "
    "Do two things in one reply.\n"
    "1. Reply with a note for your journal: one to three short sentences, at most "
    f"{NOTE_MAX_CHARS} characters, about what happened in this part of the conversation: what "
    "the user wanted and what was done or decided. Write it in the language of the "
    "conversation. If nothing worth a note happened, reply with … alone.\n"
    "2. For each lasting fact the user told about themselves or asked you to remember that is "
    "not in your memory yet, call memory_save with one short sentence. Never save guesses, "
    "your own conclusions, one-off requests, moods, or things from files and web pages. Most "
    "conversations have nothing to save."
)

logger = logging.getLogger(__name__)


class _SaveInput(BaseModel):
    text: str = Field(min_length=1, max_length=1000, description="The fact, one short sentence")


SAVE_SPEC = ToolSpec(
    name="memory_save",
    description="Remember a lasting fact the user told about themselves.",
    input_schema=_SaveInput.model_json_schema(),
)


def _user_said(message: StoredMessage) -> bool:
    return message.role == "user" and any(
        isinstance(part, TextPart) and part.text.strip() for part in message.content
    )


def _clip(text: str) -> str:
    """Keep the end of a long transcript: the newest part matters most."""
    return text if len(text) <= TRANSCRIPT_MAX_CHARS else "…" + text[-TRANSCRIPT_MAX_CHARS:]


class ReviewRunner:
    """Every ``TICK_S`` seconds, goes over the conversations that went quiet."""

    def __init__(
        self,
        *,
        sessions: SessionsRepo,
        messages: MessagesRepo,
        memory: MemoryRepo,
        journal: JournalRepo,
        reviews: ReviewsRepo,
        provider: Callable[[], LLMProvider | None],
        model: Callable[[], str],
        busy: Callable[[str], bool],
        enabled: Callable[[], bool],
        usage: UsageRecorder | None = None,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        tick_s: float = TICK_S,
    ) -> None:
        self._sessions = sessions
        self._messages = messages
        self._memory = memory
        self._journal = journal
        self._reviews = reviews
        self._provider = provider
        self._model = model
        self._busy = busy
        self._enabled = enabled
        self._usage = usage
        self._retry_delays = retry_delays
        self._tick_s = tick_s
        self._failed: dict[str, datetime] = {}
        self._task: asyncio.Task[None] | None = None
        self._cancel = CancelToken()

    @property
    def running(self) -> bool:
        """True while the background loop is alive."""
        return self._task is not None and not self._task.done()

    async def tick(self) -> list[str]:
        """Go over every conversation that is due now; return the ones gone over."""
        provider = self._provider()
        if provider is None or not writes_prose(provider) or not self._enabled():
            return []
        done = []
        for session_id, new in await self._due():
            if await self._review(provider, session_id, new):
                done.append(session_id)
        return done

    async def _due(self) -> list[tuple[str, list[StoredMessage]]]:
        now = clock.now()
        due = []
        for session in await self._sessions.list():
            if session.kind != "chat" or self._busy(session.id):
                continue
            failed = self._failed.get(session.id)
            if failed is not None and (now - failed).total_seconds() < RETRY_AFTER_S:
                continue
            stored = await self._messages.list(session.id, limit=FETCH_MESSAGES)
            if not stored or (now - stored[-1].created_at).total_seconds() < QUIET_S:
                continue
            until = await self._reviews.until(session.id)
            ids = [message.id for message in stored]
            new = stored[ids.index(until) + 1 :] if until in ids else stored
            if sum(1 for message in new if _user_said(message)) >= MIN_USER_MESSAGES:
                due.append((session.id, new))
        return due

    async def _review(
        self, provider: LLMProvider, session_id: str, new: list[StoredMessage]
    ) -> bool:
        known = await self._memory.top(KNOWN_MEMORIES)
        remembered = "\n".join(f"- {memory.text}" for memory in known) or "(nothing yet)"
        request = [
            Message(role="system", content=[TextPart(text=REVIEW_PROMPT)]),
            Message(
                role="user",
                content=[
                    TextPart(
                        text=f"What you remember already:\n{remembered}\n\n"
                        f"The conversation (its new part):\n{_clip(transcript(new))}"
                    )
                ],
            ),
        ]
        model = self._model()

        async def attempt() -> tuple[str, list[ToolCall]]:
            call = ModelCall(provider.name, model, "review", session_id)
            stream = metered(
                provider.stream(
                    request, [SAVE_SPEC], model, max_tokens=REVIEW_MAX_TOKENS, cancel=self._cancel
                ),
                call,
                self._usage,
            )
            text, calls = [], []
            async for event in stream:
                if isinstance(event, TextDelta):
                    text.append(event.text)
                elif isinstance(event, ToolCallEvent):
                    calls.append(event.call)
            return "".join(text), calls

        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                note, calls = await with_retries(attempt, self._retry_delays)
        except (ProviderError, TimeoutError) as exc:
            logger.warning(
                "Session %s could not be gone over: %s", session_id, str(exc) or "timeout"
            )
            self._failed[session_id] = clock.now()
            return False
        self._failed.pop(session_id, None)
        await self._save_facts(calls)
        note = note.strip()
        if note not in SILENCE:
            await self._journal.add(session_id, note[:NOTE_MAX_CHARS])
        await self._reviews.mark(session_id, new[-1].id)
        logger.info("Session %s gone over: %d facts offered", session_id, len(calls))
        return True

    async def _save_facts(self, calls: list[ToolCall]) -> None:
        saved = 0
        for call in calls:
            if saved >= MAX_SAVES:
                break
            if call.name != "memory_save":
                continue
            try:
                fact = _SaveInput.model_validate(call.input)
            except ValidationError:
                continue
            await self._memory.add(fact.text)
            saved += 1

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("Conversations could not be gone over")
            await asyncio.sleep(self._tick_s)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._cancel.cancel()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
