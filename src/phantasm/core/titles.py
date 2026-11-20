"""Titles for new sessions: a short name made after the first turn."""

import asyncio
import logging
from collections.abc import Sequence

from phantasm.core.cancellation import CancelToken
from phantasm.core.events import EventBus
from phantasm.core.model_call import RETRY_DELAYS_S, UsageRecorder, metered, with_retries
from phantasm.protocol.messages import SessionUpdated, TextPart
from phantasm.providers.base import LLMProvider, Message, ProviderError, TextDelta, writes_prose
from phantasm.store.sessions import SessionsRepo
from phantasm.store.usage import ModelCall

TITLE_WORDS = 6
TITLE_MAX_CHARS = 80
TITLE_INPUT_CHARS = 2000
TITLE_MAX_TOKENS = 32
TITLE_TIMEOUT_S = 60.0

TITLE_PROMPT = (
    "Name the conversation that starts with the user's message below. "
    "Reply with a title of 5 to 7 words in the language of the message and nothing else: "
    "no quotes, no punctuation at the end."
)

_QUOTES = "\"'`«»“”„"

logger = logging.getLogger(__name__)


def first_words(text: str, count: int = TITLE_WORDS) -> str:
    """The first ``count`` words of ``text``, cut to the title length limit."""
    return " ".join(text.split()[:count])[:TITLE_MAX_CHARS].rstrip()


def clean_title(raw: str) -> str:
    """The first non-empty line of a model's answer without quotes and final punctuation."""
    line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    line = line.strip(_QUOTES).rstrip(".!?:;,").strip(_QUOTES).strip()
    return line[:TITLE_MAX_CHARS].rstrip()


async def _ask_model(
    provider: LLMProvider, model: str, text: str, call: ModelCall, usage: UsageRecorder | None
) -> str:
    messages = [
        Message(role="system", content=[TextPart(text=TITLE_PROMPT)]),
        Message(role="user", content=[TextPart(text=text[:TITLE_INPUT_CHARS])]),
    ]
    parts = []
    stream = metered(
        provider.stream(messages, [], model, max_tokens=TITLE_MAX_TOKENS, cancel=CancelToken()),
        call,
        usage,
    )
    async for event in stream:
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts)


async def make_title(
    provider: LLMProvider,
    model: str,
    text: str,
    *,
    session_id: str | None = None,
    usage: UsageRecorder | None = None,
    retry_delays: Sequence[float] = RETRY_DELAYS_S,
) -> str:
    """A short title for a conversation that starts with ``text``.

    FakeProvider cannot write titles, so its sessions get the first words of
    the message. The same words are used when a real model fails (after the
    same retries as a turn), takes too long or answers with nothing. An empty
    message gives an empty title.
    """
    fallback = first_words(text)
    if not writes_prose(provider) or not fallback:
        return fallback

    async def attempt() -> str:
        call = ModelCall(provider.name, model, "title", session_id)
        return await _ask_model(provider, model, text, call, usage)

    try:
        async with asyncio.timeout(TITLE_TIMEOUT_S):
            raw = await with_retries(attempt, retry_delays)
    except (ProviderError, TimeoutError) as exc:
        logger.warning("No title from the model (%s); using the first words", str(exc) or "timeout")
        return fallback
    return clean_title(raw) or fallback


async def title_new_session(
    session_id: str,
    text: str,
    *,
    provider: LLMProvider,
    model: str,
    sessions: SessionsRepo,
    events: EventBus,
    usage: UsageRecorder | None = None,
) -> None:
    """Title an untitled session after its first turn and tell every client.

    Nothing happens if the session got a title meanwhile (the user renamed
    it) or was deleted, even while the title was being made.
    """
    session = await sessions.get(session_id)
    if session is None or session.title is not None:
        return
    title = await make_title(provider, model, text, session_id=session_id, usage=usage)
    if not title:
        return
    titled = await sessions.set_title_if_missing(session_id, title)
    if titled is None:
        return
    await events.publish(SessionUpdated(session=titled.info()))
