"""Titles for new sessions: a short name made after the first turn."""

import asyncio
import logging

from zero.core.cancellation import CancelToken
from zero.core.events import EventBus
from zero.core.model_call import UsageRecorder, metered
from zero.protocol.messages import SessionInfo, SessionUpdated, TextPart
from zero.providers.base import LLMProvider, Message, ProviderError, TextDelta
from zero.providers.fake import FakeProvider
from zero.store.sessions import SessionsRepo
from zero.store.usage import ModelCall

TITLE_WORDS = 6
TITLE_MAX_CHARS = 80
TITLE_INPUT_CHARS = 2000
TITLE_MAX_TOKENS = 32
TITLE_TIMEOUT_S = 15.0

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
) -> str:
    """A short title for a conversation that starts with ``text``.

    FakeProvider cannot write titles, so its sessions get the first words of
    the message. The same words are used when a real model fails, takes too
    long or answers with nothing. An empty message gives an empty title.
    """
    fallback = first_words(text)
    if isinstance(provider, FakeProvider) or not fallback:
        return fallback
    try:
        async with asyncio.timeout(TITLE_TIMEOUT_S):
            call = ModelCall(provider.name, model, "title", session_id)
            raw = await _ask_model(provider, model, text, call, usage)
    except (ProviderError, TimeoutError) as exc:
        logger.warning("No title from the model (%s); using the first words", type(exc).__name__)
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
    await events.publish(
        SessionUpdated(
            session=SessionInfo(id=titled.id, title=titled.title, updated_at=titled.updated_at)
        )
    )
