"""Compaction: when a session's history no longer fits, its start becomes a summary.

The summary is a running one. Each time more messages drop out of the window,
they are folded into the summary made before, and the session remembers the
last message it covers. The model writes it, except FakeProvider, which
cannot: then (and whenever the model fails) a plain digest of the dropped
messages is used.
"""

import asyncio
import logging
from collections.abc import Sequence

from zero.core.cancellation import CancelToken
from zero.core.events import EventBus
from zero.core.model_call import UsageRecorder, metered
from zero.protocol.messages import SessionCompacted, TextPart
from zero.providers.base import LLMProvider, Message, ProviderError, TextDelta
from zero.providers.fake import FakeProvider
from zero.store.messages import StoredMessage
from zero.store.summaries import SummariesRepo
from zero.store.usage import ModelCall

SUMMARY_MAX_CHARS = 4000
DIGEST_LINE_CHARS = 200
TRANSCRIPT_LINE_CHARS = 2000
SUMMARY_MAX_TOKENS = 800
SUMMARY_TIMEOUT_S = 60.0

SUMMARY_PROMPT = (
    "You keep the memory of a long conversation between a user and an assistant. "
    "Write a summary of the part given below, merged with the earlier summary if there is "
    "one. Keep facts about the user, decisions, results of tools and open tasks; leave out "
    "small talk. Write in the language of the conversation, at most 300 words, and reply "
    "with the summary only."
)

logger = logging.getLogger(__name__)


def _line(message: StoredMessage, limit: int) -> str | None:
    text = " ".join(
        " ".join(part.text.split()) for part in message.content if isinstance(part, TextPart)
    )
    if message.role == "assistant" and message.tool_calls:
        names = ", ".join(str(call.get("name", "?")) for call in message.tool_calls)
        text = f"{text} [used {names}]".strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    who = {"user": "User", "assistant": "Assistant", "tool": "Tool"}[message.role]
    return f"{who}: {text}"


def transcript(messages: Sequence[StoredMessage], limit: int = TRANSCRIPT_LINE_CHARS) -> str:
    """The messages as plain lines: who said what, with tool use in brackets."""
    return "\n".join(line for m in messages if (line := _line(m, limit)) is not None)


def _cap(text: str) -> str:
    """Keep the end of a summary that grew too long: newer things matter more."""
    text = text.strip()
    return text if len(text) <= SUMMARY_MAX_CHARS else "…" + text[-(SUMMARY_MAX_CHARS - 1) :]


def digest(previous: str | None, messages: Sequence[StoredMessage]) -> str:
    """A summary made without a model: the earlier summary and short lines of the rest."""
    parts = [previous.strip()] if previous else []
    lines = transcript(messages, DIGEST_LINE_CHARS)
    if lines:
        parts.append(lines)
    return _cap("\n".join(parts))


async def _ask_model(
    provider: LLMProvider,
    model: str,
    previous: str | None,
    messages: Sequence[StoredMessage],
    call: ModelCall,
    usage: UsageRecorder | None,
) -> str:
    earlier = f"Earlier summary:\n{previous}\n\n" if previous else ""
    request = [
        Message(role="system", content=[TextPart(text=SUMMARY_PROMPT)]),
        Message(
            role="user",
            content=[TextPart(text=f"{earlier}Conversation to add:\n{transcript(messages)}")],
        ),
    ]
    parts = []
    stream = metered(
        provider.stream(request, [], model, max_tokens=SUMMARY_MAX_TOKENS, cancel=CancelToken()),
        call,
        usage,
    )
    async for event in stream:
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts)


async def summarize(
    provider: LLMProvider,
    model: str,
    previous: str | None,
    messages: Sequence[StoredMessage],
    *,
    session_id: str | None = None,
    usage: UsageRecorder | None = None,
) -> str:
    """Fold ``messages`` into ``previous``; falls back to a digest if the model cannot help.

    The model call goes into ``usage`` as a ``summary``.
    """
    if isinstance(provider, FakeProvider):
        return digest(previous, messages)
    try:
        async with asyncio.timeout(SUMMARY_TIMEOUT_S):
            call = ModelCall(provider.name, model, "summary", session_id)
            text = await _ask_model(provider, model, previous, messages, call, usage)
    except (ProviderError, TimeoutError) as exc:
        logger.warning("No summary from the model (%s); using a digest", str(exc) or "timeout")
        return digest(previous, messages)
    return _cap(text) or digest(previous, messages)


async def compact(
    session_id: str,
    stored: Sequence[StoredMessage],
    keep_from: int,
    *,
    summaries: SummariesRepo,
    provider: LLMProvider,
    model: str,
    events: EventBus,
    usage: UsageRecorder | None = None,
) -> str:
    """The summary of everything before ``stored[keep_from]``, brought up to date.

    Only messages the summary does not cover yet are summarized. When the
    summary changes, every client gets ``session.compacted``.
    """
    dropped = stored[:keep_from]
    existing = await summaries.get(session_id)
    ids = [message.id for message in stored]
    if existing is not None and existing.until_message_id in ids:
        covered = ids.index(existing.until_message_id) + 1
        if covered >= keep_from:
            return existing.text  # nothing new dropped out
        new = dropped[covered:]
    else:
        new = list(dropped)
    previous = existing.text if existing is not None else None
    text = await summarize(provider, model, previous, new, session_id=session_id, usage=usage)
    until = dropped[-1].id
    await summaries.save(session_id, text, until)
    await events.publish(
        SessionCompacted(session_id=session_id, summary=text, until_message_id=until)
    )
    logger.info("Session %s compacted: %d more messages summarized", session_id, len(new))
    return text
