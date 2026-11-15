"""Built-in tools to look at other conversations: list, read and search them.

They read only what the user and Zero said (no tool calls or results), so the
model can pick up what was talked about elsewhere without the noise.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from zero.protocol.messages import TextPart
from zero.store.journal import note_time
from zero.store.messages import MessagesRepo, StoredMessage
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

LINE_CHARS = 2000
SNIPPET_CHARS = 200
READ_FETCH = 500


@dataclass(frozen=True)
class SessionStores:
    """What the session tools read."""

    sessions: SessionsRepo
    messages: MessagesRepo
    summaries: SummariesRepo


def _flat(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _line(message: StoredMessage) -> str | None:
    """``User: …`` or ``Zero: …`` for a message with something said in it."""
    if message.role not in ("user", "assistant"):
        return None
    parts = [part.text for part in message.content if isinstance(part, TextPart)]
    if message.role == "user" and len(parts) < len(message.content):
        parts.append("[attachment]")
    text = _flat(" ".join(parts), LINE_CHARS)
    if not text:
        return None
    return f"{'User' if message.role == 'user' else 'Zero'}: {text}"


class SessionsListInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=50, description="Most conversations to list")


class SessionsList(Tool[SessionsListInput]):
    """Lists the other conversations."""

    name = "sessions_list"
    description = (
        "List your other conversations, the latest first: id, title, kind (chat, agent or "
        "scheduled) and when it last changed. Read one with sessions_read."
    )
    input_model = SessionsListInput
    default_policy = "allow"

    def __init__(self, stores: SessionStores) -> None:
        self._stores = stores

    def label(self, params: SessionsListInput) -> str:
        return "conversations"

    async def run(self, params: SessionsListInput, ctx: ToolContext) -> ToolResult:
        others = [s for s in await self._stores.sessions.list() if s.id != ctx.session_id]
        shown = others[: params.limit]
        if not shown:
            return text_result("There are no other conversations.", summary="0 conversations")
        lines = [
            f"- {s.id} · {s.title or '(untitled)'} · {s.kind} · {note_time(s.updated_at)}"
            for s in shown
        ]
        count = len(shown)
        noun = "conversation" if count == 1 else "conversations"
        return text_result("\n".join(lines), summary=f"{count} {noun}")


class SessionsReadInput(BaseModel):
    session_id: str = Field(min_length=1, description="The conversation, as sessions_list shows it")
    last: int = Field(default=20, ge=1, le=200, description="How many of the latest lines to read")


class SessionsRead(Tool[SessionsReadInput]):
    """Reads what was said in another conversation."""

    name = "sessions_read"
    description = (
        "Read another conversation: what the user and you said, without tool calls, the "
        "latest last lines. When the start is left out, its summary is shown if there is one."
    )
    input_model = SessionsReadInput
    default_policy = "allow"

    def __init__(self, stores: SessionStores) -> None:
        self._stores = stores

    def label(self, params: SessionsReadInput) -> str:
        return params.session_id

    async def run(self, params: SessionsReadInput, ctx: ToolContext) -> ToolResult:
        session = await self._stores.sessions.get(params.session_id)
        if session is None:
            return error_result(
                f"There is no conversation {params.session_id}; find one with sessions_list."
            )
        stored = await self._stores.messages.list(session.id, limit=READ_FETCH)
        lines = [line for message in stored if (line := _line(message)) is not None]
        shown = lines[-params.last :]
        hidden = len(lines) - len(shown)
        out = [session.title or "(untitled)"]
        if hidden:
            summary = await self._stores.summaries.get(session.id)
            if summary is not None:
                out.append(
                    f"[{hidden} earlier lines not shown; summary of the start: "
                    f"{_flat(summary.text, LINE_CHARS)}]"
                )
            else:
                out.append(f"[{hidden} earlier lines not shown; ask for more with a larger last]")
        out += shown or ["(nothing said yet)"]
        return text_result("\n".join(out), summary=f"{len(shown)} lines")


class SessionsSearchInput(BaseModel):
    query: str = Field(min_length=1, description="Words to look for")
    limit: int = Field(default=10, ge=1, le=50, description="Most messages to return")


class SessionsSearch(Tool[SessionsSearchInput]):
    """Finds messages in other conversations by their words."""

    name = "sessions_search"
    description = (
        "Search what the user and you said in your other conversations, newest first; "
        "each hit names its conversation for sessions_read."
    )
    input_model = SessionsSearchInput
    default_policy = "allow"

    def __init__(self, stores: SessionStores) -> None:
        self._stores = stores

    def label(self, params: SessionsSearchInput) -> str:
        return params.query

    async def run(self, params: SessionsSearchInput, ctx: ToolContext) -> ToolResult:
        found = await self._stores.messages.search_text(params.query, ctx.session_id, params.limit)
        if not found:
            return text_result("No other conversation mentions that.", summary="0 messages")
        titles: dict[str, str] = {}
        lines = []
        for message, text in found:
            if message.session_id not in titles:
                session = await self._stores.sessions.get(message.session_id)
                titles[message.session_id] = (session.title if session else None) or "(untitled)"
            who = "User" if message.role == "user" else "Zero"
            lines.append(
                f"- {message.session_id} · {titles[message.session_id]} · "
                f"{note_time(message.created_at)} · {who}: {_flat(text, SNIPPET_CHARS)}"
            )
        count = len(lines)
        return text_result(
            "\n".join(lines), summary=f"{count} {'message' if count == 1 else 'messages'}"
        )


def session_tools(stores: SessionStores) -> list[Tool[Any]]:
    return [SessionsList(stores), SessionsRead(stores), SessionsSearch(stores)]
