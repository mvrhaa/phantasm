"""Going back to a message: the user's message and everything after it are undone.

What is undone: the messages from that one on (its text goes back to the
user), files ``write_file`` and ``edit_file`` changed since (in this session
and in the sessions of agents it started since), todo items and entries of
this conversation's memory added since, and agents it started since. A file
changed again outside Phantasm since is left as it is. The summary and the review
mark follow the messages that are left. What cannot be undone (commands that
ran, schedules, lasting memory, actions on the computer) is reported as kept.
"""

import asyncio
import contextlib
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from phantasm.protocol.errors import ErrorCode
from phantasm.protocol.messages import TextPart
from phantasm.store.agents import AgentsRepo
from phantasm.store.file_changes import FileChange, FileChangesRepo, sha256
from phantasm.store.messages import MessagesRepo, StoredMessage
from phantasm.store.reviews import ReviewsRepo
from phantasm.store.session_memory import SessionMemoryRepo
from phantasm.store.sessions import SessionsRepo
from phantasm.store.summaries import SummariesRepo
from phantasm.store.todo import TodoRepo
from phantasm.store.tool_runs import ToolRunsRepo

# Calls that change nothing, or whose change going back undoes, are not reported as kept.
NO_LASTING_EFFECT = frozenset(
    {
        "read_file",
        "list_dir",
        "search_files",
        "web_fetch",
        "memory_search",
        "journal_search",
        "sessions_list",
        "sessions_read",
        "sessions_search",
        "todo_list",
        "todo_add",
        "schedule_list",
        "agent_status",
        "agent_start",
        "job_output",
        "set_expression",
        "write_file",
        "edit_file",
        "ask_parent",
    }
)


class RewindError(Exception):
    """Going back is not possible; ``code`` is the protocol error code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FileUndo:
    path: str
    change: str  # "restored", "deleted" or "changed elsewhere"


@dataclass
class RewindReport:
    """What going back did, or would do on a dry run."""

    session_id: str
    message_id: str
    text: str
    removed_messages: int = 0
    files: list[FileUndo] = field(default_factory=list)
    removed_todos: int = 0
    removed_session_memories: int = 0
    stopped_agents: list[str] = field(default_factory=list)
    kept: dict[str, int] = field(default_factory=dict)


def _text(message: StoredMessage) -> str:
    return "\n".join(part.text for part in message.content if isinstance(part, TextPart))


def _kept(removed: list[StoredMessage]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for message in removed:
        for call in message.tool_calls:
            name = str(call.get("name", ""))
            scope = (call.get("input") or {}).get("scope")
            if name in NO_LASTING_EFFECT or (name == "memory_save" and scope == "session"):
                continue
            counts[name] += 1
    return dict(counts)


def _current(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _write_back(path: Path, data: bytes) -> None:
    """Replace the file in one step, keeping its permissions; make it again if it is gone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _undo_files(changes: list[FileChange], apply: bool) -> list[FileUndo]:
    """Put every changed file back as it was before its first change since then."""
    by_path: dict[str, list[FileChange]] = {}
    for change in changes:
        by_path.setdefault(change.path, []).append(change)
    undone = []
    for raw, path_changes in by_path.items():
        path = Path(raw)
        now = _current(path)
        if now is None or sha256(now) != path_changes[-1].after_sha256:
            undone.append(FileUndo(raw, "changed elsewhere"))
            continue
        before = path_changes[0].before
        if before is None:
            if apply:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
            undone.append(FileUndo(raw, "deleted"))
        else:
            if apply:
                _write_back(path, before)
            undone.append(FileUndo(raw, "restored"))
    return undone


class Rewinder:
    """Goes back to a user message of a session."""

    def __init__(
        self,
        *,
        sessions: SessionsRepo,
        messages: MessagesRepo,
        changes: FileChangesRepo,
        todos: TodoRepo,
        session_memory: SessionMemoryRepo,
        agents: AgentsRepo,
        summaries: SummariesRepo,
        reviews: ReviewsRepo,
        tool_runs: ToolRunsRepo,
        busy: Callable[[str], bool],
        stop_agent: Callable[[str, str], Awaitable[object]],
    ) -> None:
        self._sessions = sessions
        self._messages = messages
        self._changes = changes
        self._todos = todos
        self._session_memory = session_memory
        self._agents = agents
        self._summaries = summaries
        self._reviews = reviews
        self._tool_runs = tool_runs
        self._busy = busy
        self._stop_agent = stop_agent

    async def rewind(
        self, session_id: str, message_id: str, *, dry_run: bool = False
    ) -> RewindReport:
        """Go back to just before ``message_id``; with ``dry_run``, only say what would go."""
        if await self._sessions.get(session_id) is None:
            raise RewindError(ErrorCode.SESSION_NOT_FOUND, f"no session {session_id}")
        point = await self._messages.get(message_id)
        if point is None or point.session_id != session_id or point.role != "user":
            raise RewindError(
                ErrorCode.INVALID_MESSAGE, "message_id must be a user message of this session"
            )
        if self._busy(session_id):
            raise RewindError(ErrorCode.TURN_BUSY, "a turn is running in this session")

        since = point.created_at
        stored = await self._messages.list(session_id, limit=100_000)
        ids = [message.id for message in stored]
        removed = stored[ids.index(message_id) :]
        new_agents = [
            agent for agent in await self._agents.list(session_id) if agent.created_at >= since
        ]
        sessions = [session_id, *(a.session_id for a in new_agents if a.session_id)]
        changes = await self._changes.since(sessions, since)
        report = RewindReport(
            session_id=session_id,
            message_id=message_id,
            text=_text(point),
            removed_messages=len(removed),
            files=await asyncio.to_thread(_undo_files, changes, not dry_run),
            stopped_agents=[agent.name for agent in new_agents],
            kept=_kept(removed),
        )
        if dry_run:
            report.removed_todos = sum(
                1 for todo in await self._todos.list(session_id) if todo.created_at >= since
            )
            report.removed_session_memories = sum(
                1
                for entry in await self._session_memory.list(session_id)
                if entry.created_at >= since
            )
            return report

        await self._changes.forget([change.id for change in changes])
        for agent in new_agents:
            await self._stop_agent(agent.id, session_id)
        await self._messages.delete_from(session_id, message_id)
        await self._tool_runs.delete_turns(
            session_id, {m.turn_id for m in removed if m.turn_id is not None}
        )
        report.removed_todos = await self._todos.delete_since(session_id, since)
        report.removed_session_memories = await self._session_memory.delete_since(session_id, since)
        await self._follow_messages(session_id, stored[: ids.index(message_id)], set(ids))
        return report

    async def _follow_messages(
        self, session_id: str, left: list[StoredMessage], before: set[str]
    ) -> None:
        """A summary or review mark that covered removed messages no longer holds."""
        left_ids = {message.id for message in left}
        summary = await self._summaries.get(session_id)
        if summary is not None and summary.until_message_id in before - left_ids:
            await self._summaries.delete(session_id)
        until = await self._reviews.until(session_id)
        if until is not None and until in before - left_ids:
            await self._reviews.reset(session_id, left[-1].id if left else None)
