"""Handlers of session management and history requests."""

from zero.core.attachments import remove_files
from zero.core.events import ReplyTo
from zero.core.rewind import RewindError
from zero.protocol.errors import ErrorCode
from zero.protocol.messages import (
    Envelope,
    Error,
    FileUndo,
    HistoryGet,
    HistoryMessage,
    HistoryOk,
    SessionCreate,
    SessionCreated,
    SessionDelete,
    SessionDeleted,
    SessionList,
    SessionListOk,
    SessionRename,
    SessionRewind,
    SessionRewound,
    SessionUpdated,
    TodoUpdated,
)
from zero.server.connections import Connection
from zero.server.handlers.hello import session_info
from zero.server.state import AppState


async def _announce(
    connection: Connection, state: AppState, event: Envelope, reply_to: str | None
) -> None:
    """Tell every client about a change; only the requester's copy has ``reply_to``."""
    await state.events.publish(event, reply=ReplyTo(connection.id, reply_to))


async def _not_found(connection: Connection, session_id: str, reply_to: str | None) -> None:
    await connection.send(
        Error(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"no session with id {session_id}",
            reply_to=reply_to,
        )
    )


async def handle_session_list(
    connection: Connection, message: SessionList, state: AppState
) -> None:
    """Answer with all sessions, most recently updated first."""
    sessions = await state.sessions.list()
    await connection.send(
        SessionListOk(reply_to=message.id, sessions=[session_info(s) for s in sessions])
    )


async def handle_session_create(
    connection: Connection, message: SessionCreate, state: AppState
) -> None:
    """Create a session and announce it to everyone."""
    session = await state.sessions.create(message.title)
    await _announce(connection, state, SessionCreated(session=session_info(session)), message.id)


async def handle_session_rename(
    connection: Connection, message: SessionRename, state: AppState
) -> None:
    """Rename a session and announce the change to everyone."""
    session = await state.sessions.rename(message.session_id, message.title)
    if session is None:
        await _not_found(connection, message.session_id, message.id)
        return
    await _announce(connection, state, SessionUpdated(session=session_info(session)), message.id)


async def handle_session_delete(
    connection: Connection, message: SessionDelete, state: AppState
) -> None:
    """Delete a session with its history and announce it to everyone.

    Its running turn is cancelled first and the agents it started go with it;
    the files kept for its messages are removed.
    """
    if await state.sessions.get(message.session_id) is None:
        await _not_found(connection, message.session_id, message.id)
        return
    await state.agent_runner.session_closing(message.session_id)
    await state.turns.stop_session(message.session_id)
    files = await state.messages.file_paths(message.session_id)
    if not await state.sessions.delete(message.session_id):
        await _not_found(connection, message.session_id, message.id)
        return
    await remove_files(files)
    await _announce(connection, state, SessionDeleted(session_id=message.session_id), message.id)


async def handle_history_get(connection: Connection, message: HistoryGet, state: AppState) -> None:
    """Answer with one page of a session's messages, oldest first.

    With ``include_tools`` the page also holds the tool messages and the tool
    runs of its turns, so a client can draw the tool lines again.
    """
    if await state.sessions.get(message.session_id) is None:
        await _not_found(connection, message.session_id, message.id)
        return
    stored = await state.messages.list(message.session_id, message.before, message.limit)
    messages = [
        HistoryMessage(
            id=item.id,
            turn_id=item.turn_id,
            role=item.role,
            content=item.content,
            status=item.status,
            source=item.source,
            command=item.command,
            created_at=item.created_at,
        )
        for item in stored
        if message.include_tools or item.role != "tool"
    ]
    runs = []
    if message.include_tools:
        turn_ids = {item.turn_id for item in stored if item.turn_id is not None}
        runs = await state.tool_runs.for_turns(message.session_id, turn_ids)
    await connection.send(
        HistoryOk(
            reply_to=message.id,
            session_id=message.session_id,
            messages=messages,
            tool_runs=runs,
        )
    )


async def handle_session_rewind(
    connection: Connection, message: SessionRewind, state: AppState
) -> None:
    """Go back to just before a user message (see ``Rewinder``).

    After a real rewind every client gets ``session.rewound`` (and the new todo
    list if items went); a dry run answers only the sender.
    """
    try:
        report = await state.rewinder.rewind(
            message.session_id, message.message_id, dry_run=message.dry_run
        )
    except RewindError as exc:
        await connection.send(Error(code=exc.code, message=exc.message, reply_to=message.id))
        return
    event = SessionRewound(
        session_id=report.session_id,
        message_id=report.message_id,
        text=report.text,
        dry_run=message.dry_run,
        removed_messages=report.removed_messages,
        files=[FileUndo(path=f.path, change=f.change) for f in report.files],  # type: ignore[arg-type]
        removed_todos=report.removed_todos,
        removed_session_memories=report.removed_session_memories,
        stopped_agents=report.stopped_agents,
        kept=report.kept,
    )
    if message.dry_run:
        await connection.send(event.model_copy(update={"reply_to": message.id}))
        return
    await _announce(connection, state, event, message.id)
    if report.removed_todos:
        items = [todo.item() for todo in await state.todos.list(message.session_id)]
        await state.events.publish(TodoUpdated(session_id=message.session_id, items=items))
