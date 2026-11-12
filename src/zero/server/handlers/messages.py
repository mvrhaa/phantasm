"""Handlers of ``message.send`` and ``turn.cancel``: starting and stopping turns."""

import asyncio
import logging
from functools import partial

from zero.core.attachments import AttachmentError, save_attachments
from zero.core.events import ReplyTo
from zero.core.titles import title_new_session
from zero.protocol.errors import ErrorCode
from zero.protocol.messages import Error, MessageSend, SessionCreated, TurnCancel
from zero.providers.registry import small_model
from zero.server.connections import Connection
from zero.server.handlers.hello import session_info
from zero.server.state import AppState
from zero.server.turns import make_turn
from zero.tools.permissions import PathGuard

logger = logging.getLogger(__name__)


async def handle_message_send(
    connection: Connection, message: MessageSend, state: AppState
) -> None:
    """Start a turn for a user message, creating a session if none was named.

    The turn runs in the background; its events go to every client. Only the
    sender's ``turn.started`` carries ``reply_to``. A session runs one turn at
    a time: a message into a busy session is answered with ``turn_busy``.
    An untitled session gets a title from this message once the turn is over.
    Attachments are saved to ``files/`` first; a bad one is answered with
    ``invalid_message`` and nothing starts.
    """
    if state.stopping:
        await connection.send(
            Error(
                code=ErrorCode.INTERNAL,
                message="Zero is shutting down",
                reply_to=message.id,
            )
        )
        return
    if state.provider is None:
        await connection.send(
            Error(
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
                message="no usable provider is configured",
                reply_to=message.id,
            )
        )
        return

    found = None
    if message.session_id is not None:
        found = await state.sessions.get(message.session_id)
        if found is None:
            await connection.send(
                Error(
                    code=ErrorCode.SESSION_NOT_FOUND,
                    message=f"no session with id {message.session_id}",
                    reply_to=message.id,
                )
            )
            return

    try:
        attachments = await asyncio.to_thread(
            save_attachments,
            message.attachments,
            PathGuard.from_config(state.config.permissions),
        )
    except AttachmentError as exc:
        await connection.send(
            Error(code=ErrorCode.INVALID_MESSAGE, message=str(exc), reply_to=message.id)
        )
        return

    if found is None:
        session = await state.sessions.create()
        await state.events.publish(SessionCreated(session=session_info(session)))
    else:
        session = found

    # No await between this check and start(), so two messages cannot both pass it.
    if state.turns.busy(session.id):
        await connection.send(
            Error(
                code=ErrorCode.TURN_BUSY,
                message="a turn is already running in this session",
                reply_to=message.id,
            )
        )
        return

    after = None
    if session.title is None:
        after = partial(
            title_new_session,
            session.id,
            message.text,
            provider=state.provider,
            model=small_model(state.config.provider),
            sessions=state.sessions,
            events=state.events,
            usage=state.usage,
        )
    state.turns.start(
        make_turn(
            state,
            state.provider,
            session_id=session.id,
            text=message.text,
            source=message.source,
            client=connection.hello,
            reply=ReplyTo(connection.id, message.id),
            attachments=attachments,
        ),
        after=after,
    )


async def handle_turn_cancel(connection: Connection, message: TurnCancel, state: AppState) -> None:
    """Ask a running turn to stop; it then finishes with status ``cancelled``.

    An unknown id is ignored: usually the turn has just finished on its own
    and the client already has its ``turn.finished``.
    """
    turn = state.turns.get(message.turn_id)
    if turn is None:
        logger.debug("turn.cancel for a turn that is not running: %s", message.turn_id)
        return
    turn.cancel.cancel()
