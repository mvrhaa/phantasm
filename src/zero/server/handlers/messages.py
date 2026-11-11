"""Handlers of ``message.send`` and ``turn.cancel``: starting and stopping turns."""

import logging
from functools import partial

from zero.core.events import ReplyTo
from zero.core.titles import title_new_session
from zero.core.turn import Turn
from zero.protocol.errors import ErrorCode
from zero.protocol.messages import Error, MessageSend, SessionCreated, TurnCancel
from zero.server.connections import Connection
from zero.server.handlers.hello import session_info
from zero.server.state import AppState

logger = logging.getLogger(__name__)


async def handle_message_send(
    connection: Connection, message: MessageSend, state: AppState
) -> None:
    """Start a turn for a user message, creating a session if none was named.

    The turn runs in the background; its events go to every client. Only the
    sender's ``turn.started`` carries ``reply_to``. A session runs one turn at
    a time: a message into a busy session is answered with ``turn_busy``.
    An untitled session gets a title from this message once the turn is over.
    """
    if state.provider is None:
        await connection.send(
            Error(
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
                message="no usable provider is configured",
                reply_to=message.id,
            )
        )
        return

    if message.session_id is None:
        session = await state.sessions.create()
        await state.events.publish(SessionCreated(session=session_info(session)))
    else:
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
            model=state.config.provider.model,
            sessions=state.sessions,
            events=state.events,
        )
    state.turns.start(
        Turn(
            session_id=session.id,
            text=message.text,
            source=message.source,
            client=connection.hello,
            config=state.config,
            provider=state.provider,
            sessions=state.sessions,
            messages=state.messages,
            events=state.events,
            reply=ReplyTo(connection.id, message.id),
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
