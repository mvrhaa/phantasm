"""Handler of ``message.send``: starts a turn."""

from zero.core.events import ReplyTo
from zero.core.turn import Turn
from zero.protocol.errors import ErrorCode
from zero.protocol.messages import Error, MessageSend, SessionCreated
from zero.server.connections import Connection
from zero.server.handlers.hello import session_info
from zero.server.state import AppState


async def handle_message_send(
    connection: Connection, message: MessageSend, state: AppState
) -> None:
    """Start a turn for a user message, creating a session if none was named.

    The turn runs in the background; its events go to every client. Only the
    sender's ``turn.started`` carries ``reply_to``.
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
        )
    )
