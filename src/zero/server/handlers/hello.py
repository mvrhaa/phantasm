"""The ``hello`` handshake."""

from zero import __version__
from zero.protocol.errors import PROTOCOL_VERSION, ErrorCode
from zero.protocol.messages import Error, Hello, HelloOk, SessionInfo
from zero.server.connections import Connection
from zero.server.state import AppState
from zero.store.sessions import Session

CLOSE_PROTOCOL_MISMATCH = 4400


def session_info(session: Session) -> SessionInfo:
    """Convert a stored session into what clients are told about it."""
    return SessionInfo(id=session.id, title=session.title, updated_at=session.updated_at)


async def handle_hello(connection: Connection, message: Hello, state: AppState) -> None:
    """Remember what the client declared and answer with ``hello.ok``.

    A client that speaks another protocol version gets ``protocol_mismatch``
    and is disconnected. A repeated ``hello`` replaces the earlier declaration.
    """
    if message.protocol != PROTOCOL_VERSION:
        await connection.send(
            Error(
                code=ErrorCode.PROTOCOL_MISMATCH,
                message=f"this server speaks protocol {PROTOCOL_VERSION}",
                reply_to=message.id,
            )
        )
        await connection.close(CLOSE_PROTOCOL_MISMATCH, "protocol mismatch")
        return

    connection.hello = message.client
    sessions = await state.sessions.list()
    await connection.send(
        HelloOk(
            reply_to=message.id,
            zero_version=__version__,
            protocol=PROTOCOL_VERSION,
            sessions=[session_info(session) for session in sessions],
        )
    )
