"""The WebSocket endpoint: authorization, the read loop and the connection lifetime."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from zero.protocol.errors import ErrorCode
from zero.protocol.messages import Error
from zero.server.auth import verify
from zero.server.connections import Connection
from zero.server.handlers import dispatch
from zero.server.state import get_state

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN_ORIGIN = 4403

_BEARER = "bearer "

logger = logging.getLogger(__name__)

router = APIRouter()


def origin_allowed(origin: str | None) -> bool:
    """Native clients send no Origin (or ``null``); anything else is a browser page."""
    return origin is None or origin == "null"


def presented_token(websocket: WebSocket) -> str | None:
    """Take the token from ``Authorization: Bearer`` or, for debugging, ``?token=``."""
    header = websocket.headers.get("authorization")
    if header is not None:
        if header[: len(_BEARER)].lower() == _BEARER:
            return header[len(_BEARER) :].strip()
        return None
    return websocket.query_params.get("token")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept a client after checking its Origin and token.

    A rejected client is accepted and closed at once, because a close code
    (4403 or 4401) can only be delivered over an open WebSocket.
    """
    state = get_state(websocket.app)
    await websocket.accept()

    if not origin_allowed(websocket.headers.get("origin")):
        logger.warning("Rejected WebSocket from a browser origin")
        await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="origin not allowed")
        return
    if not verify(presented_token(websocket), state.token):
        logger.warning("Rejected WebSocket with a missing or wrong token")
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="unauthorized")
        return

    connection = Connection(websocket)
    state.connections.add(connection)
    try:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                break
            text = frame.get("text")
            if text is None:
                await connection.send(
                    Error(code=ErrorCode.INVALID_MESSAGE, message="binary frames are not supported")
                )
                continue
            await dispatch(connection, text, state)
            if connection.closed:
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.connections.remove(connection)
