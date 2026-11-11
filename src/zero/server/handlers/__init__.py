"""Handlers of client messages and the dispatcher that routes to them."""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import TypeAdapter, ValidationError

from zero.protocol.errors import ErrorCode
from zero.protocol.messages import (
    ApprovalRespond,
    ClientMessage,
    Envelope,
    Error,
    Hello,
    HistoryGet,
    MessageSend,
    Ping,
    Pong,
    SessionCreate,
    SessionDelete,
    SessionList,
    SessionRename,
    TurnCancel,
)
from zero.server.connections import Connection
from zero.server.handlers.approvals import handle_approval_respond
from zero.server.handlers.hello import handle_hello
from zero.server.handlers.messages import handle_message_send, handle_turn_cancel
from zero.server.handlers.sessions import (
    handle_history_get,
    handle_session_create,
    handle_session_delete,
    handle_session_list,
    handle_session_rename,
)
from zero.server.state import AppState

logger = logging.getLogger(__name__)

Handler = Callable[[Connection, Any, AppState], Awaitable[None]]

_client_message: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


async def handle_ping(connection: Connection, message: Ping, state: AppState) -> None:
    """Answer ``ping`` with ``pong``."""
    await connection.send(Pong(reply_to=message.id))


HANDLERS: dict[type[Envelope], Handler] = {
    Hello: handle_hello,
    Ping: handle_ping,
    MessageSend: handle_message_send,
    TurnCancel: handle_turn_cancel,
    ApprovalRespond: handle_approval_respond,
    SessionList: handle_session_list,
    SessionCreate: handle_session_create,
    SessionRename: handle_session_rename,
    SessionDelete: handle_session_delete,
    HistoryGet: handle_history_get,
}


def _message_id(raw: str) -> str | None:
    """Best-effort ``id`` of a message that failed validation, for ``reply_to``."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("id"), str):
        return str(data["id"])
    return None


def _describe(exc: ValidationError) -> tuple[ErrorCode, str]:
    """Turn a validation failure into an error code and a text without input values."""
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    if errors and errors[0]["type"] == "union_tag_invalid":
        return ErrorCode.UNKNOWN_TYPE, "unknown message type"
    parts = []
    for error in errors:
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return ErrorCode.INVALID_MESSAGE, "; ".join(parts)


async def dispatch(connection: Connection, raw: str, state: AppState) -> None:
    """Parse one incoming text frame and run its handler.

    Nothing here closes the connection: a bad message or a failing handler is
    answered with an ``error`` event.
    """
    try:
        message = _client_message.validate_json(raw)
    except ValidationError as exc:
        code, text = _describe(exc)
        await connection.send(Error(code=code, message=text, reply_to=_message_id(raw)))
        return

    if connection.hello is None and not isinstance(message, Hello | Ping):
        await connection.send(
            Error(
                code=ErrorCode.INVALID_MESSAGE,
                message="send hello before anything else",
                reply_to=message.id,
            )
        )
        return

    handler = HANDLERS.get(type(message))
    if handler is None:
        await connection.send(
            Error(
                code=ErrorCode.INTERNAL,
                message=f"{message.type} is not supported yet",
                reply_to=message.id,
            )
        )
        return

    try:
        await handler(connection, message, state)
    except Exception:
        logger.exception("Handler for %s failed", message.type)
        await connection.send(
            Error(code=ErrorCode.INTERNAL, message="internal error", reply_to=message.id)
        )
