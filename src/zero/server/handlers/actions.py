"""Handler of ``action.result``: a client's answer to an action Zero asked it to perform."""

from zero.protocol.messages import ActionResult
from zero.server.connections import Connection
from zero.server.state import AppState


async def handle_action_result(
    connection: Connection, message: ActionResult, state: AppState
) -> None:
    """Hand the answer to the waiting tool call.

    Only the connection the action was sent to can answer it. A late answer
    (after a timeout or a cancel), a repeated one or one for an unknown call
    is ignored.
    """
    state.actions.resolve(connection.id, message)
