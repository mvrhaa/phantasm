"""Handler of ``approval.respond``: the user's answer to an approval request."""

from phantasm.protocol.messages import ApprovalRespond
from phantasm.server.connections import Connection
from phantasm.server.state import AppState


async def handle_approval_respond(
    connection: Connection, message: ApprovalRespond, state: AppState
) -> None:
    """Pass the answer on; every client then gets ``approval.resolved``.

    Only the first answer counts. A late or repeated one, or one for an
    approval that no longer waits (its turn was cancelled), is ignored.
    """
    await state.approvals.respond(
        message.approval_id,
        message.decision,
        by_client_id=connection.client_id or "",
        note=message.note,
    )
