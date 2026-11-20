"""The approval manager: asks every client to allow a tool call and waits for the answer."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from phantasm.core.cancellation import CancelToken
from phantasm.core.events import EventBus
from phantasm.protocol.messages import AgentRef, ApprovalRequest, ApprovalResolved

Decision = Literal["allow", "deny", "always"]

WITHDRAWN_BY = "phantasm"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovalAsk:
    """What the user is asked to allow."""

    session_id: str
    turn_id: str
    tool: str
    label: str
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    diff: str | None = None
    agent: AgentRef | None = None
    always_scope: str | None = None


@dataclass(frozen=True)
class Answer:
    """The user's answer and which client gave it."""

    decision: Decision
    by_client_id: str
    note: str | None = None


class Approver(Protocol):
    """Something that can ask the user; tools get it through their context."""

    async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Answer: ...


@dataclass
class _Pending:
    ask: ApprovalAsk
    future: asyncio.Future[Answer]
    answered: bool = False


def _request(approval_id: str, ask: ApprovalAsk) -> ApprovalRequest:
    return ApprovalRequest(
        session_id=ask.session_id,
        turn_id=ask.turn_id,
        approval_id=approval_id,
        tool=ask.tool,
        label=ask.label,
        input=ask.input,
        reason=ask.reason,
        diff=ask.diff,
        agent=ask.agent,
        always_scope=ask.always_scope,
    )


class ApprovalManager:
    """Keeps the approvals that wait for an answer.

    ``request`` sends ``approval.request`` to every client and waits. The
    first ``respond`` for an approval wins: it is announced to every client
    with ``approval.resolved`` (so the others can drop their card) and ends
    the wait; later answers are ignored. Cancelling the turn ends the wait too.
    """

    def __init__(self, events: EventBus) -> None:
        self._events = events
        self._pending: dict[str, _Pending] = {}

    @property
    def waiting(self) -> list[ApprovalAsk]:
        """Approvals that have not been answered yet."""
        return [pending.ask for pending in self._pending.values() if not pending.answered]

    @property
    def open_requests(self) -> list[ApprovalRequest]:
        """The ``approval.request`` of every approval that still waits, for a late client."""
        return [
            _request(approval_id, pending.ask)
            for approval_id, pending in self._pending.items()
            if not pending.answered
        ]

    async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Answer:
        """Ask every client and return the first answer.

        Raises ``TurnCancelled`` if ``cancel`` is set before an answer arrives,
        or at the same moment as one: a cancelled turn runs nothing more.
        """
        cancel.raise_if_cancelled()
        approval_id = str(uuid.uuid4())
        future: asyncio.Future[Answer] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = _Pending(ask, future)
        cancelled = asyncio.ensure_future(cancel.wait())
        try:
            await self._events.publish(_request(approval_id, ask))
            await asyncio.wait({future, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if cancel.cancelled:
                await self._withdraw(approval_id)
            cancel.raise_if_cancelled()
            return future.result()
        finally:
            cancelled.cancel()
            self._pending.pop(approval_id, None)

    async def _withdraw(self, approval_id: str) -> None:
        """The turn ended before an answer: clients drop the card as denied."""
        pending = self._pending[approval_id]
        if pending.answered:
            return
        pending.answered = True
        await self._events.publish(
            ApprovalResolved(
                session_id=pending.ask.session_id,
                turn_id=pending.ask.turn_id,
                approval_id=approval_id,
                decision="deny",
                by_client_id=WITHDRAWN_BY,
                agent=pending.ask.agent,
            )
        )

    async def respond(
        self, approval_id: str, decision: Decision, *, by_client_id: str, note: str | None = None
    ) -> bool:
        """Answer an approval. False if it is unknown or was answered already."""
        pending = self._pending.get(approval_id)
        if pending is None or pending.answered:
            logger.debug("Ignored an answer to approval %s", approval_id)
            return False
        pending.answered = True
        await self._events.publish(
            ApprovalResolved(
                session_id=pending.ask.session_id,
                turn_id=pending.ask.turn_id,
                approval_id=approval_id,
                decision=decision,
                by_client_id=by_client_id,
                agent=pending.ask.agent,
            )
        )
        if not pending.future.done():
            pending.future.set_result(Answer(decision, by_client_id, note))
        return True
