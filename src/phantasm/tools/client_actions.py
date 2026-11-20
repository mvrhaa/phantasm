"""Client actions: what a client declared in ``hello``, offered to the model as tools.

The model calls them like any other tool. The schema and the description come
from the client; its policy can only be made stricter by the config. Running
one means asking the client that declared it, through an ``ActionCaller``.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from phantasm.core.cancellation import TurnCancelled
from phantasm.protocol.errors import ErrorCode
from phantasm.protocol.messages import ActionCall, ActionDecl, ActionResult, Envelope, TextPart
from phantasm.providers.base import ToolSpec
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result

COMPUTER_PREFIX = "computer_"

logger = logging.getLogger(__name__)


class ActionInput(BaseModel):
    """Any JSON object: the client's own schema describes it, and the client checks it."""

    model_config = ConfigDict(extra="allow")


class ActionCaller(Protocol):
    """Sends an action to the client that declared it and waits for the result."""

    async def call(
        self, connection_id: str, action: ActionDecl, params: dict[str, Any], ctx: ToolContext
    ) -> ToolResult: ...


class ClientActionTool(Tool[ActionInput]):
    """One action of one connected client, as a tool."""

    input_model = ActionInput
    config_tightens_only = True

    def __init__(self, action: ActionDecl, connection_id: str, caller: ActionCaller | None) -> None:
        self.action = action
        self.connection_id = connection_id
        self.name = action.name
        self.description = action.description
        self.default_policy = action.policy
        self._caller = caller

    def spec(self) -> ToolSpec:
        """The tool as the model sees it, with the schema the client declared."""
        schema = self.action.input_schema or {"type": "object", "properties": {}}
        return ToolSpec(name=self.name, description=self.description, input_schema=schema)

    async def run(self, params: ActionInput, ctx: ToolContext) -> ToolResult:
        if self._caller is None:
            return error_result(f"{self.name} cannot be called: the client is not reachable.")
        return await self._caller.call(self.connection_id, self.action, params.model_dump(), ctx)


def action_tools(
    actions: Sequence[ActionDecl], connection_id: str, caller: ActionCaller | None
) -> list[ClientActionTool]:
    """Tools for the actions a client declared; a repeated name keeps its first declaration."""
    tools: dict[str, ClientActionTool] = {}
    for action in actions:
        tools.setdefault(action.name, ClientActionTool(action, connection_id, caller))
    return list(tools.values())


class ActionTarget(Protocol):
    """A connected client that can be sent an ``action.call``."""

    async def send(self, event: Envelope) -> bool: ...


class Connections(Protocol):
    """Finds a live connection by its id."""

    def get(self, connection_id: str) -> ActionTarget | None: ...


class _ClientGone(Exception):
    """The connection closed while its action was running."""


@dataclass
class _Pending:
    connection_id: str
    future: asyncio.Future[ActionResult]


class ActionBroker:
    """Sends actions to clients and waits for their answers.

    An ``action.call`` goes only to the connection that declared the action.
    The wait ends with that connection's ``action.result`` for the same call
    id, after the action's ``timeout_s``, when the connection closes, or when
    the turn is cancelled. Every failure comes back to the model as an error.
    """

    def __init__(self, connections: Connections) -> None:
        self._connections = connections
        self._pending: dict[str, _Pending] = {}

    @property
    def waiting(self) -> int:
        """How many actions are waiting for an answer."""
        return len(self._pending)

    async def call(
        self, connection_id: str, action: ActionDecl, params: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        """Run ``action`` on the client and return what it answered."""
        gone = error_result(
            f"{action.name} could not run: the client that offers it has disconnected "
            f"({ErrorCode.CLIENT_GONE})."
        )
        target = self._connections.get(connection_id)
        if target is None:
            return gone
        call_id = str(uuid.uuid4())
        future: asyncio.Future[ActionResult] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = _Pending(connection_id, future)
        cancelled = asyncio.ensure_future(ctx.cancel.wait())
        try:
            sent = await target.send(
                ActionCall(
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    call_id=call_id,
                    action=action.name,
                    input=params,
                )
            )
            if not sent:
                return gone
            await asyncio.wait(
                {future, cancelled}, timeout=action.timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            if ctx.cancel.cancelled:
                raise TurnCancelled()
            if not future.done():
                return error_result(
                    f"{action.name} did not answer within {action.timeout_s:g} s "
                    f"({ErrorCode.ACTION_TIMEOUT})."
                )
            try:
                answer = future.result()
            except _ClientGone:
                return gone
            return _to_result(action.name, answer)
        finally:
            cancelled.cancel()
            self._pending.pop(call_id, None)
            if future.done() and not future.cancelled():
                future.exception()

    def resolve(self, connection_id: str, answer: ActionResult) -> bool:
        """Deliver an ``action.result``. False if no action of that connection waits for it."""
        pending = self._pending.get(answer.call_id)
        if pending is None or pending.connection_id != connection_id or pending.future.done():
            logger.debug("Ignored action.result for %s", answer.call_id)
            return False
        pending.future.set_result(answer)
        return True

    def connection_closed(self, connection_id: str) -> None:
        """End every action still waiting on a connection that has gone away."""
        for pending in self._pending.values():
            if pending.connection_id == connection_id and not pending.future.done():
                pending.future.set_exception(_ClientGone())


def _to_result(name: str, answer: ActionResult) -> ToolResult:
    if not answer.ok:
        reason = answer.error or "no reason given"
        return ToolResult(
            content=[TextPart(text=f"{name} failed on the client: {reason}"), *answer.content],
            is_error=True,
        )
    return ToolResult(content=answer.content or [TextPart(text=f"{name} done.")])
