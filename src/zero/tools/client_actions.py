"""Client actions: what a client declared in ``hello``, offered to the model as tools.

The model calls them like any other tool. The schema and the description come
from the client; its policy can only be made stricter by the config. Running
one means asking the client that declared it, through an ``ActionCaller``.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from zero.protocol.messages import ActionDecl
from zero.providers.base import ToolSpec
from zero.tools.base import Tool, ToolContext, ToolResult, error_result

COMPUTER_PREFIX = "computer_"  # actions that drive the screen; turns using them may take more steps


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
