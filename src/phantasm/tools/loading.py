"""Tools on demand: the tools that are not core reach the model only once it loads them.

Their schemas would cost every request a few thousand tokens. Instead the
system prompt lists them in brief (``more_tools_block``), and ``tools_load``
adds the ones a task needs to the requests of this conversation from the
model's next step on.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from phantasm.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, text_result

TOOLS_LOAD = "tools_load"
SUMMARY_CHARS = 90


class ToolsLoadInput(BaseModel):
    names: list[str] = Field(min_length=1, description="Names of tools listed under # More tools.")


class ToolsLoadTool(Tool[ToolsLoadInput]):
    """Loads tools of the ``# More tools`` list into ``loaded`` (this conversation's set)."""

    name = TOOLS_LOAD
    description = (
        "Load tools listed under # More tools so you can call them. They come with their "
        "parameters from your next step and stay loaded in this conversation. Load only "
        "what the task needs."
    )
    input_model = ToolsLoadInput
    default_policy = "allow"

    def __init__(self, registry: ToolRegistry, loaded: set[str]) -> None:
        self._registry = registry
        self._loaded = loaded

    def label(self, params: ToolsLoadInput) -> str:
        return ", ".join(params.names)

    async def run(self, params: ToolsLoadInput, ctx: ToolContext) -> ToolResult:
        found, unknown = [], []
        for name in dict.fromkeys(params.names):
            tool = self._registry.get(name)
            if tool is None or tool.core:
                unknown.append(name)
            else:
                found.append(name)
        self._loaded.update(found)
        lines = []
        if found:
            lines.append(f"Loaded: {', '.join(found)}. Call them from your next step.")
        if unknown:
            lines.append(f"Not in # More tools: {', '.join(unknown)}.")
        return text_result("\n".join(lines))


def brief(tool: Tool[Any]) -> str:
    """``name(param, optional?): first sentence`` of a tool, for the prompt."""
    schema = tool.spec().input_schema
    required = set(schema.get("required") or [])
    params = ", ".join(
        name if name in required else f"{name}?" for name in schema.get("properties") or {}
    )
    summary = tool.description.strip().split(". ")[0].rstrip(".")
    if len(summary) > SUMMARY_CHARS:
        summary = summary[: SUMMARY_CHARS - 1].rstrip() + "…"
    return f"- {tool.name}({params}): {summary}"


def more_tools_block(tools: Sequence[Tool[Any]]) -> str | None:
    """The prompt block that lists the tools to load, or None when there are none."""
    if not tools:
        return None
    lines = [
        "# More tools",
        f"These tools are not loaded. To use one, load it with {TOOLS_LOAD} first, "
        "then call it on your next step.",
        *(brief(tool) for tool in tools),
    ]
    return "\n".join(lines)
