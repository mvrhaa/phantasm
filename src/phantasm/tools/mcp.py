"""MCP servers as Phantasm's tools: each tool of a server becomes ``mcp__<server>__<tool>``.

Servers are listed in ``mcp.json`` of Phantasm's data folder (stdio servers):

    {"servers": {"notes": {"command": "uv", "args": ["run", "notes-server"],
                           "env": {"NOTES_DIR": "~/notes", "TOKEN": "${NOTES_TOKEN}"}}}}

``${NAME}`` is filled from the environment and Phantasm's ``.env``. Servers start in
the background when the brain starts. Their tools are loaded on demand (they are
listed under ``# More tools``) and each call asks the user unless the user
allowed that tool. A server that stopped starts again on its next call.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from phantasm.core.cancellation import CancelToken
from phantasm.mcp_client import McpClient, McpError
from phantasm.paths import home_dir
from phantasm.protocol.messages import ImagePart, TextPart
from phantasm.providers.base import ToolSpec
from phantasm.secrets import env_file, read_env_file
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result

logger = logging.getLogger(__name__)

CONFIG_FILE = "mcp.json"
PREFIX = "mcp__"
CALL_TIMEOUT_S = 600.0
NAME_CHARS = 64
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def tool_name(server: str, tool: str) -> str:
    """``mcp__server__tool`` with only the characters every provider takes."""
    return f"{PREFIX}{_UNSAFE.sub('_', server)}__{_UNSAFE.sub('_', tool)}"[:NAME_CHARS]


class AnyInput(BaseModel):
    """Whatever the server's schema asks for; the server checks it."""

    model_config = ConfigDict(extra="allow")


class McpTool(Tool[AnyInput]):
    """One tool of an MCP server."""

    input_model = AnyInput
    default_policy = "ask"
    core = False

    def __init__(self, host: "McpHost", server: str, tool: dict[str, Any]) -> None:
        self._host = host
        self._server = server
        self._tool = str(tool["name"])
        self.name = tool_name(server, self._tool)
        self.description = (tool.get("description") or "").strip() or (
            f"The tool {self._tool} of the MCP server {server}."
        )
        schema = tool.get("inputSchema")
        self._schema = schema if isinstance(schema, dict) else {"type": "object", "properties": {}}

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self._schema)

    def label(self, params: AnyInput) -> str:
        return f"{self._server}: {self._tool}"

    async def run(self, params: AnyInput, ctx: ToolContext) -> ToolResult:
        try:
            result = await self._host.call(
                self._server, self._tool, params.model_dump(), ctx.cancel
            )
        except McpError as exc:
            return error_result(str(exc))
        return to_result(result)


def to_result(result: dict[str, Any]) -> ToolResult:
    """An MCP tool result as Phantasm's: text, images, resources as text."""
    parts: list[TextPart | ImagePart] = []
    for item in result.get("content") or []:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            parts.append(TextPart(text=str(item.get("text", ""))))
        elif kind == "image" and item.get("data"):
            parts.append(ImagePart(mime=item.get("mimeType") or "image/png", data_b64=item["data"]))
        elif kind == "resource":
            resource = item.get("resource") or {}
            text = resource.get("text") or f"[resource {resource.get('uri', '')}]"
            parts.append(TextPart(text=str(text)))
        elif kind == "resource_link":
            parts.append(TextPart(text=f"[resource {item.get('uri', '')}]"))
        elif item:
            parts.append(TextPart(text=json.dumps(item, ensure_ascii=False)[:4000]))
    if not parts and result.get("structuredContent") is not None:
        parts.append(TextPart(text=json.dumps(result["structuredContent"], ensure_ascii=False)))
    return ToolResult(
        content=list(parts) or [TextPart(text="(no output)")],
        is_error=bool(result.get("isError")),
    )


def _fill(value: str, known: dict[str, str]) -> str:
    return _VARIABLE.sub(lambda match: known.get(match.group(1), ""), value)


class McpHost:
    """The servers of ``mcp.json``, their clients and their tools."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or home_dir() / CONFIG_FILE
        self.problems: dict[str, str] = {}
        self._servers: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, McpClient] = {}
        self._tools: dict[str, list[McpTool]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def tools(self) -> list[McpTool]:
        return [tool for tools in self._tools.values() for tool in tools]

    def read_config(self) -> dict[str, dict[str, Any]]:
        """The servers of the file, with ``${NAME}`` filled in; a bad entry is noted and
        left out."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.problems["mcp.json"] = f"cannot be read ({type(exc).__name__})"
            return {}
        known = {**read_env_file(env_file()), **os.environ}
        servers: dict[str, dict[str, Any]] = {}
        raw = data.get("servers") if isinstance(data, dict) else None
        for name, entry in (raw or {}).items():
            command = entry.get("command") if isinstance(entry, dict) else None
            args, env = (entry or {}).get("args", []), (entry or {}).get("env", {})
            if (
                not isinstance(command, str)
                or not isinstance(args, list)
                or not isinstance(env, dict)
            ):
                self.problems[name] = "needs a command, a list of args and an object of env"
                continue
            servers[name] = {
                "command": os.path.expanduser(_fill(command, known)),
                "args": [os.path.expanduser(_fill(str(arg), known)) for arg in args],
                "env": {str(k): os.path.expanduser(_fill(str(v), known)) for k, v in env.items()},
            }
        return servers

    async def start(self) -> None:
        """Start every server and learn its tools; a server that fails is noted and skipped."""
        self._servers = self.read_config()
        await asyncio.gather(*(self._connect(name) for name in self._servers))

    async def _connect(self, name: str) -> McpClient | None:
        server = self._servers[name]
        client = McpClient(name, server["command"], server["args"], server["env"])
        try:
            await client.start()
            tools = await client.list_tools()
        except McpError as exc:
            await client.close()
            self.problems[name] = str(exc)
            logger.warning("MCP server %s did not start: %s", name, exc)
            return None
        self.problems.pop(name, None)
        self._clients[name] = client
        self._tools[name] = [McpTool(self, name, tool) for tool in tools if tool.get("name")]
        logger.info("MCP server %s: %d tools", name, len(self._tools[name]))
        return client

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any], cancel: CancelToken
    ) -> dict[str, Any]:
        """Call a tool, starting the server again if it stopped; the turn's cancel stops waiting."""
        lock = self._locks.setdefault(server, asyncio.Lock())
        async with lock:
            client = self._clients.get(server)
            if (client is None or not client.alive) and server in self._servers:
                if client is not None:
                    await client.close()
                client = await self._connect(server)
            if client is None:
                raise McpError(f"{server}: {self.problems.get(server, 'not running')}")
            if client.tools_changed:
                with contextlib.suppress(McpError):
                    listed = await client.list_tools()
                    self._tools[server] = [
                        McpTool(self, server, t) for t in listed if t.get("name")
                    ]
        return await _unless_cancelled(client.call(tool, arguments, CALL_TIMEOUT_S), cancel)

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()))
        self._clients.clear()


async def _unless_cancelled(work: Awaitable[dict[str, Any]], cancel: CancelToken) -> dict[str, Any]:
    task = asyncio.ensure_future(work)
    waiter = asyncio.ensure_future(cancel.wait())
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    if task.done():
        return task.result()
    task.cancel()
    raise McpError("the call was stopped")
