"""Plugins: tools the model writes for itself in Python, in ``plugins/<name>/`` of Phantasm's data.

A plugin is a folder with ``plugin.json`` (``name``, ``version``, ``description``)
and ``tools.py``, whose functions marked with ``phantasm.plugin_api.tool`` become
tools. Its code runs inside the brain, so the model installs (or reloads) a
plugin only with the user's approval (``plugin_install``); installed plugins
load again when the brain starts. Their tools are loaded on demand and ask the
user before every call unless the user allowed them.
"""

import asyncio
import importlib.util
import inspect
import json
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from phantasm.paths import home_dir
from phantasm.plugin_api import MARK, ToolMark
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

logger = logging.getLogger(__name__)

NAME = re.compile(r"[a-z][a-z0-9_]{0,40}")
INSTALLED = "installed.json"


class PluginTool(Tool[BaseModel]):
    """One function of a plugin."""

    default_policy = "ask"
    core = False

    def __init__(self, plugin: str, mark: ToolMark, func: Callable[..., Any]) -> None:
        self.plugin = plugin
        self.name = mark.name
        self.description = mark.description
        self._func = func
        fields: dict[str, Any] = {}
        for param in inspect.signature(func).parameters.values():
            kind = Any if param.annotation is inspect.Parameter.empty else param.annotation
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (kind, default)
        self.input_model = create_model(f"{mark.name}_input", **fields)

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        arguments = params.model_dump()
        try:
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**arguments)
            else:
                result = await asyncio.to_thread(self._func, **arguments)
        except Exception as exc:  # the plugin's own failure, for the model to read
            return error_result(f"{self.plugin}.{self.name} failed: {type(exc).__name__}: {exc}")
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        return text_result(result)


class PluginHost:
    """The plugins folder, the plugins installed and their tools."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or home_dir() / "plugins"
        self.problems: dict[str, str] = {}
        self._tools: dict[str, list[PluginTool]] = {}

    def tools(self) -> list[PluginTool]:
        return [tool for tools in self._tools.values() for tool in tools]

    def installed(self) -> list[str]:
        try:
            names = json.loads((self.folder / INSTALLED).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [name for name in names if isinstance(name, str)] if isinstance(names, list) else []

    def _save(self, names: list[str]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / INSTALLED).write_text(json.dumps(sorted(set(names))), encoding="utf-8")

    def load_installed(self) -> None:
        """Load every installed plugin; one that fails is noted, the others still load."""
        for name in self.installed():
            try:
                self.load(name)
            except ValueError as exc:
                logger.warning("Plugin %s did not load: %s", name, exc)

    def describe(self) -> list[dict[str, Any]]:
        """Every plugin folder: its manifest, whether it is installed, its tools or problem."""
        if not self.folder.is_dir():
            return []
        installed = set(self.installed())
        found = []
        for folder in sorted(p for p in self.folder.iterdir() if p.is_dir()):
            manifest = _manifest(folder)
            found.append(
                {
                    "name": folder.name,
                    "version": manifest.get("version", ""),
                    "description": manifest.get("description", ""),
                    "installed": folder.name in installed,
                    "tools": [tool.name for tool in self._tools.get(folder.name, [])],
                    "problem": self.problems.get(folder.name, ""),
                }
            )
        return found

    def install(self, name: str) -> list[PluginTool]:
        """Load (or load again) a plugin and remember it; ``ValueError`` says what is wrong."""
        tools = self.load(name)
        self._save([*self.installed(), name])
        return tools

    def remove(self, name: str) -> bool:
        """Unload a plugin; its folder stays."""
        known = name in self.installed() or name in self._tools
        self._tools.pop(name, None)
        self.problems.pop(name, None)
        sys.modules.pop(f"phantasm_plugins.{name}", None)
        self._save([n for n in self.installed() if n != name])
        return known

    def load(self, name: str) -> list[PluginTool]:
        try:
            tools = self._import(name)
        except ValueError as exc:
            self._tools.pop(name, None)
            self.problems[name] = str(exc)
            raise
        taken = {tool.name for other, ts in self._tools.items() if other != name for tool in ts}
        clash = [tool.name for tool in tools if tool.name in taken]
        if clash:
            raise ValueError(f"tool names already used by another plugin: {', '.join(clash)}")
        self._tools[name] = tools
        self.problems.pop(name, None)
        return tools

    def _import(self, name: str) -> list[PluginTool]:
        if not NAME.fullmatch(name):
            raise ValueError("a plugin name is lowercase letters, digits and _")
        folder = self.folder / name
        manifest = _manifest(folder)
        if manifest.get("name") != name:
            raise ValueError(f'{folder}/plugin.json needs "name": "{name}"')
        source = folder / "tools.py"
        spec = importlib.util.spec_from_file_location(f"phantasm_plugins.{name}", source)
        if spec is None or spec.loader is None or not source.is_file():
            raise ValueError(f"{source} is missing")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # the plugin's code failed to load
            raise ValueError(f"tools.py failed: {type(exc).__name__}: {exc}") from None
        sys.modules[spec.name] = module
        tools = []
        for _, func in inspect.getmembers(module, callable):
            mark = getattr(func, MARK, None)
            if isinstance(mark, ToolMark):
                try:
                    tools.append(PluginTool(name, mark, func))
                except (TypeError, ValidationError, ValueError) as exc:
                    raise ValueError(f"tool {mark.name}: {exc}") from None
        if not tools:
            raise ValueError("tools.py has no function marked with @tool")
        return sorted(tools, key=lambda tool: tool.name)


def _manifest(folder: Path) -> dict[str, Any]:
    try:
        data = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
