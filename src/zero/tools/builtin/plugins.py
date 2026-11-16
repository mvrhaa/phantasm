"""Tools for plugins (``zero/plugins.py``): list them, install or reload one, remove one."""

from typing import Any

from pydantic import BaseModel, Field

from zero.plugins import PluginHost
from zero.tools.base import Preview, Tool, ToolContext, ToolResult, error_result, text_result

FORMAT = (
    "A plugin is a folder {folder}/<name>/ (name: lowercase letters, digits, _) with "
    'plugin.json {{"name": "<name>", "version": "1.0", "description": "..."}} and tools.py: '
    "from zero.plugin_api import tool, then functions marked @tool with typed parameters and "
    "a docstring; each returns text and raises on failure. Write the files with write_file."
)


class NoInput(BaseModel):
    pass


class PluginNameInput(BaseModel):
    name: str = Field(min_length=1, description="The plugin's folder name")


class PluginList(Tool[NoInput]):
    name = "plugin_list"
    input_model = NoInput
    default_policy = "allow"
    core = False

    def __init__(self, host: PluginHost) -> None:
        self._host = host
        self.description = (
            "List your plugins: installed or not, their tools, what is wrong with a broken one. "
            + FORMAT.format(folder=host.folder)
        )

    async def run(self, params: NoInput, ctx: ToolContext) -> ToolResult:
        lines = []
        for info in self._host.describe():
            state = "installed" if info["installed"] else "not installed"
            tools = ", ".join(info["tools"]) or "no tools loaded"
            line = f"- {info['name']} {info['version']} ({state}): {info['description']} [{tools}]"
            if info["problem"]:
                line += f" PROBLEM: {info['problem']}"
            lines.append(line)
        return text_result("\n".join(lines) or f"No plugins in {self._host.folder}.")


class PluginInstall(Tool[PluginNameInput]):
    name = "plugin_install"
    input_model = PluginNameInput
    default_policy = "ask"
    core = False

    def __init__(self, host: PluginHost) -> None:
        self._host = host
        self.description = (
            "Install a plugin you wrote, or load it again after you changed it. Its code runs "
            "inside Zero, so the user is asked every time. Its tools then come on demand and "
            "ask the user before each call until the user allows them. "
            + FORMAT.format(folder=host.folder)
        )

    def label(self, params: PluginNameInput) -> str:
        return params.name

    async def preview(self, params: PluginNameInput, ctx: ToolContext) -> Preview:
        return Preview(approval_reason=f"the plugin {params.name} would run its code inside Zero")

    async def run(self, params: PluginNameInput, ctx: ToolContext) -> ToolResult:
        try:
            tools = self._host.install(params.name)
        except ValueError as exc:
            return error_result(f"{params.name} did not load: {exc}. Fix its files and try again.")
        names = ", ".join(tool.name for tool in tools)
        return text_result(
            f"Installed {params.name}: {names}. From the next turn they are under # More tools; "
            "load them with tools_load."
        )


class PluginRemove(Tool[PluginNameInput]):
    name = "plugin_remove"
    description = "Uninstall a plugin: its tools go away, its folder stays."
    input_model = PluginNameInput
    default_policy = "ask"
    core = False

    def __init__(self, host: PluginHost) -> None:
        self._host = host

    def label(self, params: PluginNameInput) -> str:
        return params.name

    async def run(self, params: PluginNameInput, ctx: ToolContext) -> ToolResult:
        if not self._host.remove(params.name):
            return error_result(f"No plugin {params.name!r} is installed.")
        return text_result(f"Removed {params.name}.")


def plugin_tools(host: PluginHost) -> list[Tool[Any]]:
    return [PluginList(host), PluginInstall(host), PluginRemove(host)]
