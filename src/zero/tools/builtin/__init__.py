"""Built-in tools that come with Zero."""

from zero.tools.base import ToolRegistry
from zero.tools.builtin.edit import EditFile, WriteFile
from zero.tools.builtin.files import ListDir, ReadFile
from zero.tools.builtin.search import SearchFiles
from zero.tools.builtin.shell import RunCommand


def builtin_registry() -> ToolRegistry:
    """A registry holding every built-in tool."""
    registry = ToolRegistry()
    tools = (ReadFile(), ListDir(), SearchFiles(), WriteFile(), EditFile(), RunCommand())
    for tool in tools:
        registry.register(tool)
    return registry
