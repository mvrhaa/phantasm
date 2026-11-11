"""Built-in tools that come with Zero."""

from zero.tools.base import ToolRegistry
from zero.tools.builtin.files import ListDir, ReadFile
from zero.tools.builtin.search import SearchFiles


def builtin_registry() -> ToolRegistry:
    """A registry holding every built-in tool."""
    registry = ToolRegistry()
    for tool in (ReadFile(), ListDir(), SearchFiles()):
        registry.register(tool)
    return registry
