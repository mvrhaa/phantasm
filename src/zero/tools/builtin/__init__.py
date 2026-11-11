"""Built-in tools that come with Zero."""

from zero.store.memory import MemoryRepo
from zero.tools.base import ToolRegistry
from zero.tools.builtin.edit import EditFile, WriteFile
from zero.tools.builtin.files import ListDir, ReadFile, SendFile
from zero.tools.builtin.memory import memory_tools
from zero.tools.builtin.search import SearchFiles
from zero.tools.builtin.shell import RunCommand
from zero.tools.builtin.web import WebFetch


def builtin_registry(memory: MemoryRepo | None = None) -> ToolRegistry:
    """A registry holding every built-in tool; the memory tools only if ``memory`` is given."""
    registry = ToolRegistry()
    tools = (
        ReadFile(),
        ListDir(),
        SearchFiles(),
        WriteFile(),
        EditFile(),
        RunCommand(),
        WebFetch(),
        SendFile(),
    )
    for tool in [*tools, *(memory_tools(memory) if memory is not None else [])]:
        registry.register(tool)
    return registry
