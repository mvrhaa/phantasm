"""Built-in tools that come with Zero."""

from zero.store.journal import JournalRepo
from zero.store.memory import MemoryRepo
from zero.store.schedules import SchedulesRepo
from zero.store.session_memory import SessionMemoryRepo
from zero.store.todo import TodoRepo
from zero.tools.base import ToolRegistry
from zero.tools.builtin.agents import AgentControl, agent_tools
from zero.tools.builtin.edit import EditFile, WriteFile
from zero.tools.builtin.files import ListDir, ReadFile, SendFile
from zero.tools.builtin.jobs import JobManager, job_tools
from zero.tools.builtin.journal import journal_tools
from zero.tools.builtin.memory import memory_tools
from zero.tools.builtin.schedule import schedule_tools
from zero.tools.builtin.search import SearchFiles
from zero.tools.builtin.sessions import SessionStores, session_tools
from zero.tools.builtin.shell import RunCommand
from zero.tools.builtin.todo import todo_tools
from zero.tools.builtin.web import WebFetch


def builtin_registry(
    memory: MemoryRepo | None = None,
    todo: TodoRepo | None = None,
    schedules: SchedulesRepo | None = None,
    agents: AgentControl | None = None,
    jobs: JobManager | None = None,
    session_memory: SessionMemoryRepo | None = None,
    journal: JournalRepo | None = None,
    sessions: SessionStores | None = None,
) -> ToolRegistry:
    """A registry with every built-in tool; the ones with state need their stores."""
    registry = ToolRegistry()
    tools = (
        ReadFile(),
        ListDir(),
        SearchFiles(),
        WriteFile(),
        EditFile(),
        RunCommand(jobs),
        WebFetch(),
        SendFile(),
    )
    extra = [
        *(memory_tools(memory, session_memory) if memory is not None else []),
        *(todo_tools(todo) if todo is not None else []),
        *(schedule_tools(schedules) if schedules is not None else []),
        *(agent_tools(agents) if agents is not None else []),
        *(job_tools(jobs) if jobs is not None else []),
        *(journal_tools(journal) if journal is not None else []),
        *(session_tools(sessions) if sessions is not None else []),
    ]
    for tool in [*tools, *extra]:
        registry.register(tool)
    return registry
