"""Built-in tools that come with Phantasm."""

from phantasm.browser import BrowserBridge
from phantasm.plugins import PluginHost
from phantasm.skills import SkillBook
from phantasm.store.file_changes import FileChangesRepo
from phantasm.store.journal import JournalRepo
from phantasm.store.memory import MemoryRepo
from phantasm.store.schedules import SchedulesRepo
from phantasm.store.session_memory import SessionMemoryRepo
from phantasm.store.todo import TodoRepo
from phantasm.tools.base import ToolRegistry
from phantasm.tools.builtin.agents import AgentControl, agent_tools
from phantasm.tools.builtin.browser import browser_tools
from phantasm.tools.builtin.edit import EditFile, WriteFile
from phantasm.tools.builtin.files import ListDir, ReadFile, SendFile
from phantasm.tools.builtin.github import github_tools
from phantasm.tools.builtin.jobs import JobManager, job_tools
from phantasm.tools.builtin.journal import journal_tools
from phantasm.tools.builtin.memory import memory_tools
from phantasm.tools.builtin.plugins import plugin_tools
from phantasm.tools.builtin.sandbox import RunPython
from phantasm.tools.builtin.schedule import schedule_tools
from phantasm.tools.builtin.search import SearchFiles
from phantasm.tools.builtin.sessions import SessionStores, session_tools
from phantasm.tools.builtin.shell import RunCommand
from phantasm.tools.builtin.skills import SkillUse
from phantasm.tools.builtin.todo import todo_tools
from phantasm.tools.builtin.web import WebFetch


def builtin_registry(
    memory: MemoryRepo | None = None,
    todo: TodoRepo | None = None,
    schedules: SchedulesRepo | None = None,
    agents: AgentControl | None = None,
    jobs: JobManager | None = None,
    session_memory: SessionMemoryRepo | None = None,
    journal: JournalRepo | None = None,
    sessions: SessionStores | None = None,
    file_changes: FileChangesRepo | None = None,
    skills: SkillBook | None = None,
    plugins: PluginHost | None = None,
    github: bool = False,
    browser: BrowserBridge | None = None,
) -> ToolRegistry:
    """A registry with every built-in tool; the ones with state need their stores."""
    registry = ToolRegistry()
    tools = (
        ReadFile(),
        ListDir(),
        SearchFiles(),
        WriteFile(file_changes),
        EditFile(file_changes),
        RunCommand(jobs),
        RunPython(),
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
        *([SkillUse(skills)] if skills is not None else []),
        *(plugin_tools(plugins) if plugins is not None else []),
        *(github_tools() if github else []),
        *(browser_tools(browser) if browser is not None else []),
    ]
    for tool in [*tools, *extra]:
        registry.register(tool)
    return registry
