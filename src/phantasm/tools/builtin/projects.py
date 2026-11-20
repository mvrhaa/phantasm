"""Project memory tools: ``project_read``, ``project_write`` and ``project_note``.

They keep what Phantasm knows about each project in ``projects/<name>.md`` of its data
folder (``phantasm/projects.py``). The files are Phantasm's own memory, so the tools need no
approval.
"""

import asyncio

from pydantic import BaseModel, Field

from phantasm import clock
from phantasm.projects import NAME_CHARS, ProjectBook, valid_name
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

MAX_MEMORY_CHARS = 20_000
MAX_NOTE_CHARS = 1000
BAD_NAME = "A project name is lowercase letters, digits and - _ . (for example phantasm-app)."


class ProjectReadInput(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_CHARS, description="The project's name")


class ProjectWriteInput(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=NAME_CHARS,
        description="Short name: lowercase letters, digits and - _ . (the folder's name)",
    )
    path: str = Field(default="", description="The project's folder, such as ~/phantasm-app")
    summary: str = Field(min_length=1, max_length=300, description="What it is, in one sentence")
    memory: str = Field(
        min_length=1,
        max_length=MAX_MEMORY_CHARS,
        description=(
            "The whole memory in Markdown: structure, how to run and test, conventions, "
            "decisions and agreements. It replaces what was kept, notes included."
        ),
    )


class ProjectNoteInput(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_CHARS, description="The project's name")
    note: str = Field(
        min_length=1,
        max_length=MAX_NOTE_CHARS,
        description="One decision or agreement, as a short sentence",
    )


class ProjectRead(Tool[ProjectReadInput]):
    """Gives back what is kept about a project."""

    name = "project_read"
    description = (
        "Read what you keep about one of the user's projects: its folder, structure, how to "
        "run and test it, conventions, decisions and agreements."
    )
    input_model = ProjectReadInput
    default_policy = "allow"

    def __init__(self, book: ProjectBook) -> None:
        self._book = book

    def label(self, params: ProjectReadInput) -> str:
        return params.name

    async def run(self, params: ProjectReadInput, ctx: ToolContext) -> ToolResult:
        project = await asyncio.to_thread(self._book.get, params.name)
        if project is None:
            known = ", ".join(p.name for p in await asyncio.to_thread(self._book.all)) or "none"
            return error_result(f"There is no project {params.name}. Known projects: {known}.")
        text = f"# {project.name} ({project.path})\n{project.summary}\n\n{project.body}"
        return text_result(text, summary="read")


class ProjectWrite(Tool[ProjectWriteInput]):
    """Creates a project's memory or replaces it."""

    name = "project_write"
    description = (
        "Save what you know about one of the user's projects, so that next time you know it "
        "without being told: its folder, a one-sentence summary and the whole memory "
        "(structure, how to run and test it, conventions, decisions and agreements). It "
        "replaces what was kept, so read it first and keep what still holds."
    )
    input_model = ProjectWriteInput
    default_policy = "allow"

    def __init__(self, book: ProjectBook) -> None:
        self._book = book

    def label(self, params: ProjectWriteInput) -> str:
        return params.name

    async def run(self, params: ProjectWriteInput, ctx: ToolContext) -> ToolResult:
        if not valid_name(params.name):
            return error_result(BAD_NAME)
        project = await asyncio.to_thread(
            self._book.write,
            params.name,
            path=params.path,
            summary=params.summary,
            body=params.memory,
        )
        return text_result(f"Saved the memory of {project.name}.", summary="saved")


class ProjectNote(Tool[ProjectNoteInput]):
    """Adds one decision or agreement to a project's memory."""

    name = "project_note"
    description = (
        "Add one decision or agreement to a project's memory, with today's date, when the "
        "user and you settle something about it (a convention, a choice, what not to do)."
    )
    input_model = ProjectNoteInput
    default_policy = "allow"

    def __init__(self, book: ProjectBook) -> None:
        self._book = book

    def label(self, params: ProjectNoteInput) -> str:
        return f"{params.name}: {params.note[:50]}"

    async def run(self, params: ProjectNoteInput, ctx: ToolContext) -> ToolResult:
        project = await asyncio.to_thread(self._book.note, params.name, params.note, clock.now())
        if project is None:
            return error_result(
                f"There is no project {params.name}; create it with project_write first."
            )
        return text_result(f"Noted in {project.name}.", summary="noted")


def project_tools(book: ProjectBook) -> list[Tool]:
    return [ProjectRead(book), ProjectWrite(book), ProjectNote(book)]
