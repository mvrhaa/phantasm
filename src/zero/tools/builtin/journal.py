"""Built-in ``journal_search``: find notes about what happened in earlier conversations."""

from typing import Any

from pydantic import BaseModel, Field

from zero.store.journal import JournalRepo, note_time
from zero.tools.base import Tool, ToolContext, ToolResult, text_result


class JournalSearchInput(BaseModel):
    query: str = Field(min_length=1, description="Words to look for")
    limit: int = Field(default=10, ge=1, le=50, description="Most notes to return")


class JournalSearch(Tool[JournalSearchInput]):
    """Finds journal notes by their words."""

    name = "journal_search"

    core = False
    description = (
        "Search the journal: short dated notes about what happened in your earlier "
        "conversations, newest first. The newest ones are already in the system prompt."
    )
    input_model = JournalSearchInput
    default_policy = "allow"

    def __init__(self, journal: JournalRepo) -> None:
        self._journal = journal

    def label(self, params: JournalSearchInput) -> str:
        return params.query

    async def run(self, params: JournalSearchInput, ctx: ToolContext) -> ToolResult:
        found = await self._journal.search(params.query, params.limit)
        if not found:
            return text_result("No journal notes match.", summary="0 notes")
        lines = [f"- [{note_time(note.created_at)}] {note.text}" for note in found]
        count = len(found)
        return text_result("\n".join(lines), summary=f"{count} {'note' if count == 1 else 'notes'}")


def journal_tools(journal: JournalRepo) -> list[Tool[Any]]:
    return [JournalSearch(journal)]
