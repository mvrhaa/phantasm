"""Built-in memory tools: ``memory_save``, ``memory_search`` and ``memory_delete``."""

from typing import Any

from pydantic import BaseModel, Field

from zero.store.memory import Memory, MemoryRepo
from zero.tools.base import Preview, Tool, ToolContext, ToolResult, error_result, text_result

MAX_MEMORY_CHARS = 1000
MAX_TAGS = 10
LABEL_CHARS = 60


def _line(memory: Memory) -> str:
    tags = f" (tags: {', '.join(memory.tags)})" if memory.tags else ""
    return f"- [{memory.id}] {memory.text}{tags}"


def _short(text: str) -> str:
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 1] + "…"


class MemorySaveInput(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_MEMORY_CHARS,
        description="The fact, as a short sentence about the user",
    )
    tags: list[str] = Field(
        default_factory=list, max_length=MAX_TAGS, description="A few keywords to find it by"
    )


class MemorySave(Tool[MemorySaveInput]):
    """Keeps a fact about the user for later conversations."""

    name = "memory_save"
    description = (
        "Remember a fact for later conversations. Save only what the user said about "
        "themselves or asked you to remember, as one short sentence."
    )
    input_model = MemorySaveInput
    default_policy = "allow"

    def __init__(self, memory: MemoryRepo) -> None:
        self._memory = memory

    def label(self, params: MemorySaveInput) -> str:
        return _short(params.text)

    async def run(self, params: MemorySaveInput, ctx: ToolContext) -> ToolResult:
        try:
            saved = await self._memory.add(params.text, params.tags)
        except ValueError as exc:
            return error_result(f"Nothing was saved: {exc}.")
        return text_result(f"Remembered: {saved.text} (id {saved.id}).", summary="saved")


class MemorySearchInput(BaseModel):
    query: str = Field(min_length=1, description="Words to look for")
    limit: int = Field(default=10, ge=1, le=50, description="Most memories to return")


class MemorySearch(Tool[MemorySearchInput]):
    """Finds remembered facts by their words."""

    name = "memory_search"
    description = (
        "Search what you remember about the user. Memories that are not in the system "
        "prompt can still be found here."
    )
    input_model = MemorySearchInput
    default_policy = "allow"

    def __init__(self, memory: MemoryRepo) -> None:
        self._memory = memory

    def label(self, params: MemorySearchInput) -> str:
        return _short(params.query)

    async def run(self, params: MemorySearchInput, ctx: ToolContext) -> ToolResult:
        found = await self._memory.search(params.query, params.limit)
        if not found:
            return text_result("No memories match.", summary="0 memories")
        await self._memory.bump_uses([memory.id for memory in found])
        count = len(found)
        return text_result(
            "\n".join(_line(memory) for memory in found),
            summary=f"{count} {'memory' if count == 1 else 'memories'}",
        )


class MemoryDeleteInput(BaseModel):
    id: str = Field(min_length=1, description="Id of the memory, as memory_search shows it")


class MemoryDelete(Tool[MemoryDeleteInput]):
    """Forgets a remembered fact."""

    name = "memory_delete"
    description = "Forget a memory by its id. The user usually approves this first."
    input_model = MemoryDeleteInput
    default_policy = "ask"

    def __init__(self, memory: MemoryRepo) -> None:
        self._memory = memory

    def label(self, params: MemoryDeleteInput) -> str:
        return f"forget {params.id}"

    async def preview(self, params: MemoryDeleteInput, ctx: ToolContext) -> Preview:
        memory = await self._memory.get(params.id)
        if memory is None:
            return Preview(refusal=f"There is no memory with id {params.id}.")
        return Preview(label=f"forget: {_short(memory.text)}")

    async def run(self, params: MemoryDeleteInput, ctx: ToolContext) -> ToolResult:
        memory = await self._memory.get(params.id)
        if memory is None or not await self._memory.delete(params.id):
            return error_result(f"There is no memory with id {params.id}.")
        return text_result(f"Forgot: {memory.text}", summary="forgotten")


def memory_tools(memory: MemoryRepo) -> list[Tool[Any]]:
    """The three memory tools, working on ``memory``."""
    return [MemorySave(memory), MemorySearch(memory), MemoryDelete(memory)]
