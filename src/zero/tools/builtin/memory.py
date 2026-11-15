"""Built-in memory tools: ``memory_save``, ``memory_search`` and ``memory_delete``.

``memory_save`` keeps a fact in the user's lasting memory or, with
``scope: "session"``, in the memory of this conversation only.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from zero.store.memory import Memory, MemoryRepo
from zero.store.session_memory import SessionMemoryRepo
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
    scope: Literal["everywhere", "session"] = Field(
        default="everywhere",
        description=(
            "everywhere: a lasting fact for every conversation; session: an instruction or "
            "fact for this conversation only"
        ),
    )


class MemorySave(Tool[MemorySaveInput]):
    """Keeps a fact about the user for later conversations."""

    name = "memory_save"
    description = (
        "Remember a fact for later conversations. Save only what the user said about "
        "themselves or asked you to remember, as one short sentence. With scope session "
        "it is kept for this conversation only, such as an instruction for this chat."
    )
    input_model = MemorySaveInput
    default_policy = "allow"

    def __init__(self, memory: MemoryRepo, session_memory: SessionMemoryRepo | None = None) -> None:
        self._memory = memory
        self._session_memory = session_memory

    def label(self, params: MemorySaveInput) -> str:
        return _short(params.text)

    async def run(self, params: MemorySaveInput, ctx: ToolContext) -> ToolResult:
        if params.scope == "session":
            if self._session_memory is None:
                return error_result("This conversation has no memory of its own here.")
            try:
                entry = await self._session_memory.add(ctx.session_id, params.text)
            except ValueError as exc:
                return error_result(f"Nothing was saved: {exc}.")
            return text_result(
                f"Remembered for this conversation: {entry.text} (id {entry.id}).",
                summary="saved here",
            )
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
    """Forgets a remembered fact, lasting or of this conversation."""

    name = "memory_delete"
    description = (
        "Forget a memory by its id, a lasting one or one of this conversation. "
        "The user usually approves this first."
    )
    input_model = MemoryDeleteInput
    default_policy = "ask"

    def __init__(self, memory: MemoryRepo, session_memory: SessionMemoryRepo | None = None) -> None:
        self._memory = memory
        self._session_memory = session_memory

    def label(self, params: MemoryDeleteInput) -> str:
        return f"forget {params.id}"

    async def _text(self, memory_id: str) -> str | None:
        memory = await self._memory.get(memory_id)
        if memory is not None:
            return memory.text
        if self._session_memory is not None:
            entry = await self._session_memory.get(memory_id)
            if entry is not None:
                return entry.text
        return None

    async def preview(self, params: MemoryDeleteInput, ctx: ToolContext) -> Preview:
        text = await self._text(params.id)
        if text is None:
            return Preview(refusal=f"There is no memory with id {params.id}.")
        return Preview(label=f"forget: {_short(text)}")

    async def run(self, params: MemoryDeleteInput, ctx: ToolContext) -> ToolResult:
        text = await self._text(params.id)
        deleted = text is not None and (
            await self._memory.delete(params.id)
            or (self._session_memory is not None and await self._session_memory.delete(params.id))
        )
        if not deleted:
            return error_result(f"There is no memory with id {params.id}.")
        return text_result(f"Forgot: {text}", summary="forgotten")


def memory_tools(
    memory: MemoryRepo, session_memory: SessionMemoryRepo | None = None
) -> list[Tool[Any]]:
    """The three memory tools, working on ``memory`` and the sessions' own memory."""
    return [
        MemorySave(memory, session_memory),
        MemorySearch(memory),
        MemoryDelete(memory, session_memory),
    ]
