"""Built-in todo tools: a numbered todo list per session that the model keeps up to date."""

from typing import Any

from pydantic import BaseModel, Field

from zero.protocol.messages import TodoStatus, TodoUpdated
from zero.store.todo import Todo, TodoRepo
from zero.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

MAX_TODO_CHARS = 300


def _render(todos: list[Todo]) -> ToolResult:
    if not todos:
        return text_result("The todo list is empty.", summary="0 items")
    done = sum(1 for todo in todos if todo.status == "done")
    lines = [f"{todo.number}. [{todo.status}] {todo.text}" for todo in todos]
    return text_result("\n".join(lines), summary=f"{len(todos)} items, {done} done")


class _TodoTool[InputT: BaseModel](Tool[InputT]):
    default_policy = "allow"

    def __init__(self, todo: TodoRepo) -> None:
        self._todo = todo

    async def _changed(self, ctx: ToolContext) -> ToolResult:
        """Tell the clients about the new list and give it to the model."""
        todos = await self._todo.list(ctx.session_id)
        if ctx.events is not None:
            await ctx.events.publish(
                TodoUpdated(session_id=ctx.session_id, items=[todo.item() for todo in todos])
            )
        return _render(todos)


class TodoAddInput(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TODO_CHARS, description="What has to be done")


class TodoAdd(_TodoTool[TodoAddInput]):
    """Adds an item to the session's todo list."""

    name = "todo_add"

    core = False
    description = (
        "Add an item to this conversation's todo list. Use the list to plan work with "
        "several steps; the user sees it."
    )
    input_model = TodoAddInput

    def label(self, params: TodoAddInput) -> str:
        return params.text

    async def run(self, params: TodoAddInput, ctx: ToolContext) -> ToolResult:
        try:
            await self._todo.add(ctx.session_id, params.text)
        except ValueError as exc:
            return error_result(f"Nothing was added: {exc}.")
        return await self._changed(ctx)


class TodoUpdateInput(BaseModel):
    number: int = Field(ge=1, description="Number of the item, as todo_list shows it")
    status: TodoStatus = Field(description="pending, in_progress or done")


class TodoUpdate(_TodoTool[TodoUpdateInput]):
    """Changes the status of a todo item."""

    name = "todo_update"

    core = False
    description = "Mark a todo item as pending, in_progress or done."
    input_model = TodoUpdateInput

    def label(self, params: TodoUpdateInput) -> str:
        return f"#{params.number} {params.status}"

    async def run(self, params: TodoUpdateInput, ctx: ToolContext) -> ToolResult:
        if await self._todo.set_status(ctx.session_id, params.number, params.status) is None:
            return error_result(f"There is no item {params.number} in the todo list.")
        return await self._changed(ctx)


class TodoListInput(BaseModel):
    pass


class TodoList(_TodoTool[TodoListInput]):
    """Shows the session's todo list."""

    name = "todo_list"

    core = False
    description = "Show this conversation's todo list with the number and status of each item."
    input_model = TodoListInput

    async def run(self, params: TodoListInput, ctx: ToolContext) -> ToolResult:
        return _render(await self._todo.list(ctx.session_id))


def todo_tools(todo: TodoRepo) -> list[Tool[Any]]:
    """The three todo tools, working on ``todo``."""
    return [TodoAdd(todo), TodoUpdate(todo), TodoList(todo)]
