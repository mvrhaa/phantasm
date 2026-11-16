"""What a plugin's ``tools.py`` imports: the ``tool`` decorator.

    from zero.plugin_api import tool

    @tool
    def weather(city: str, days: int = 1) -> str:
        \"\"\"The forecast for a city.\"\"\"
        ...

The function's name is the tool's name, its docstring the description, its
typed parameters the input. It may be ``async``; it returns text (or anything
JSON can hold) and raises to report a failure.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MARK = "__zero_tool__"


@dataclass(frozen=True)
class ToolMark:
    name: str
    description: str


def tool(
    func: Callable[..., Any] | None = None, *, name: str | None = None, description: str = ""
) -> Any:
    """Make a function of a plugin one of Zero's tools."""

    def mark(target: Callable[..., Any]) -> Callable[..., Any]:
        text = description or (target.__doc__ or "").strip() or target.__name__
        setattr(target, MARK, ToolMark(name or target.__name__, " ".join(text.split())))
        return target

    return mark(func) if func is not None else mark
