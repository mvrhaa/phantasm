"""Tests for zero.tools.base: the tool interface and the registry."""

import logging
from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.turns import FuncTool, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.protocol.messages import ImagePart, TextPart
from zero.providers.base import ToolCall
from zero.tools.base import (
    CANCELLED,
    MAX_OUTPUT_CHARS,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    error_result,
    text_result,
    truncate_text,
)
from zero.tools.permissions import PathGuard


class GreetInput(BaseModel):
    name: str = Field(description="Who to greet")
    times: int = Field(default=1, ge=1, le=3)


class Greet(Tool[GreetInput]):
    name = "greet"
    description = "Say hello"
    input_model = GreetInput
    default_policy = "allow"

    def __init__(self) -> None:
        self.seen: list[tuple[GreetInput, ToolContext]] = []

    def label(self, params: GreetInput) -> str:
        return f"hello {params.name}"

    async def run(self, params: GreetInput, ctx: ToolContext) -> ToolResult:
        self.seen.append((params, ctx))
        return text_result(" ".join([f"Hello, {params.name}!"] * params.times), summary="greeted")


class Echo(Tool[GreetInput]):
    """A tool that keeps every default, to check what ``Tool`` gives for free."""

    name = "echo"
    description = "Echo"
    input_model = GreetInput

    async def run(self, params: GreetInput, ctx: ToolContext) -> ToolResult:
        return text_result(params.name)


def context() -> ToolContext:
    config = Config()
    return ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )


def registry_with(*tools: Tool[Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


async def execute(
    registry: ToolRegistry, tool: str, tool_input: dict[str, Any] | None = None
) -> ToolResult:
    call = ToolCall(id="call-1", name=tool, input=tool_input or {})
    return await registry.execute(call, context())


def returning(result: ToolResult) -> FuncTool:
    """A tool whose ``run`` returns ``result`` as it is."""

    class Fixed(FuncTool):
        async def run(self, params: Any, ctx: ToolContext) -> ToolResult:
            return result

    async def unused(params: Any, ctx: ToolContext) -> str:
        return ""

    return Fixed("fixed", unused)


def raising(exc: BaseException) -> FuncTool:
    async def fail(params: Any, ctx: ToolContext) -> str:
        raise exc

    return FuncTool("broken", fail)


# --- Tool --------------------------------------------------------------------


def test_spec_takes_its_schema_from_the_input_model() -> None:
    spec = Greet().spec()

    assert (spec.name, spec.description) == ("greet", "Say hello")
    assert spec.input_schema == GreetInput.model_json_schema()
    assert spec.input_schema["required"] == ["name"]
    assert spec.input_schema["properties"]["times"]["maximum"] == 3


def test_tool_defaults_ask_for_approval_and_label_with_the_name() -> None:
    tool = Echo()

    assert tool.default_policy == "ask"
    assert tool.label(GreetInput(name="Ann")) == "echo"


def test_tool_can_label_its_call() -> None:
    assert Greet().label(GreetInput(name="Ann")) == "hello Ann"


def test_result_helpers() -> None:
    assert text_result("ok", summary="1 line") == ToolResult(
        content=[TextPart(text="ok")], summary="1 line"
    )
    assert error_result("nope") == ToolResult(content=[TextPart(text="nope")], is_error=True)


# --- Registration ------------------------------------------------------------


def test_registry_lists_specs_in_registration_order() -> None:
    registry = registry_with(Greet(), Echo())

    assert [spec.name for spec in registry.specs()] == ["greet", "echo"]


def test_registry_finds_a_tool_by_name() -> None:
    greet = Greet()
    registry = registry_with(greet)

    assert registry.get("greet") is greet
    assert registry.get("missing") is None


def test_registry_refuses_a_second_tool_with_the_same_name() -> None:
    registry = registry_with(Greet())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(Greet())


def test_empty_registry_has_no_specs() -> None:
    assert ToolRegistry().specs() == []


# --- Execution ---------------------------------------------------------------


async def test_execute_runs_the_tool_with_validated_input_and_context() -> None:
    greet = Greet()
    ctx = context()

    result = await registry_with(greet).execute(
        ToolCall(id="c", name="greet", input={"name": "Ann", "times": 2}), ctx
    )

    assert result == ToolResult(
        content=[TextPart(text="Hello, Ann! Hello, Ann!")], summary="greeted"
    )
    ((params, seen_ctx),) = greet.seen
    assert params == GreetInput(name="Ann", times=2)
    assert seen_ctx is ctx


async def test_unknown_tool_is_an_error_naming_the_available_ones() -> None:
    result = await execute(registry_with(Greet(), Echo()), "fly")

    assert result.is_error
    assert text_of(result) == "Unknown tool 'fly'. Available tools: greet, echo."


async def test_unknown_tool_in_an_empty_registry() -> None:
    result = await execute(ToolRegistry(), "fly")

    assert text_of(result) == "Unknown tool 'fly'. Available tools: none."


@pytest.mark.parametrize(
    ("tool_input", "problem"),
    [
        ({}, "name: Field required"),
        ({"name": "Ann", "times": 5}, "times: Input should be less than or equal to 3"),
        ({"name": ["Ann"]}, "name: Input should be a valid string"),
    ],
)
async def test_invalid_input_is_an_error_and_the_tool_does_not_run(
    tool_input: dict[str, Any], problem: str
) -> None:
    greet = Greet()

    result = await execute(registry_with(greet), "greet", tool_input)

    assert result.is_error
    assert text_of(result) == f"Invalid input for 'greet': {problem}. Fix it and call again."
    assert greet.seen == []


async def test_invalid_input_message_does_not_repeat_the_value() -> None:
    result = await execute(registry_with(Greet()), "greet", {"name": "Ann", "times": "sk-secret"})

    assert "sk-secret" not in text_of(result)


async def test_failing_tool_is_an_error_and_its_message_is_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = await execute(registry_with(raising(OSError("disk is full"))), "broken")

    assert result.is_error
    assert text_of(result) == "Tool 'broken' failed: disk is full"
    assert "Tool broken failed: OSError" in caplog.text
    assert "disk is full" not in caplog.text


async def test_failure_without_a_message_names_the_exception() -> None:
    result = await execute(registry_with(raising(RuntimeError())), "broken")

    assert text_of(result) == "Tool 'broken' failed: RuntimeError"


async def test_tool_that_notices_cancellation_gives_a_cancelled_result() -> None:
    result = await execute(registry_with(raising(TurnCancelled())), "broken")

    assert result.is_error
    assert text_of(result) == CANCELLED


async def test_error_returned_by_the_tool_is_passed_on() -> None:
    result = await execute(registry_with(returning(error_result("no such file"))), "fixed")

    assert result == error_result("no such file")


# --- Output size -------------------------------------------------------------


def test_short_text_is_not_truncated() -> None:
    text = "x" * MAX_OUTPUT_CHARS

    assert truncate_text(text) == text


def test_long_text_keeps_its_head_and_tail() -> None:
    text = "h" * 20_000 + "m" * 30_000 + "t" * 20_000

    truncated = truncate_text(text)

    assert truncated == "h" * 15_000 + "\n[… 40000 chars truncated …]\n" + "t" * 15_000


def test_truncate_with_a_custom_limit() -> None:
    assert truncate_text("abcdefghij", limit=4) == "ab\n[… 6 chars truncated …]\nij"


async def test_execute_truncates_every_text_part_and_keeps_the_rest() -> None:
    image = ImagePart(mime="image/png", data_b64="AAAA" * 20_000)
    big = "a" * 40_000
    result = ToolResult(
        content=[TextPart(text=big), image, TextPart(text="small")], summary="3 parts"
    )

    executed = await execute(registry_with(returning(result)), "fixed")

    first, second, third = executed.content
    assert isinstance(first, TextPart)
    assert first.text == truncate_text(big)
    assert "[… 10000 chars truncated …]" in first.text
    assert second == image
    assert third == TextPart(text="small")
    assert (executed.summary, executed.is_error) == ("3 parts", False)
    assert result.content[0] == TextPart(text=big)  # the tool's own result is not changed
