"""Tests of the tool-call cycle of a turn."""

from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel

from tests.turns import FuncTool, Harness, registry, roles, text_of
from zero.config import Config
from zero.protocol.messages import ClientInfo, TextPart, ToolStarted
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent, UsageEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, text_result


class AddInput(BaseModel):
    a: int
    b: int


class Add(Tool[AddInput]):
    """Adds two numbers; every input it gets is appended to ``log``."""

    name = "add"
    description = "Add two numbers"
    input_model = AddInput

    def __init__(self, log: list[dict[str, Any]] | None = None) -> None:
        self.log = log

    async def run(self, params: AddInput, ctx: ToolContext) -> ToolResult:
        if self.log is not None:
            self.log.append(params.model_dump())
        return text_result(str(params.a + params.b))


def call(name: str = "add", call_id: str = "call-1", **tool_input: Any) -> ToolCallEvent:
    return ToolCallEvent(call=ToolCall(id=call_id, name=name, input=tool_input))


def calculator(log: list[dict[str, Any]] | None = None) -> ToolRegistry:
    """A registry with one tool, ``add``."""
    return registry(Add(log))


# --- Tool-call cycle ---------------------------------------------------------


async def test_tool_call_and_answer_take_two_provider_calls(harness: Harness) -> None:
    provider = FakeProvider([[call(a=2, b=3)], [TextDelta(text="5")]])

    await harness.run(provider, calculator())

    first, second = provider.calls
    assert [tool.name for tool in first.tools] == ["add"]
    assert [tool.name for tool in second.tools] == ["add"]
    assistant, result = second.messages[-2:]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == [ToolCall(id="call-1", name="add", input={"a": 2, "b": 3})]
    assert (result.role, result.tool_call_id, result.is_error) == ("tool", "call-1", False)
    assert result.content == [TextPart(text="5")]
    assert (harness.finished.status, harness.finished.text) == ("ok", "5")


async def test_tool_steps_are_stored_with_the_turn(harness: Harness) -> None:
    provider = FakeProvider([[call(a=2, b=3)], [TextDelta(text="5")]])

    turn = await harness.run(provider, calculator())

    stored = await harness.stored()
    assert roles(stored) == "user assistant tool assistant"
    user, assistant, tool, answer = stored
    assert assistant.tool_calls == [{"id": "call-1", "name": "add", "input": {"a": 2, "b": 3}}]
    assert assistant.content == []
    assert (tool.tool_call_id, tool.is_error, text_of(tool)) == ("call-1", False, "5")
    assert (answer.tool_calls, text_of(answer)) == ([], "5")
    assert all(m.turn_id == turn.id for m in (user, assistant, tool, answer))
    assert all((m.status, m.source) == ("ok", "desktop") for m in (assistant, tool, answer))


async def test_clients_see_the_tool_run_between_steps(harness: Harness) -> None:
    provider = FakeProvider([[call(a=2, b=3)], [TextDelta(text="5")]])

    turn = await harness.run(provider, calculator())

    assert harness.types == [
        "turn.started",
        "tool.started",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    started, finished = harness.tool_events()
    assert started == ToolStarted(
        session_id=harness.session_id,
        turn_id=turn.id,
        call_id="call-1",
        tool="add",
        input={"a": 2, "b": 3},
    )
    assert (finished.call_id, finished.status) == ("call-1", "ok")  # type: ignore[union-attr]


async def test_unknown_tool_is_an_error_for_the_model_not_for_the_turn(harness: Harness) -> None:
    provider = FakeProvider([[call(name="fly")], [TextDelta(text="I cannot fly.")]])

    await harness.run(provider, calculator())

    result = provider.calls[1].messages[-1]
    assert (result.role, result.is_error) == ("tool", True)
    assert text_of(result) == "Unknown tool 'fly'. Available tools: add."
    assert harness.tool_events()[1].status == "error"  # type: ignore[union-attr]
    assert (harness.finished.status, harness.finished.text) == ("ok", "I cannot fly.")
    stored_result = (await harness.stored())[2]
    assert (stored_result.role, stored_result.is_error) == ("tool", True)


async def test_failing_tool_does_not_break_the_turn(harness: Harness) -> None:
    async def broken(params: Any, ctx: ToolContext) -> str:
        raise RuntimeError("disk is full")

    provider = FakeProvider([[call(name="save")], [TextDelta(text="Could not save.")]])

    await harness.run(provider, registry(FuncTool("save", broken)))

    result = provider.calls[1].messages[-1]
    assert result.is_error
    assert text_of(result) == "Tool 'save' failed: disk is full"
    assert harness.finished.status == "ok"


async def test_invalid_input_goes_back_to_the_model(harness: Harness) -> None:
    log: list[dict[str, Any]] = []
    provider = FakeProvider(
        [[call(a=2)], [call(call_id="call-2", a=2, b=3)], [TextDelta(text="5")]]
    )

    await harness.run(provider, calculator(log))

    result = provider.calls[1].messages[-1]
    assert result.is_error
    assert text_of(result) == "Invalid input for 'add': b: Field required. Fix it and call again."
    assert log == [{"a": 2, "b": 3}]
    assert harness.finished.text == "5"


async def test_tools_get_the_context_of_their_turn(harness: Harness) -> None:
    contexts: list[ToolContext] = []

    async def remember(params: Any, ctx: ToolContext) -> str:
        contexts.append(ctx)
        return "ok"

    provider = FakeProvider([[call(name="remember")], [TextDelta(text="done")]])
    client = ClientInfo(app="desktop", client_id="c-42")

    turn = await harness.run(provider, registry(FuncTool("remember", remember)), client=client)

    (ctx,) = contexts
    assert (ctx.session_id, ctx.turn_id, ctx.client_id) == (harness.session_id, turn.id, "c-42")
    assert ctx.cancel is turn.cancel
    assert ctx.config.agent.max_steps == 40


async def test_calls_of_one_step_run_in_order_and_share_one_message(harness: Harness) -> None:
    log: list[dict[str, Any]] = []
    provider = FakeProvider(
        [
            [call(call_id="c1", a=1, b=1), call(call_id="c2", a=2, b=2)],
            [TextDelta(text="2 and 4")],
        ]
    )

    await harness.run(provider, calculator(log))

    assert log == [{"a": 1, "b": 1}, {"a": 2, "b": 2}]
    assistant, first, second = provider.calls[1].messages[-3:]
    assert [c.id for c in assistant.tool_calls] == ["c1", "c2"]
    assert [(first.tool_call_id, text_of(first)), (second.tool_call_id, text_of(second))] == [
        ("c1", "2"),
        ("c2", "4"),
    ]
    stored = await harness.stored()
    assert roles(stored) == "user assistant tool tool assistant"
    assert [e.call_id for e in harness.tool_events()] == ["c1", "c1", "c2", "c2"]


async def test_text_of_every_step_reaches_the_client_and_the_history(harness: Harness) -> None:
    provider = FakeProvider(
        [[TextDelta(text="Let me count. "), call(a=2, b=3)], [TextDelta(text="It is 5.")]]
    )

    await harness.run(provider, calculator())

    assert harness.deltas == ["Let me count. ", "It is 5."]
    assert harness.finished.text == "Let me count. It is 5."
    assert text_of(provider.calls[1].messages[-2]) == "Let me count. "
    _, assistant, _, answer = await harness.stored()
    assert (text_of(assistant), len(assistant.tool_calls)) == ("Let me count. ", 1)
    assert text_of(answer) == "It is 5."


async def test_usage_adds_up_over_the_steps(harness: Harness) -> None:
    provider = FakeProvider(
        [
            [call(a=1, b=1), UsageEvent(input_tokens=10, output_tokens=3)],
            [TextDelta(text="2"), UsageEvent(input_tokens=15, output_tokens=1)],
        ]
    )

    await harness.run(provider, calculator())

    assert harness.finished.usage.model_dump() == {"input_tokens": 25, "output_tokens": 4}


async def test_next_turn_sends_the_earlier_tool_exchange(harness: Harness) -> None:
    provider = FakeProvider(
        [[call(a=2, b=3)], [TextDelta(text="5")], [TextDelta(text="You are welcome.")]]
    )

    await harness.run(provider, calculator(), text="add 2 and 3")
    await harness.run(provider, calculator(), text="thanks")

    messages = provider.calls[2].messages
    assert roles(messages) == "system user assistant tool assistant user"
    assert messages[2].tool_calls == [ToolCall(id="call-1", name="add", input={"a": 2, "b": 3})]
    assert messages[3].tool_call_id == "call-1"


# --- Step limit --------------------------------------------------------------


async def test_last_allowed_step_goes_without_tools(harness: Harness) -> None:
    log: list[dict[str, Any]] = []
    provider = FakeProvider(
        [
            [call(call_id="c1", a=1, b=1)],
            [call(call_id="c2", a=2, b=2)],
            [TextDelta(text="Out of steps."), call(call_id="c3", a=3, b=3)],
        ]
    )

    await harness.run(provider, calculator(log), max_steps=3)

    assert [[tool.name for tool in c.tools] for c in provider.calls] == [["add"], ["add"], []]
    assert log == [{"a": 1, "b": 1}, {"a": 2, "b": 2}]
    finished = harness.finished
    assert (finished.status, finished.text, finished.error) == ("max_steps", "Out of steps.", None)
    stored = await harness.stored()
    assert roles(stored) == "user assistant tool assistant tool assistant"
    assert (stored[-1].tool_calls, text_of(stored[-1]), stored[-1].status) == (
        [],
        "Out of steps.",
        "ok",
    )


async def test_turn_that_ends_before_the_limit_is_ok(harness: Harness) -> None:
    provider = FakeProvider([[call(a=1, b=1)], [TextDelta(text="2")]])

    await harness.run(provider, calculator(), max_steps=3)

    assert len(provider.calls) == 2
    assert provider.calls[1].tools != []
    assert harness.finished.status == "ok"


async def test_provider_failure_after_a_tool_step_keeps_calls_paired(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="Checking. "), call(a=1, b=1)]])  # no step 2

    await harness.run(provider, calculator())

    assert harness.finished.status == "error"
    assert harness.finished.text == "Checking. "
    stored = await harness.stored()
    assert [(m.role, m.status) for m in stored] == [
        ("user", "ok"),
        ("assistant", "ok"),
        ("tool", "ok"),
        ("assistant", "error"),
    ]
    assert (stored[1].tool_calls[0]["id"], stored[2].tool_call_id) == ("call-1", "call-1")
    assert stored[-1].content == []


async def test_empty_final_answer_is_not_stored(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="Adding."), call(a=1, b=1)], []])

    await harness.run(provider, calculator())

    assert (harness.finished.status, harness.finished.text) == ("ok", "Adding.")
    assert roles(await harness.stored()) == "user assistant tool"


# --- Over the WebSocket ------------------------------------------------------


def test_tool_call_over_websocket() -> None:
    provider = FakeProvider([[call(name="fly")], [TextDelta(text="No wings.")]])
    with TestClient(create_app(Config(), provider)) as client:
        state = get_state(client.app)  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={state.token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            assert ws.receive_json()["type"] == "hello.ok"
            ws.send_json({"type": "message.send", "id": "m1", "text": "fly", "source": "desktop"})
            events = []
            while not events or events[-1]["type"] != "turn.finished":
                events.append(ws.receive_json())
            assert ws.receive_json()["type"] == "session.updated"

            ws.send_json({"type": "history.get", "session_id": events[0]["session"]["id"]})
            stored = ws.receive_json()["messages"]

    assert [e["type"] for e in events] == [
        "session.created",
        "turn.started",
        "tool.started",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    assert (events[2]["tool"], events[3]["status"]) == ("fly", "error")
    assert (events[-1]["status"], events[-1]["text"]) == ("ok", "No wings.")
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
