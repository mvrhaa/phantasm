"""Tests of client actions offered to the model as tools."""

from typing import Any

import pytest

from tests.turns import FakeApprover, Harness, text_of
from zero.config import Config, PermissionsConfig, Policy
from zero.core.cancellation import CancelToken
from zero.core.events import ReplyTo
from zero.core.turn import Turn
from zero.protocol.messages import ActionDecl, Character, ClientInfo
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.tools.base import ToolContext, ToolResult, text_result
from zero.tools.builtin import builtin_registry
from zero.tools.client_actions import ClientActionTool, action_tools
from zero.tools.permissions import PathGuard, decide, stricter

FROM_DESKTOP = ReplyTo("conn-1", "m1")

SCREENSHOT = ActionDecl(
    name="screenshot",
    description="Capture the user's screen",
    input_schema={"type": "object", "properties": {"display": {"type": "integer"}}},
    policy="allow",
    timeout_s=15,
)


def client_with(*actions: ActionDecl, character: bool = False) -> ClientInfo:
    return ClientInfo(
        app="desktop",
        client_id="desk",
        actions=list(actions),
        character=Character(name="Z", poses=["idle"], moods=["neutral"]) if character else None,
    )


class RecordingCaller:
    """Answers every action at once and remembers the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def call(
        self, connection_id: str, action: ActionDecl, params: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        self.calls.append((connection_id, action.name, params))
        return text_result(f"{action.name} done")


@pytest.mark.parametrize(
    ("first", "second", "strictest"),
    [("allow", "ask", "ask"), ("deny", "allow", "deny"), ("ask", "ask", "ask")],
)
def test_stricter(first: Policy, second: Policy, strictest: Policy) -> None:
    assert stricter(first, second) == strictest


@pytest.mark.parametrize(
    ("declared", "configured", "expected"),
    [
        ("allow", None, "allow"),
        ("allow", "ask", "ask"),
        ("allow", "deny", "deny"),
        ("ask", "allow", "ask"),
        ("deny", "allow", "deny"),
        ("ask", "deny", "deny"),
    ],
)
def test_config_can_only_tighten_a_client_action(
    declared: Policy, configured: Policy | None, expected: Policy
) -> None:
    permissions = PermissionsConfig(tool_policies={"act": configured} if configured else {})

    assert decide("act", declared, permissions, config_tightens_only=True) == expected


def test_config_still_loosens_a_built_in_tool() -> None:
    permissions = PermissionsConfig(tool_policies={"write_file": "allow"})

    assert decide("write_file", "ask", permissions) == "allow"


def test_action_becomes_a_tool_with_the_declared_schema_and_policy() -> None:
    tool = ClientActionTool(SCREENSHOT, "conn-1", None)

    assert tool.spec().model_dump() == {
        "name": "screenshot",
        "description": "Capture the user's screen",
        "input_schema": {"type": "object", "properties": {"display": {"type": "integer"}}},
    }
    assert (tool.default_policy, tool.config_tightens_only) == ("allow", True)


def test_action_without_a_schema_takes_any_object() -> None:
    tool = ClientActionTool(ActionDecl(name="wave", description="Wave"), "c", None)

    assert tool.spec().input_schema == {"type": "object", "properties": {}}
    assert tool.default_policy == "ask"


def test_repeated_action_name_keeps_the_first() -> None:
    second = SCREENSHOT.model_copy(update={"description": "Another one"})

    tools = action_tools([SCREENSHOT, second], "c", None)

    assert [t.description for t in tools] == ["Capture the user's screen"]


async def test_action_input_goes_to_the_caller_as_it_is() -> None:
    caller = RecordingCaller()
    tool = ClientActionTool(SCREENSHOT, "conn-1", caller)
    params = tool.input_model.model_validate({"display": 2, "extra": [1, 2]})
    config = Config()
    ctx = ToolContext(
        session_id="s",
        turn_id="t",
        client_id="desk",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )

    result = await tool.run(params, ctx)

    assert text_of(result) == "screenshot done"
    assert caller.calls == [("conn-1", "screenshot", {"display": 2, "extra": [1, 2]})]


def test_extended_registry_adds_actions_after_the_built_in_tools() -> None:
    base = builtin_registry()

    extended = base.extended(action_tools([SCREENSHOT], "c", None))

    names = [spec.name for spec in extended.specs()]
    assert names[-1] == "screenshot"
    assert names[:-1] == [spec.name for spec in base.specs()]
    assert base.get("screenshot") is None


def test_action_cannot_replace_a_built_in_tool_or_set_expression(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_read = ActionDecl(name="read_file", description="Not the real one", policy="allow")
    fake_expression = ActionDecl(name="set_expression", description="Nope")

    extended = builtin_registry().extended(
        action_tools([fake_read, fake_expression, SCREENSHOT], "c", None),
        reserved=["set_expression"],
    )

    assert not isinstance(extended.get("read_file"), ClientActionTool)
    assert extended.get("set_expression") is None
    assert isinstance(extended.get("screenshot"), ClientActionTool)
    assert "Tool 'read_file' is already taken" in caplog.text


def turn_for(
    harness: Harness,
    provider: FakeProvider,
    client: ClientInfo | None,
    *,
    reply: ReplyTo | None = FROM_DESKTOP,
    caller: RecordingCaller | None = None,
    config: Config | None = None,
    approver: FakeApprover | None = None,
) -> Turn:
    return Turn(
        session_id=harness.session_id,
        text="hi",
        source="desktop",
        client=client,
        config=config or Config(),
        provider=provider,
        sessions=harness.sessions,
        messages=harness.messages,
        events=harness.bus,
        tools=builtin_registry(),
        approvals=approver,
        action_caller=caller,
        reply=reply,
        retry_delays=(0.0, 0.0),
    )


def call(name: str, call_id: str = "a1", **tool_input: Any) -> list[Any]:
    return [ToolCallEvent(call=ToolCall(id=call_id, name=name, input=tool_input))]


async def test_model_sees_the_actions_of_the_client_that_started_the_turn(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]])

    await turn_for(harness, provider, client_with(SCREENSHOT, character=True)).run()

    names = [tool.name for tool in provider.calls[0].tools]
    assert names[-2:] == ["screenshot", "set_expression"]


@pytest.mark.parametrize(
    ("client", "reply"),
    [(None, ReplyTo("conn-1", "m1")), (client_with(SCREENSHOT), None)],
    ids=["no client", "no connection"],
)
async def test_without_a_client_connection_there_are_no_actions(
    harness: Harness, client: ClientInfo | None, reply: ReplyTo | None
) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]])

    await turn_for(harness, provider, client, reply=reply).run()

    assert "screenshot" not in [tool.name for tool in provider.calls[0].tools]


async def test_calling_an_action_reaches_the_caller(harness: Harness) -> None:
    caller = RecordingCaller()
    provider = FakeProvider([call("screenshot", display=1), [TextDelta(text="I see it.")]])

    await turn_for(harness, provider, client_with(SCREENSHOT), caller=caller).run()

    assert caller.calls == [("conn-1", "screenshot", {"display": 1})]
    assert text_of(provider.calls[1].messages[-1]) == "screenshot done"


async def test_action_without_a_caller_is_an_error_for_the_model(harness: Harness) -> None:
    provider = FakeProvider([call("screenshot"), [TextDelta(text="Sorry.")]])

    await turn_for(harness, provider, client_with(SCREENSHOT)).run()

    result = provider.calls[1].messages[-1]
    assert (result.is_error, text_of(result)) == (
        True,
        "screenshot cannot be called: the client is not reachable.",
    )


async def test_config_makes_a_client_action_ask_first(harness: Harness) -> None:
    config = Config()
    config.permissions.tool_policies["screenshot"] = "ask"
    approver = FakeApprover("allow")
    caller = RecordingCaller()
    provider = FakeProvider([call("screenshot"), [TextDelta(text="ok")]])

    await turn_for(
        harness, provider, client_with(SCREENSHOT), caller=caller, config=config, approver=approver
    ).run()

    assert [ask.tool for ask in approver.asked] == ["screenshot"]
    assert len(caller.calls) == 1


def steps_script(action: str, tool_steps: int) -> list[list[Any]]:
    return [call(action, call_id=f"a{n}") for n in range(tool_steps)] + [[TextDelta(text="done")]]


async def test_computer_actions_raise_the_step_limit(harness: Harness) -> None:
    config = Config()
    config.agent.max_steps = 2
    config.agent.max_steps_computer = 5
    view = ActionDecl(name="computer_view", description="Look at the screen", policy="allow")
    provider = FakeProvider(steps_script("computer_view", 3))

    await turn_for(
        harness, provider, client_with(view), caller=RecordingCaller(), config=config
    ).run()

    assert len(provider.calls) == 4
    assert all(call.tools for call in provider.calls)
    assert harness.finished.status == "ok"


async def test_other_actions_keep_the_normal_limit(harness: Harness) -> None:
    config = Config()
    config.agent.max_steps = 2
    config.agent.max_steps_computer = 5
    provider = FakeProvider(steps_script("screenshot", 3))

    await turn_for(
        harness, provider, client_with(SCREENSHOT), caller=RecordingCaller(), config=config
    ).run()

    assert len(provider.calls) == 2
    assert provider.calls[1].tools == []
    assert harness.finished.status == "max_steps"
