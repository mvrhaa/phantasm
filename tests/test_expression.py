"""Tests of the character's expression."""

from typing import Any

import pytest

from tests.turns import FuncTool, Harness, registry, roles, text_of
from zero.core.expression import (
    EXPRESSION_SET,
    EXPRESSION_TOOL,
    ExpressionTool,
    default_expression,
)
from zero.protocol.messages import (
    Character,
    ClientInfo,
    Expression,
    ExpressionState,
    ToolFinished,
    ToolStarted,
)
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.tools.base import ToolContext

CHARACTER = Character(name="Zero", poses=["idle", "sit", "wave"], moods=["happy", "neutral", "sad"])
CLIENT = ClientInfo(app="desktop", client_id="c-1", character=CHARACTER)


def express(pose: Any, mood: Any, call_id: str = "e1") -> ToolCallEvent:
    return ToolCallEvent(
        call=ToolCall(id=call_id, name=EXPRESSION_TOOL, input={"pose": pose, "mood": mood})
    )


def expressions(harness: Harness) -> list[tuple[str, str]]:
    return [(e.pose, e.mood) for e in harness.client.received if isinstance(e, Expression)]


def test_schema_lists_the_poses_and_moods_of_the_character() -> None:
    spec = ExpressionTool(CHARACTER).spec

    assert spec.name == "set_expression"
    assert spec.input_schema == {
        "type": "object",
        "properties": {
            "pose": {"type": "string", "enum": ["idle", "sit", "wave"]},
            "mood": {"type": "string", "enum": ["happy", "neutral", "sad"]},
        },
        "required": ["pose", "mood"],
        "additionalProperties": False,
    }


def test_default_is_the_first_pose_and_a_neutral_mood() -> None:
    assert default_expression(CHARACTER) == ExpressionState(pose="idle", mood="neutral")


def test_default_mood_is_the_first_one_when_there_is_no_neutral() -> None:
    character = Character(name="Z", poses=["stand"], moods=["calm", "angry"])

    assert default_expression(character) == ExpressionState(pose="stand", mood="calm")


def test_apply_changes_the_current_expression() -> None:
    tool = ExpressionTool(CHARACTER)
    assert (tool.current, tool.final) == (None, default_expression(CHARACTER))

    state = tool.apply({"pose": "wave", "mood": "happy"})

    assert state == ExpressionState(pose="wave", mood="happy")
    assert tool.current == tool.final == state


@pytest.mark.parametrize(
    ("tool_input", "message"),
    [
        (
            {"pose": "dance", "mood": "happy"},
            "Invalid expression: pose must be one of idle, sit, wave (got 'dance').",
        ),
        (
            {"pose": "sit"},
            "Invalid expression: mood must be one of happy, neutral, sad (got None).",
        ),
        (
            {"pose": 1, "mood": "Happy"},
            "Invalid expression: pose must be one of idle, sit, wave (got 1); "
            "mood must be one of happy, neutral, sad (got 'Happy').",
        ),
    ],
)
def test_apply_rejects_what_the_character_cannot_show(
    tool_input: dict[str, Any], message: str
) -> None:
    tool = ExpressionTool(CHARACTER)

    with pytest.raises(ValueError) as error:
        tool.apply(tool_input)

    assert str(error.value) == message
    assert tool.current is None


async def test_turn_with_a_character_offers_set_expression(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]])

    await harness.run(provider, client=CLIENT)

    assert provider.calls[0].tools == [ExpressionTool(CHARACTER).spec]
    assert harness.finished.expression == ExpressionState(pose="idle", mood="neutral")


async def test_turn_without_a_character_has_no_expression(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]])

    await harness.run(provider)

    assert provider.calls[0].tools == []
    assert harness.finished.expression is None


async def test_expression_set_with_the_answer_ends_the_turn(harness: Harness) -> None:
    provider = FakeProvider(
        [[TextDelta(text="Hi!"), express("wave", "happy")], [TextDelta(text="?")]]
    )

    turn = await harness.run(provider, client=CLIENT)

    assert len(provider.calls) == 1
    assert harness.types == [
        "turn.started",
        "text.delta",
        "tool.started",
        "expression",
        "tool.finished",
        "turn.finished",
    ]
    started, expression, finished = harness.client.received[2:5]
    assert isinstance(started, ToolStarted) and started.hidden
    assert expression == Expression(
        session_id=harness.session_id, turn_id=turn.id, pose="wave", mood="happy"
    )
    assert isinstance(finished, ToolFinished) and finished.status == "ok"
    assert (harness.finished.status, harness.finished.text) == ("ok", "Hi!")
    assert harness.finished.expression == ExpressionState(pose="wave", mood="happy")

    stored = await harness.stored()
    assert roles(stored) == "user assistant tool"
    assert (text_of(stored[1]), text_of(stored[2])) == ("Hi!", EXPRESSION_SET)


async def test_expression_before_the_answer_lets_the_model_go_on(harness: Harness) -> None:
    provider = FakeProvider([[express("sit", "sad")], [TextDelta(text="Sorry to hear.")]])

    await harness.run(provider, client=CLIENT)

    assert len(provider.calls) == 2
    assert provider.calls[1].messages[-1].content[0].text == EXPRESSION_SET  # type: ignore[union-attr]
    assert harness.finished.text == "Sorry to hear."
    assert harness.finished.expression == ExpressionState(pose="sit", mood="sad")
    assert roles(await harness.stored()) == "user assistant tool assistant"


async def test_last_expression_of_the_turn_wins(harness: Harness) -> None:
    provider = FakeProvider(
        [[express("wave", "happy", "e1")], [TextDelta(text="Oh no."), express("sit", "sad", "e2")]]
    )

    await harness.run(provider, client=CLIENT)

    assert expressions(harness) == [("wave", "happy"), ("sit", "sad")]
    assert harness.finished.expression == ExpressionState(pose="sit", mood="sad")


async def test_invalid_expression_goes_back_to_the_model(harness: Harness) -> None:
    provider = FakeProvider(
        [
            [TextDelta(text="Hi!"), express("dance", "happy")],
            [TextDelta(text=" Better."), express("wave", "happy", "e2")],
        ]
    )

    await harness.run(provider, client=CLIENT)

    assert len(provider.calls) == 2
    result = provider.calls[1].messages[-1]
    assert result.is_error
    assert (
        text_of(result) == "Invalid expression: pose must be one of idle, sit, wave (got 'dance')."
    )
    assert [e.status for e in harness.tool_events() if isinstance(e, ToolFinished)] == [
        "error",
        "ok",
    ]
    assert expressions(harness) == [("wave", "happy")]
    assert harness.finished.expression == ExpressionState(pose="wave", mood="happy")


async def test_expression_next_to_another_tool_does_not_end_the_turn(harness: Harness) -> None:
    async def add(params: Any, ctx: ToolContext) -> str:
        return "5"

    tools = registry(FuncTool("add", add))
    provider = FakeProvider(
        [
            [
                TextDelta(text="Adding. "),
                ToolCallEvent(call=ToolCall(id="a1", name="add")),
                express("sit", "neutral"),
            ],
            [TextDelta(text="It is 5.")],
        ]
    )

    await harness.run(provider, tools, client=CLIENT)

    assert [tool.name for tool in provider.calls[0].tools] == ["add", "set_expression"]
    assert len(provider.calls) == 2
    assert harness.finished.text == "Adding. It is 5."


async def test_failed_turn_still_reports_the_default_expression(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="hi")]], fail_times=1, fail_retryable=False)

    await harness.run(provider, client=CLIENT)

    assert harness.finished.status == "error"
    assert harness.finished.expression == ExpressionState(pose="idle", mood="neutral")


async def test_set_expression_without_a_character_is_an_unknown_tool(harness: Harness) -> None:
    provider = FakeProvider(
        [[TextDelta(text="Hi!"), express("wave", "happy")], [TextDelta(text=" Hello.")]]
    )

    await harness.run(provider)

    assert len(provider.calls) == 2
    assert provider.calls[1].messages[-1].is_error
    assert expressions(harness) == []
    assert harness.finished.expression is None
