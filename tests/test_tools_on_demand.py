"""Tools on demand: tools that are not core reach the model once it loads them."""

from pydantic import BaseModel

from phantasm.config import Config
from phantasm.core.turn import Turn
from phantasm.providers.base import TextDelta, ToolCall, ToolCallEvent
from phantasm.providers.fake import FakeProvider
from phantasm.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, text_result
from phantasm.tools.loading import TOOLS_LOAD, brief, more_tools_block
from tests.turns import Harness


class Params(BaseModel):
    text: str
    count: int = 1


class Core(Tool[Params]):
    name = "core_tool"
    description = "Always there. More words."
    input_model = Params
    default_policy = "allow"

    async def run(self, params: Params, ctx: ToolContext) -> ToolResult:
        return text_result("core")


class Rare(Core):
    name = "rare_tool"
    description = "Used now and then."
    core = False

    async def run(self, params: Params, ctx: ToolContext) -> ToolResult:
        return text_result(f"rare {params.text}")


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(Core())
    tools.register(Rare())
    return tools


def turn(harness: Harness, provider: FakeProvider, loaded: set[str], config: Config) -> Turn:
    return Turn(
        session_id=harness.session_id,
        text="hi",
        source="desktop",
        client=None,
        config=config,
        provider=provider,
        sessions=harness.sessions,
        messages=harness.messages,
        events=harness.bus,
        tools=registry(),
        retry_delays=(),
        loaded_tools=loaded,
    )


def names(call) -> list[str]:
    return [spec.name for spec in call.tools]


def test_registry_sends_core_and_loaded_tools() -> None:
    tools = registry()
    assert [s.name for s in tools.specs()] == ["core_tool", "rare_tool"]
    assert [s.name for s in tools.specs(set())] == ["core_tool"]
    assert [s.name for s in tools.specs({"rare_tool"})] == ["core_tool", "rare_tool"]
    assert [tool.name for tool in tools.deferred()] == ["rare_tool"]


def test_brief_lists_parameters_and_the_first_sentence() -> None:
    assert brief(Core()) == "- core_tool(text, count?): Always there"
    block = more_tools_block([Rare()])
    assert block is not None and block.startswith("# More tools")
    assert "- rare_tool(text, count?): Used now and then" in block
    assert more_tools_block([]) is None


async def test_loading_a_tool_sends_it_from_the_next_step(harness: Harness) -> None:
    load = ToolCall(id="l1", name=TOOLS_LOAD, input={"names": ["rare_tool", "nope"]})
    use = ToolCall(id="u1", name="rare_tool", input={"text": "x"})
    provider = FakeProvider(
        [[ToolCallEvent(call=load)], [ToolCallEvent(call=use)], [TextDelta(text="done")]]
    )
    loaded: set[str] = set()
    await turn(harness, provider, loaded, Config()).run()

    first, second, third = provider.calls
    assert names(first) == ["core_tool", TOOLS_LOAD]
    assert names(second) == ["core_tool", "rare_tool", TOOLS_LOAD]
    assert loaded == {"rare_tool"}
    system = first.messages[0].content[0].text
    assert "# More tools" in system and "rare_tool(text, count?)" in system
    load_result = second.messages[-1].content[0].text
    assert "Loaded: rare_tool" in load_result and "Not in # More tools: nope" in load_result
    assert third.messages[-1].content[0].text == "rare x"


async def test_a_conversation_keeps_its_loaded_tools(harness: Harness) -> None:
    provider = FakeProvider([[TextDelta(text="ok")]])
    await turn(harness, provider, {"rare_tool"}, Config()).run()
    assert names(provider.calls[0]) == ["core_tool", "rare_tool", TOOLS_LOAD]


async def test_without_deferring_every_tool_is_sent(harness: Harness) -> None:
    config = Config()
    config.agent.defer_tools = False
    provider = FakeProvider([[TextDelta(text="ok")]])
    await turn(harness, provider, set(), config).run()
    assert names(provider.calls[0]) == ["core_tool", "rare_tool"]
    assert "# More tools" not in provider.calls[0].messages[0].content[0].text


async def test_a_thinker_that_takes_all_tools_gets_them_all(harness: Harness) -> None:
    claude = FakeProvider([[TextDelta(text="ok")]])
    claude.takes_all_tools = True  # type: ignore[attr-defined]
    made = Turn(
        session_id=harness.session_id,
        text="hi",
        source="desktop",
        client=None,
        config=Config(),
        provider=FakeProvider([]),
        sessions=harness.sessions,
        messages=harness.messages,
        events=harness.bus,
        tools=registry(),
        retry_delays=(),
        thinker=(claude, "sonnet"),
    )
    await made.run()
    assert names(claude.calls[0]) == ["core_tool", "rare_tool"]
