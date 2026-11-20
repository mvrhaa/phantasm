"""Tests for phantasm.tools.permissions.decide and how ToolRegistry.execute obeys it."""

from itertools import product
from typing import Any

import pytest

from phantasm.config import Config, PermissionsConfig, Policy
from phantasm.core.cancellation import CancelToken
from phantasm.providers.base import ToolCall
from phantasm.tools.base import NEEDS_APPROVAL, Preview, ToolContext, ToolRegistry, ToolResult
from phantasm.tools.permissions import PathGuard, decide
from tests.turns import FuncTool, text_of

POLICIES: list[Policy] = ["allow", "ask", "deny"]


def permissions(
    configured: Policy | None = None, *, bypass: bool = False, tool: str = "tool"
) -> PermissionsConfig:
    return PermissionsConfig(
        bypass_approvals=bypass, tool_policies={tool: configured} if configured else {}
    )


@pytest.mark.parametrize(
    ("default", "configured", "saved", "bypass", "needs_approval", "expected"),
    [
        ("allow", None, False, False, False, "allow"),
        ("ask", None, False, False, False, "ask"),
        ("deny", None, False, False, False, "deny"),
        ("ask", "allow", False, False, False, "allow"),
        ("allow", "ask", False, False, False, "ask"),
        ("allow", "deny", False, False, False, "deny"),
        ("ask", None, True, False, False, "allow"),
        ("ask", None, False, True, False, "allow"),
        ("ask", "deny", True, True, False, "deny"),
        ("deny", None, True, True, False, "deny"),
        ("allow", None, False, False, True, "ask"),
        ("allow", None, True, True, True, "ask"),
        ("allow", "deny", False, False, True, "deny"),
        ("deny", "allow", False, False, False, "allow"),
    ],
    ids=[
        "default allow",
        "default ask",
        "default deny",
        "config loosens",
        "config tightens",
        "config denies",
        "saved always allows",
        "bypass allows",
        "config deny beats saved and bypass",
        "default deny beats saved and bypass",
        "tool demands approval",
        "tool demand beats saved and bypass",
        "deny beats tool demand",
        "config can turn on a tool off by default",
    ],
)
def test_decide(
    default: Policy,
    configured: Policy | None,
    saved: bool,
    bypass: bool,
    needs_approval: bool,
    expected: Policy,
) -> None:
    rules = ["tool"] if saved else []
    config = permissions(configured, bypass=bypass)

    assert decide("tool", default, config, rules, needs_approval=needs_approval) == expected


def spec_9_4(policy: Policy, saved: bool, bypass: bool, needs_approval: bool) -> Policy:
    """The order of §9.4 written out once more, for the exhaustive check."""
    if policy == "deny":
        return "deny"
    if needs_approval:
        return "ask"
    if saved or bypass:
        return "allow"
    return policy


@pytest.mark.parametrize(
    ("default", "configured", "saved", "bypass", "needs_approval"),
    list(product(POLICIES, [None, *POLICIES], [False, True], [False, True], [False, True])),
)
def test_every_combination_follows_the_spec(
    default: Policy,
    configured: Policy | None,
    saved: bool,
    bypass: bool,
    needs_approval: bool,
) -> None:
    rules = ["tool"] if saved else []
    decision = decide(
        "tool",
        default,
        permissions(configured, bypass=bypass),
        rules,
        needs_approval=needs_approval,
    )

    assert decision == spec_9_4(configured or default, saved, bypass, needs_approval)
    if (configured or default) == "deny":
        assert decision == "deny"


def test_rules_and_config_of_other_tools_do_not_matter() -> None:
    config = PermissionsConfig(tool_policies={"other": "deny"})

    assert decide("tool", "ask", config, ["other"]) == "ask"


def test_default_config_lets_read_tools_run_and_asks_before_writing() -> None:
    config = Config().permissions

    assert decide("read_file", "allow", config) == "allow"
    assert decide("write_file", "ask", config) == "ask"
    assert decide("run_command", "ask", config) == "ask"


class Recorder:
    def __init__(self) -> None:
        self.ran = 0


def tool_with(policy: Policy, recorder: Recorder, reason: str | None = None) -> FuncTool:
    async def work(params: Any, ctx: ToolContext) -> str:
        recorder.ran += 1
        return "done"

    class Policed(FuncTool):
        async def preview(self, params: Any, ctx: ToolContext) -> Preview:
            return Preview(approval_reason=reason)

    tool = Policed("work", work)
    tool.default_policy = policy
    return tool


async def execute(tool: FuncTool, config: Config) -> ToolResult:
    registry = ToolRegistry()
    registry.register(tool)
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )
    return await registry.execute(ToolCall(id="c", name="work"), ctx)


def config_with(configured: Policy | None = None, *, bypass: bool = False) -> Config:
    config = Config()
    config.permissions = permissions(configured, bypass=bypass, tool="work")
    return config


async def test_allowed_tool_runs() -> None:
    recorder = Recorder()

    result = await execute(tool_with("allow", recorder), config_with())

    assert (text_of(result), recorder.ran) == ("done", 1)


async def test_tool_denied_in_the_config_does_not_run() -> None:
    recorder = Recorder()

    result = await execute(tool_with("allow", recorder), config_with("deny", bypass=True))

    assert result.is_error
    assert text_of(result) == "work is turned off in the settings."
    assert recorder.ran == 0


async def test_tool_that_asks_is_refused_until_approvals_exist() -> None:
    recorder = Recorder()

    result = await execute(tool_with("ask", recorder), config_with())

    assert text_of(result) == f"work is set to ask the user first, {NEEDS_APPROVAL}"
    assert recorder.ran == 0


async def test_bypass_lets_a_tool_that_asks_run() -> None:
    recorder = Recorder()

    result = await execute(tool_with("ask", recorder), config_with(bypass=True))

    assert (result.is_error, recorder.ran) == (False, 1)


async def test_config_can_let_a_tool_run_unasked() -> None:
    recorder = Recorder()

    result = await execute(tool_with("ask", recorder), config_with("allow"))

    assert (result.is_error, recorder.ran) == (False, 1)


async def test_bypass_does_not_skip_an_approval_the_tool_demands() -> None:
    recorder = Recorder()
    tool = tool_with("allow", recorder, reason="~/.env may hold secrets")

    result = await execute(tool, config_with(bypass=True))

    assert text_of(result) == f"~/.env may hold secrets, {NEEDS_APPROVAL}"
    assert recorder.ran == 0
