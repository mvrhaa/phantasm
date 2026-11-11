"""Tests of "always allow" rules: the repository, the registry using it and the protocol."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import FakeApprover, FuncTool, registry, text_of
from zero.approvals.manager import Decision
from zero.config import Config, Policy
from zero.core.cancellation import CancelToken
from zero.paths import db_path
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.store.approvals import ApprovalRule, ApprovalRulesRepo
from zero.store.db import Database
from zero.tools.base import NEEDS_APPROVAL, ToolContext, ToolResult
from zero.tools.permissions import PathGuard


@pytest.fixture
async def rules() -> AsyncIterator[ApprovalRulesRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield ApprovalRulesRepo(db)


# --- ApprovalRulesRepo -------------------------------------------------------


async def test_no_rules_at_first(rules: ApprovalRulesRepo) -> None:
    assert await rules.always_allowed() == set()


async def test_rule_is_saved(rules: ApprovalRulesRepo, frozen_clock) -> None:
    rule = await rules.allow_always("write_file")

    assert rule == ApprovalRule(tool="write_file", decision="allow", created_at=frozen_clock())
    assert await rules.always_allowed() == {"write_file"}


async def test_saving_a_rule_again_keeps_the_first(rules: ApprovalRulesRepo, frozen_clock) -> None:
    first = await rules.allow_always("write_file")
    frozen_clock.advance(60)

    again = await rules.allow_always("write_file")

    assert again == first
    assert await rules.always_allowed() == {"write_file"}


async def test_rules_are_kept_per_tool(rules: ApprovalRulesRepo) -> None:
    await rules.allow_always("write_file")
    await rules.allow_always("run_command")

    assert await rules.always_allowed() == {"write_file", "run_command"}


async def test_rules_survive_a_restart() -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        await ApprovalRulesRepo(db).allow_always("write_file")

    async with Database(db_path()) as db:
        await db.migrate()
        assert await ApprovalRulesRepo(db).always_allowed() == {"write_file"}


# --- In ToolRegistry.execute -------------------------------------------------


class Counter:
    def __init__(self) -> None:
        self.runs = 0


def work_tool(counter: Counter, *, policy: Policy = "ask", reason: str | None = None) -> FuncTool:
    async def work(params: Any, ctx: ToolContext) -> str:
        counter.runs += 1
        return "done"

    class Work(FuncTool):
        async def approval_reason(self, params: Any, ctx: ToolContext) -> str | None:
            return reason

    tool = Work("work", work)
    tool.default_policy = policy
    return tool


async def execute(
    tool: FuncTool,
    approver: FakeApprover,
    rules: Any,
    config: Config | None = None,
) -> ToolResult:
    config = config or Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
        approval_rules=rules,
    )
    return await registry(tool).execute(ToolCall(id="c", name="work"), ctx)


async def test_always_saves_a_rule_and_the_next_call_runs_unasked(
    rules: ApprovalRulesRepo,
) -> None:
    counter = Counter()
    tool = work_tool(counter)
    approver = FakeApprover("always")

    first = await execute(tool, approver, rules)
    second = await execute(tool, approver, rules)

    assert (text_of(first), text_of(second), counter.runs) == ("done", "done", 2)
    assert len(approver.asked) == 1
    assert await rules.always_allowed() == {"work"}


@pytest.mark.parametrize("decision", ["allow", "deny"])
async def test_other_answers_save_nothing(rules: ApprovalRulesRepo, decision: Decision) -> None:
    approver = FakeApprover(decision)
    tool = work_tool(Counter())

    await execute(tool, approver, rules)
    await execute(tool, approver, rules)

    assert len(approver.asked) == 2
    assert await rules.always_allowed() == set()


async def test_rule_does_not_skip_an_approval_the_tool_demands(rules: ApprovalRulesRepo) -> None:
    await rules.allow_always("work")
    approver = FakeApprover("allow")

    await execute(
        work_tool(Counter(), policy="allow", reason="~/.env may hold secrets"), approver, rules
    )

    assert [ask.reason for ask in approver.asked] == ["~/.env may hold secrets"]


async def test_rule_does_not_lift_a_deny(rules: ApprovalRulesRepo) -> None:
    await rules.allow_always("work")
    counter = Counter()
    config = Config()
    config.permissions.tool_policies["work"] = "deny"

    result = await execute(work_tool(counter), FakeApprover(), rules, config)

    assert (text_of(result), counter.runs) == ("work is turned off in the settings.", 0)


async def test_call_runs_even_if_the_rule_cannot_be_saved(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenRules:
        async def always_allowed(self) -> set[str]:
            return set()

        async def allow_always(self, tool: str) -> object:
            raise OSError("disk is full")

    counter = Counter()

    result = await execute(work_tool(counter), FakeApprover("always"), BrokenRules())

    assert (text_of(result), counter.runs) == ("done", 1)
    assert "Could not save the always-allow rule for work" in caplog.text


async def test_without_rules_a_tool_that_asks_still_asks() -> None:
    approver = FakeApprover("always")

    await execute(work_tool(Counter()), approver, None)
    await execute(work_tool(Counter()), approver, None)

    assert len(approver.asked) == 2


async def test_without_an_approver_the_rule_still_lets_the_call_run(
    rules: ApprovalRulesRepo,
) -> None:
    await rules.allow_always("work")
    counter = Counter()
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id=None,
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approval_rules=rules,
    )

    result = await registry(work_tool(counter)).execute(ToolCall(id="c", name="work"), ctx)

    assert (text_of(result), counter.runs) == ("done", 1)
    assert NEEDS_APPROVAL not in text_of(result)


# --- Over the WebSocket ------------------------------------------------------


def config_asking_for_list_dir() -> Config:
    config = Config()
    config.permissions.tool_policies["list_dir"] = "ask"
    return config


def list_home_script() -> list[list[Any]]:
    call = ToolCall(id="l1", name="list_dir", input={"path": "~"})
    return [[ToolCallEvent(call=call)], [TextDelta(text="Listed.")]]


def connect(client: TestClient) -> Any:
    token = get_state(client.app).token  # type: ignore[arg-type]
    return client.websocket_connect(f"/ws?token={token}")


def turn_events(ws: WebSocketTestSession, *, answer: Decision | None = None) -> list[str]:
    """Send a message, answer an approval if one comes, return the event types."""
    ws.send_json({"type": "message.send", "text": "list my home", "source": "desktop"})
    types = []
    while True:
        event = ws.receive_json()
        types.append(event["type"])
        if event["type"] == "approval.request":
            assert answer is not None, "unexpected approval.request"
            ws.send_json(
                {
                    "type": "approval.respond",
                    "approval_id": event["approval_id"],
                    "decision": answer,
                }
            )
        if event["type"] == "turn.finished":
            return types


@pytest.fixture
def home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.mark.usefixtures("home")
def test_always_is_remembered_across_turns_and_restarts() -> None:
    provider = FakeProvider(list_home_script() * 2)
    with (
        TestClient(create_app(config_asking_for_list_dir(), provider)) as client,
        connect(client) as ws,
    ):
        ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
        ws.receive_json()
        first = turn_events(ws, answer="always")
        second = turn_events(ws)

    provider = FakeProvider(list_home_script())
    with (
        TestClient(create_app(config_asking_for_list_dir(), provider)) as client,
        connect(client) as ws,
    ):
        ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
        ws.receive_json()
        after_restart = turn_events(ws)

    assert "approval.request" in first
    assert "approval.request" not in second
    assert "approval.request" not in after_restart
    assert text_of(provider.calls[1].messages[-1]).startswith("~/")
