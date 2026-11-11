"""Tests of approvals: the manager, the registry asking it, turns and the protocol."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tests.turns import Audience, FuncTool, Harness, RecordingClient, registry, text_of, until
from zero.approvals.manager import Answer, ApprovalAsk, ApprovalManager, Decision
from zero.config import Config
from zero.core.cancellation import CancelToken, TurnCancelled
from zero.core.events import EventBus
from zero.protocol.messages import ApprovalRequest, ApprovalResolved, Envelope
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.base import CANCELLED, DENIED, ToolContext, ToolResult
from zero.tools.permissions import PathGuard

ASK = ApprovalAsk(
    session_id="s1",
    turn_id="t1",
    tool="write_file",
    label="~/a.txt",
    input={"path": "~/a.txt"},
    reason="write_file is set to ask the user first",
)


def manager_with_client() -> tuple[ApprovalManager, RecordingClient]:
    client = RecordingClient()
    return ApprovalManager(EventBus(Audience([client]))), client


async def next_request(client: RecordingClient, count: int = 1) -> ApprovalRequest:
    await until(lambda: len(requests(client.received)) >= count)
    return requests(client.received)[count - 1]


def requests(events: list[Envelope]) -> list[ApprovalRequest]:
    return [e for e in events if isinstance(e, ApprovalRequest)]


def resolutions(events: list[Envelope]) -> list[ApprovalResolved]:
    return [e for e in events if isinstance(e, ApprovalResolved)]


# --- ApprovalManager ---------------------------------------------------------


async def test_request_asks_every_client_and_returns_the_answer() -> None:
    manager, client = manager_with_client()
    waiting = asyncio.create_task(manager.request(ASK, CancelToken()))

    request = await next_request(client)
    assert manager.waiting == [ASK]
    answered = await manager.respond(request.approval_id, "allow", by_client_id="c-2", note="ok")

    assert answered
    assert await waiting == Answer("allow", "c-2", "ok")
    assert request.model_dump(exclude={"approval_id"}) == {
        "type": "approval.request",
        "id": None,
        "reply_to": None,
        "session_id": "s1",
        "turn_id": "t1",
        "tool": "write_file",
        "label": "~/a.txt",
        "input": {"path": "~/a.txt"},
        "reason": "write_file is set to ask the user first",
        "diff": None,
    }
    assert resolutions(client.received) == [
        ApprovalResolved(
            session_id="s1",
            turn_id="t1",
            approval_id=request.approval_id,
            decision="allow",
            by_client_id="c-2",
        )
    ]
    assert manager.waiting == []


async def test_first_answer_wins() -> None:
    manager, client = manager_with_client()
    waiting = asyncio.create_task(manager.request(ASK, CancelToken()))
    request = await next_request(client)

    first = await manager.respond(request.approval_id, "deny", by_client_id="phone")
    second = await manager.respond(request.approval_id, "allow", by_client_id="desktop")

    assert (first, second) == (True, False)
    assert (await waiting).decision == "deny"
    assert [r.by_client_id for r in resolutions(client.received)] == ["phone"]


async def test_answer_to_an_unknown_approval_is_ignored() -> None:
    manager, client = manager_with_client()

    assert not await manager.respond("missing", "allow", by_client_id="c-1")
    assert client.received == []


async def test_cancel_ends_the_wait() -> None:
    manager, client = manager_with_client()
    cancel = CancelToken()
    waiting = asyncio.create_task(manager.request(ASK, cancel))
    request = await next_request(client)

    cancel.cancel()

    with pytest.raises(TurnCancelled):
        await waiting
    assert manager.waiting == []
    assert not await manager.respond(request.approval_id, "allow", by_client_id="c-1")
    assert resolutions(client.received) == []


async def test_cancelled_turn_asks_nothing() -> None:
    manager, client = manager_with_client()
    cancel = CancelToken()
    cancel.cancel()

    with pytest.raises(TurnCancelled):
        await manager.request(ASK, cancel)
    assert client.received == []


async def test_cancel_wins_over_an_answer_at_the_same_moment() -> None:
    manager, client = manager_with_client()
    cancel = CancelToken()
    waiting = asyncio.create_task(manager.request(ASK, cancel))
    request = await next_request(client)

    cancel.cancel()
    await manager.respond(request.approval_id, "allow", by_client_id="c-1")

    with pytest.raises(TurnCancelled):
        await waiting


async def test_clients_hear_the_answer_before_the_turn_goes_on() -> None:
    manager, client = manager_with_client()

    async def ask_then_act() -> None:
        await manager.request(ASK, CancelToken())
        client.received.append(Envelope(type="tool.ran"))

    acting = asyncio.create_task(ask_then_act())
    request = await next_request(client)
    await manager.respond(request.approval_id, "allow", by_client_id="c-1")
    await acting

    assert [e.type for e in client.received] == [
        "approval.request",
        "approval.resolved",
        "tool.ran",
    ]


async def test_approvals_are_independent() -> None:
    manager, client = manager_with_client()
    first = asyncio.create_task(manager.request(ASK, CancelToken()))
    second = asyncio.create_task(manager.request(ASK, CancelToken()))
    request_1 = await next_request(client, 1)
    request_2 = await next_request(client, 2)

    await manager.respond(request_2.approval_id, "deny", by_client_id="c-1")
    await manager.respond(request_1.approval_id, "allow", by_client_id="c-1")

    assert ((await first).decision, (await second).decision) == ("allow", "deny")


# --- ToolRegistry.execute asks -----------------------------------------------


class FakeApprover:
    """Answers every approval at once with ``decision`` and remembers what was asked."""

    def __init__(self, decision: Decision = "allow", note: str | None = None) -> None:
        self.decision: Decision = decision
        self.note = note
        self.asked: list[ApprovalAsk] = []

    async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Answer:
        cancel.raise_if_cancelled()
        self.asked.append(ask)
        return Answer(self.decision, "c-1", self.note)


class Counter:
    def __init__(self) -> None:
        self.runs = 0


def asking_tool(counter: Counter, reason: str | None = None) -> FuncTool:
    async def work(params: Any, ctx: ToolContext) -> str:
        counter.runs += 1
        return "done"

    class Asking(FuncTool):
        default_policy = "ask"

        def label(self, params: Any) -> str:
            return "work on things"

        async def approval_reason(self, params: Any, ctx: ToolContext) -> str | None:
            return reason

    return Asking("work", work)


async def execute(
    tool: FuncTool, approver: FakeApprover | None, cancel: CancelToken | None = None
) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=cancel or CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
    )
    call = ToolCall(id="call-1", name="work", input={"x": 1})
    return await registry(tool).execute(call, ctx)


@pytest.mark.parametrize("decision", ["allow", "always"])
async def test_allowed_call_runs(decision: Decision) -> None:
    counter = Counter()
    approver = FakeApprover(decision)

    result = await execute(asking_tool(counter), approver)

    assert (text_of(result), result.is_error, counter.runs) == ("done", False, 1)
    assert approver.asked == [
        ApprovalAsk(
            session_id="s1",
            turn_id="t1",
            tool="work",
            label="work on things",
            input={"x": 1},
            reason="work is set to ask the user first",
        )
    ]


async def test_denied_call_does_not_run() -> None:
    counter = Counter()

    result = await execute(asking_tool(counter), FakeApprover("deny"))

    assert (text_of(result), result.is_error, result.denied, counter.runs) == (
        DENIED,
        True,
        True,
        0,
    )


async def test_the_users_note_reaches_the_model() -> None:
    result = await execute(asking_tool(Counter()), FakeApprover("deny", note="not today"))

    assert text_of(result) == f"{DENIED} Their note: not today"


async def test_tool_reason_is_shown_to_the_user() -> None:
    approver = FakeApprover()

    await execute(asking_tool(Counter(), reason="~/.env may hold secrets"), approver)

    assert approver.asked[0].reason == "~/.env may hold secrets"


async def test_allowed_tool_asks_nobody() -> None:
    tool = asking_tool(Counter())
    tool.default_policy = "allow"
    approver = FakeApprover()

    await execute(tool, approver)

    assert approver.asked == []


async def test_cancel_while_asking_gives_a_cancelled_result() -> None:
    cancel = CancelToken()
    cancel.cancel()
    counter = Counter()

    result = await execute(asking_tool(counter), FakeApprover(), cancel)

    assert (text_of(result), result.cancelled, counter.runs) == (CANCELLED, True, 0)


# --- In a turn ---------------------------------------------------------------


def work_call() -> list[list[Any]]:
    return [[ToolCallEvent(call=ToolCall(id="w1", name="work"))], [TextDelta(text="Done.")]]


async def run_and_answer(harness: Harness, decision: Decision) -> tuple[FakeProvider, Counter]:
    counter = Counter()
    provider = FakeProvider(work_call())
    manager = ApprovalManager(harness.bus)
    turn = harness.turn(provider, registry(asking_tool(counter)), approvals=manager)
    running = asyncio.create_task(turn.run())

    request = await next_request(harness.client)
    await manager.respond(request.approval_id, decision, by_client_id="c-1")
    await running
    return provider, counter


async def test_turn_waits_for_approval_then_runs_the_tool(harness: Harness) -> None:
    provider, counter = await run_and_answer(harness, "allow")

    assert harness.types == [
        "turn.started",
        "tool.started",
        "approval.request",
        "approval.resolved",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    assert counter.runs == 1
    assert harness.tool_events()[1].status == "ok"  # type: ignore[union-attr]
    assert text_of(provider.calls[1].messages[-1]) == "done"


async def test_denied_call_is_reported_and_the_turn_goes_on(harness: Harness) -> None:
    provider, counter = await run_and_answer(harness, "deny")

    assert counter.runs == 0
    assert harness.tool_events()[1].status == "denied"  # type: ignore[union-attr]
    result = provider.calls[1].messages[-1]
    assert (result.is_error, text_of(result)) == (True, DENIED)
    assert (harness.finished.status, harness.finished.text) == ("ok", "Done.")


async def test_cancel_while_waiting_ends_the_turn(harness: Harness) -> None:
    counter = Counter()
    provider = FakeProvider(work_call())
    manager = ApprovalManager(harness.bus)
    turn = harness.turn(provider, registry(asking_tool(counter)), approvals=manager)
    running = asyncio.create_task(turn.run())
    await next_request(harness.client)

    turn.cancel.cancel()
    await running

    assert counter.runs == 0
    assert harness.tool_events()[1].status == "cancelled"  # type: ignore[union-attr]
    assert harness.finished.status == "cancelled"
    assert manager.waiting == []
    assert len(provider.calls) == 1
    tool_message = (await harness.stored())[2]
    assert (tool_message.role, text_of(tool_message)) == ("tool", CANCELLED)


# --- Over the WebSocket ------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    (home / ".env").write_text("API_KEY=sk-123\n")
    monkeypatch.setenv("HOME", str(home))
    return home


def read_env_script() -> list[list[Any]]:
    call = ToolCall(id="r1", name="read_file", input={"path": "~/.env"})
    return [[ToolCallEvent(call=call)], [TextDelta(text="Read it.")]]


@pytest.fixture
def two_clients(
    home: Path,
) -> Iterator[tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession]]:
    provider = FakeProvider(read_env_script())
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with (
            client.websocket_connect(f"/ws?token={token}") as desktop,
            client.websocket_connect(f"/ws?token={token}") as phone,
        ):
            for ws, name in ((desktop, "desktop"), (phone, "phone")):
                ws.send_json(
                    {"type": "hello", "protocol": 1, "client": {"app": name, "client_id": name}}
                )
                assert ws.receive_json()["type"] == "hello.ok"
            yield provider, desktop, phone


def read_until(ws: WebSocketTestSession, event_type: str) -> list[dict[str, Any]]:
    events = [ws.receive_json()]
    while events[-1]["type"] != event_type:
        events.append(ws.receive_json())
    return events


def test_secret_file_is_read_after_the_user_allows_it(
    two_clients: tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession],
) -> None:
    provider, desktop, phone = two_clients
    desktop.send_json({"type": "message.send", "text": "read my env", "source": "desktop"})
    request = read_until(desktop, "approval.request")[-1]
    assert read_until(phone, "approval.request")[-1] == request

    phone.send_json(
        {"type": "approval.respond", "approval_id": request["approval_id"], "decision": "allow"}
    )
    mine = read_until(desktop, "turn.finished")
    theirs = read_until(phone, "turn.finished")

    assert (request["tool"], request["label"], request["reason"]) == (
        "read_file",
        "~/.env",
        "~/.env may hold secrets",
    )
    resolved = [e for e in mine if e["type"] == "approval.resolved"]
    assert [(e["decision"], e["by_client_id"]) for e in resolved] == [("allow", "phone")]
    assert [e for e in theirs if e["type"] == "approval.resolved"] == resolved
    assert [e["status"] for e in mine if e["type"] == "tool.finished"] == ["ok"]
    assert text_of(provider.calls[1].messages[-1]) == "     1\tAPI_KEY=sk-123"


def test_secret_file_stays_unread_when_the_user_denies(
    two_clients: tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession],
) -> None:
    provider, desktop, phone = two_clients
    desktop.send_json({"type": "message.send", "text": "read my env", "source": "desktop"})
    request = read_until(desktop, "approval.request")[-1]

    desktop.send_json(
        {
            "type": "approval.respond",
            "approval_id": request["approval_id"],
            "decision": "deny",
            "note": "keep out",
        }
    )
    events = read_until(desktop, "turn.finished")
    desktop.send_json(
        {"type": "approval.respond", "approval_id": request["approval_id"], "decision": "allow"}
    )
    desktop.send_json({"type": "ping", "id": "p1"})

    assert read_until(desktop, "pong")[-1] == {"type": "pong", "reply_to": "p1"}
    assert [e["status"] for e in events if e["type"] == "tool.finished"] == ["denied"]
    result = provider.calls[1].messages[-1]
    assert text_of(result) == f"{DENIED} Their note: keep out"
    assert "sk-123" not in text_of(result)


def test_cancelling_the_turn_drops_the_approval(
    two_clients: tuple[FakeProvider, WebSocketTestSession, WebSocketTestSession],
) -> None:
    provider, desktop, _ = two_clients
    desktop.send_json({"type": "message.send", "text": "read my env", "source": "desktop"})
    request = read_until(desktop, "approval.request")[-1]

    desktop.send_json({"type": "turn.cancel", "turn_id": request["turn_id"]})
    events = read_until(desktop, "turn.finished")

    assert [e["status"] for e in events if e["type"] == "tool.finished"] == ["cancelled"]
    assert events[-1]["status"] == "cancelled"
    assert len(provider.calls) == 1
