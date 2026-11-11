"""Tests of send_file and of the records of tool runs: events, storage and history."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.turns import FakeApprover, FuncTool, Harness, registry, text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.paths import db_path, files_dir
from zero.protocol.messages import (
    Character,
    ClientInfo,
    FilePart,
    FileSent,
    ImagePart,
    TextPart,
    ToolFinished,
    ToolRun,
    ToolStarted,
)
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.store.db import Database
from zero.store.tool_runs import ToolRunsRepo
from zero.tools.base import ToolContext, ToolResult, output_preview, text_result
from zero.tools.builtin import builtin_registry, files
from zero.tools.permissions import PathGuard


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def call(name: str, call_id: str = "c1", **tool_input: Any) -> ToolCallEvent:
    return ToolCallEvent(call=ToolCall(id=call_id, name=name, input=tool_input))


# --- output_preview and labels -----------------------------------------------


@pytest.mark.parametrize(
    ("content", "preview"),
    [
        ([TextPart(text="  line one\n\n  line two  ")], "line one line two"),
        ([TextPart(text="a"), TextPart(text="b")], "a b"),
        ([ImagePart(mime="image/png", data_b64="AA==")], "[image]"),
        ([FilePart(name="r.pdf", path="/x/r.pdf")], "[file r.pdf]"),
        ([TextPart(text="Sent."), FilePart(name="r.pdf", path="/x")], "Sent."),
        ([], ""),
    ],
)
def test_output_preview(content: list[Any], preview: str) -> None:
    assert output_preview(ToolResult(content=content)) == preview


def test_long_output_preview_is_cut() -> None:
    preview = output_preview(text_result("x" * 1000), limit=20)

    assert preview == "x" * 19 + "…"


def test_registry_labels_a_call_with_the_tools_label() -> None:
    tools = builtin_registry()

    assert (
        tools.label(ToolCall(id="c", name="run_command", input={"command": "ls -la"})) == "$ ls -la"
    )
    assert tools.label(ToolCall(id="c", name="run_command", input={})) == "run_command"
    assert tools.label(ToolCall(id="c", name="fly")) == "fly"


# --- send_file ---------------------------------------------------------------


async def send(path: str | Path, approver: FakeApprover | None = None) -> ToolResult:
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
        approvals=approver,
    )
    return await builtin_registry().execute(
        ToolCall(id="s", name="send_file", input={"path": str(path)}), ctx
    )


async def test_send_file_gives_the_user_a_copy(home: Path) -> None:
    (home / "report.pdf").write_bytes(b"%PDF report")

    result = await send("~/report.pdf")

    text, file = result.content
    assert isinstance(file, FilePart)
    copy = Path(file.path)
    assert copy.parent.parent == files_dir()
    assert (file.name, copy.read_bytes()) == ("report.pdf", b"%PDF report")
    assert text == TextPart(text="Sent ~/report.pdf to the user as report.pdf (11 B).")
    assert (result.summary, result.is_error) == ("report.pdf, 11 B", False)
    assert (home / "report.pdf").exists()


async def test_each_send_is_its_own_copy(home: Path) -> None:
    (home / "a.txt").write_text("one")

    first = await send("~/a.txt")
    (home / "a.txt").write_text("two")
    second = await send("~/a.txt")

    paths = [Path(r.content[1].path) for r in (first, second)]  # type: ignore[union-attr]
    assert paths[0] != paths[1]
    assert [p.read_text() for p in paths] == ["one", "two"]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("~/missing.pdf", "~/missing.pdf does not exist."),
        ("~", "~ is not a regular file; only files can be sent."),
        ("relative.txt", "'relative.txt' is a relative path"),
    ],
)
async def test_what_cannot_be_sent_is_refused(home: Path, path: str, message: str) -> None:
    result = await send(path)

    assert result.is_error
    assert text_of(result).startswith(message)
    assert not files_dir().exists() or not any(files_dir().iterdir())


async def test_file_outside_the_sandbox_is_refused_without_asking(home: Path) -> None:
    (home.parent / "secret.txt").write_text("x")
    config_asks = FakeApprover("allow")

    result = await send(home.parent / "secret.txt", config_asks)

    assert "outside the folders tools may use" in text_of(result)
    assert config_asks.asked == []


async def test_too_big_file_is_refused(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "MAX_SEND_BYTES", 10)
    (home / "big.bin").write_bytes(b"x" * 11)

    result = await send("~/big.bin")

    assert text_of(result) == "~/big.bin is 11 B; files over 10 B cannot be sent."


# --- ToolRunsRepo ------------------------------------------------------------


def run(call_id: str, turn_id: str, created: Any, **fields: Any) -> ToolRun:
    return ToolRun(
        call_id=call_id, turn_id=turn_id, tool="add", status="ok", created_at=created, **fields
    )


@pytest.fixture
async def repo() -> AsyncIterator[ToolRunsRepo]:
    async with Database(db_path()) as db:
        await db.migrate()
        yield ToolRunsRepo(db)


async def test_runs_are_read_back_by_turn_in_order(repo: ToolRunsRepo, frozen_clock: Any) -> None:
    first = run("c1", "t1", frozen_clock(), label="add 1", input={"a": 1}, duration_ms=5)
    frozen_clock.advance(1)
    second = run("c2", "t2", frozen_clock())
    other_turn = run("c3", "t3", frozen_clock())
    await repo.add("s1", first)
    await repo.add("s1", second)
    await repo.add("s1", other_turn)
    await repo.add("s2", run("c4", "t1", frozen_clock()))

    assert await repo.for_turns("s1", ["t1", "t2"]) == [first, second]
    assert await repo.for_turns("s1", []) == []


async def test_a_repeated_call_id_is_recorded_twice(repo: ToolRunsRepo, frozen_clock: Any) -> None:
    await repo.add("s1", run("call-1", "t1", frozen_clock()))
    await repo.add("s1", run("call-1", "t2", frozen_clock()))

    assert [r.turn_id for r in await repo.for_turns("s1", ["t1", "t2"])] == ["t1", "t2"]


# --- In a turn ---------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.runs: list[tuple[str, ToolRun]] = []

    async def add(self, session_id: str, run: ToolRun) -> None:
        self.runs.append((session_id, run))


async def test_events_carry_the_label_summary_preview_and_duration(
    harness: Harness, frozen_clock: Any
) -> None:
    async def slow_add(params: Any, ctx: ToolContext) -> str:
        frozen_clock.advance(1.5)
        return "the answer is 5"

    class Adder(FuncTool):
        def label(self, params: Any) -> str:
            return "2 + 3"

    recorder = Recorder()
    provider = FakeProvider([[call("add")], [TextDelta(text="5")]])

    turn = await harness.run(provider, registry(Adder("add", slow_add)), tool_runs=recorder)

    started, finished = harness.tool_events()
    assert isinstance(started, ToolStarted) and started.label == "2 + 3"
    assert isinstance(finished, ToolFinished)
    assert (finished.status, finished.output_preview, finished.duration_ms) == (
        "ok",
        "the answer is 5",
        1500,
    )
    ((session_id, record),) = recorder.runs
    assert session_id == harness.session_id
    assert record == ToolRun(
        call_id="c1",
        turn_id=turn.id,
        tool="add",
        label="2 + 3",
        input={},
        status="ok",
        summary="",
        output_preview="the answer is 5",
        duration_ms=1500,
        created_at=record.created_at,
    )


async def test_denied_call_is_recorded_as_denied(harness: Harness) -> None:
    async def never(params: Any, ctx: ToolContext) -> str:
        return "ran"

    tool = FuncTool("risky", never)
    tool.default_policy = "ask"
    recorder = Recorder()
    provider = FakeProvider([[call("risky")], [TextDelta(text="ok")]])

    await harness.run(provider, registry(tool), tool_runs=recorder, approvals=FakeApprover("deny"))

    assert [r.status for _, r in recorder.runs] == ["denied"]


async def test_set_expression_is_not_recorded(harness: Harness) -> None:
    character = Character(name="Zero", poses=["idle"], moods=["happy"])
    client = ClientInfo(app="desktop", client_id="c-1", character=character)
    recorder = Recorder()
    provider = FakeProvider(
        [[TextDelta(text="Hi!"), call("set_expression", pose="idle", mood="happy")]]
    )

    await harness.run(provider, client=client, tool_runs=recorder)

    assert recorder.runs == []
    started = harness.tool_events()[0]
    assert (started.hidden, started.label) == (True, "set_expression")  # type: ignore[union-attr]


async def test_a_failing_recorder_does_not_break_the_turn(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        async def add(self, session_id: str, run: ToolRun) -> None:
            raise OSError("disk is full")

    async def work(params: Any, ctx: ToolContext) -> str:
        return "done"

    provider = FakeProvider([[call("work")], [TextDelta(text="ok")]])

    await harness.run(provider, registry(FuncTool("work", work)), tool_runs=Broken())

    assert harness.finished.status == "ok"
    assert "Could not record tool run c1" in caplog.text


async def test_file_parts_reach_clients_as_file_events(harness: Harness, home: Path) -> None:
    (home / "plan.md").write_text("# Plan")
    provider = FakeProvider([[call("send_file", path="~/plan.md")], [TextDelta(text="Sent.")]])

    await harness.run(provider, builtin_registry())

    assert harness.types == [
        "turn.started",
        "tool.started",
        "file",
        "tool.finished",
        "text.delta",
        "turn.finished",
    ]
    event = harness.client.received[2]
    assert isinstance(event, FileSent)
    assert (event.call_id, event.file.name) == ("c1", "plan.md")
    assert Path(event.file.path).read_text() == "# Plan"
    finished = harness.client.received[3]
    assert finished.summary == "plan.md, 6 B"  # type: ignore[attr-defined]


# --- Over the WebSocket ------------------------------------------------------


def read_until(ws: Any, event_type: str) -> dict[str, Any]:
    """The first event of ``event_type``; others (such as session.updated) are skipped."""
    while (event := ws.receive_json())["type"] != event_type:
        pass
    return event


def test_history_returns_the_tool_runs_of_its_turns(home: Path) -> None:
    (home / "plan.md").write_text("# Plan")
    script = [[call("send_file", path="~/plan.md")], [TextDelta(text="Sent.")]] * 2
    with TestClient(create_app(Config(), FakeProvider(script))) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            message = {"type": "message.send", "text": "send it", "source": "desktop"}
            ws.send_json(message)
            session_id = read_until(ws, "session.created")["session"]["id"]
            read_until(ws, "turn.finished")
            ws.send_json({**message, "session_id": session_id})
            read_until(ws, "turn.finished")

            ws.send_json({"type": "history.get", "session_id": session_id})
            with_tools = read_until(ws, "history.ok")
            ws.send_json({"type": "history.get", "session_id": session_id, "include_tools": False})
            without_tools = read_until(ws, "history.ok")
            ws.send_json({"type": "history.get", "session_id": session_id, "limit": 3})
            last_page = read_until(ws, "history.ok")

    runs = with_tools["tool_runs"]
    assert [(r["tool"], r["label"], r["status"], r["summary"]) for r in runs] == [
        ("send_file", "~/plan.md", "ok", "plan.md, 6 B"),
        ("send_file", "~/plan.md", "ok", "plan.md, 6 B"),
    ]
    assert runs[0]["call_id"] == runs[1]["call_id"] == "c1"
    assert runs[0]["turn_id"] != runs[1]["turn_id"]
    assert without_tools["tool_runs"] == []
    assert [m["role"] for m in without_tools["messages"]].count("tool") == 0
    assert len(last_page["tool_runs"]) == 1  # only the turn on this page
