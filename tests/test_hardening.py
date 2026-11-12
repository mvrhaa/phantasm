"""Tests of fixes found by a review of the whole code."""

import asyncio
import io
import logging
from pathlib import Path

import pytest

from tests.turns import FuncTool, Harness, NoInput, registry, until
from zero import clock
from zero.approvals.manager import ApprovalAsk, ApprovalManager
from zero.config import Config, ConfigError, load_config
from zero.core.attachments import images_to_files, remove_files
from zero.core.cancellation import CancelToken
from zero.core.context import NO_ANSWER, to_history
from zero.core.prompting import HISTORY_FETCH_LIMIT, PromptBuilder, _first_user
from zero.core.turn import IMAGE_UNSEEN, _keep_latest_images
from zero.logging_setup import RedactingFormatter, RedactTokenFilter
from zero.paths import db_path, files_dir
from zero.protocol.messages import FilePart, ImagePart, TextPart
from zero.providers.base import Message, TextDelta, ToolCall, ToolCallEvent, writes_prose
from zero.providers.chat_format import wire_messages
from zero.providers.fake import FakeProvider
from zero.scheduler.rules import parse_rule, zone_named
from zero.scheduler.runner import MIN_PAUSE_S, SchedulerRunner
from zero.store.db import Database
from zero.store.messages import StoredMessage
from zero.store.schedules import SchedulesRepo
from zero.store.sessions import SessionsRepo
from zero.store.summaries import SummariesRepo
from zero.store.tool_runs import ToolRunsRepo
from zero.tools.base import Tool, ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.builtin.edit import EditFile, EditFileInput
from zero.tools.builtin.shell import KEEP_BYTES, _Output
from zero.tools.permissions import PathGuard


def context(folder: Path) -> ToolContext:
    return ToolContext(
        session_id="s",
        turn_id="t",
        client_id=None,
        cancel=CancelToken(),
        config=Config(),
        paths=PathGuard([str(folder)], []),
    )


def text_of(result: ToolResult) -> str:
    return "".join(part.text for part in result.content if isinstance(part, TextPart))


def stored(number: int, role: str, text: str = "", status: str | None = None) -> StoredMessage:
    return StoredMessage(
        id=str(number),
        session_id="s",
        role=role,  # type: ignore[arg-type]
        content=[TextPart(text=text)] if text else [],
        status=status,  # type: ignore[arg-type]
        tool_call_id="c" if role == "tool" else None,
        created_at=clock.now(),
    )


async def test_edit_file_refuses_text_that_is_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "settings.ini"
    original = "name = Привет\nport = 80\n".encode("cp1251")
    path.write_bytes(original)

    params = EditFileInput(path=str(path), old="port = 80", new="port = 8080")
    result = await EditFile().run(params, context(tmp_path))

    assert result.is_error
    assert "not UTF-8" in text_of(result)
    assert path.read_bytes() == original


async def test_search_files_asks_before_a_file_that_may_hold_secrets(tmp_path: Path) -> None:
    secret = tmp_path / ".env.local"
    secret.write_text("DB_PASSWORD=hunter2hunter2\n", encoding="utf-8")
    call = ToolCall(id="1", name="search_files", input={"pattern": ".", "path": str(secret)})

    result = await builtin_registry().execute(call, context(tmp_path))

    assert result.is_error
    assert "may hold secrets" in text_of(result)
    assert "hunter2" not in text_of(result)


async def test_search_files_skips_secret_files_in_a_folder(tmp_path: Path) -> None:
    (tmp_path / "id_rsa").write_text("PRIVATE KEY abc\n", encoding="utf-8")
    (tmp_path / "server.pem").write_text("PRIVATE KEY def\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("the KEY is under the mat\n", encoding="utf-8")
    call = ToolCall(id="1", name="search_files", input={"pattern": "KEY", "path": str(tmp_path)})

    result = await builtin_registry().execute(call, context(tmp_path))

    assert text_of(result) == "notes.txt:1: the KEY is under the mat"


def test_a_cut_off_history_starts_with_a_user_message() -> None:
    page = [stored(1, "tool", "result"), stored(2, "user", "hi"), stored(3, "assistant", "hello")]

    assert _first_user(page) == 1
    assert _first_user([stored(1, "tool", "result")]) == 1


def test_an_empty_answer_of_a_broken_turn_is_not_sent_empty() -> None:
    history = to_history(
        [
            stored(1, "user", "hi"),
            stored(2, "assistant", status="error"),
            stored(3, "user", "again"),
            stored(4, "assistant", status="cancelled"),
            stored(5, "user", "once more"),
        ]
    )

    wire = wire_messages(history)

    assert wire[1] == {"role": "assistant", "content": NO_ANSWER["error"]}
    assert wire[3] == {"role": "assistant", "content": NO_ANSWER["cancelled"]}


async def test_a_turn_that_runs_too_long_ends_with_an_error(harness: Harness) -> None:
    turn = harness.turn(FakeProvider(infinite=True))
    turn._config.agent.turn_timeout_s = 1
    asyncio.get_running_loop().call_later(0.05, turn._time_out)

    await turn.run()

    assert harness.finished.status == "error"
    assert "ran longer than 1 s" in (harness.finished.error or "")
    assert (await harness.stored())[-1].status == "error"


async def test_a_turn_survives_the_loss_of_its_session(harness: Harness) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(_: object, __: ToolContext) -> str:
        started.set()
        await release.wait()
        return "done"

    provider = FakeProvider([[ToolCallEvent(call=ToolCall(id="c1", name="slow"))]])
    turn = harness.turn(provider, registry(FuncTool("slow", slow)))
    task = asyncio.create_task(turn.run())
    await until(started.is_set)
    await harness.sessions.delete(harness.session_id)
    release.set()

    await task

    assert turn.done
    assert harness.finished.status == "error"


async def test_a_model_without_sight_gets_a_note_instead_of_a_tool_image(
    harness: Harness,
) -> None:
    class Blind(FakeProvider):
        def supports_images(self, model: str) -> bool:
            return False

    class Picture(Tool[NoInput]):
        name = "picture"
        description = "Returns an image."
        input_model = NoInput
        default_policy = "allow"

        async def run(self, params: NoInput, ctx: ToolContext) -> ToolResult:
            return ToolResult(content=[ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")])

    provider = Blind(
        [[ToolCallEvent(call=ToolCall(id="c1", name="picture"))], [TextDelta(text="ok")]]
    )
    await harness.run(provider, registry(Picture()))

    sent = provider.calls[1].messages[-1]
    assert sent.content == [TextPart(text=IMAGE_UNSEEN)]
    kept = (await harness.stored())[2].content[0]
    assert isinstance(kept, FilePart) and kept.mime == "image/png"
    assert Path(kept.path).read_bytes().startswith(b"\x89PNG")
    assert Path(kept.path).parent.parent == files_dir()


def test_secrets_in_a_traceback_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "zero.logging_setup.mask_secrets", lambda t: t.replace("s3cretvalue", "***")
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    handler.addFilter(RedactTokenFilter())
    log = logging.getLogger("test.hardening")
    log.addHandler(handler)
    log.propagate = False
    try:
        try:
            raise RuntimeError("bad header: Bearer s3cretvalue at ws://x/ws?token=abc123")
        except RuntimeError:
            log.exception("request failed")
    finally:
        log.removeHandler(handler)

    assert "s3cretvalue" not in stream.getvalue()
    assert "abc123" not in stream.getvalue()
    assert "RuntimeError" in stream.getvalue()


def test_output_that_fits_is_decoded_in_one_piece() -> None:
    output = _Output()
    text = "a" + "я" * (KEEP_BYTES // 2 + 10)
    output.add(text.encode())

    assert output.text() == text


async def test_deleting_a_session_removes_its_tool_runs() -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        sessions = SessionsRepo(db)
        session = await sessions.create()
        await db.conn.execute(
            "INSERT INTO tool_runs (id, call_id, session_id, turn_id, tool, status, created_at) "
            "VALUES ('r1', 'c1', ?, 't1', 'read_file', 'ok', ?)",
            (session.id, clock.to_iso(clock.now())),
        )
        await db.conn.commit()
        assert len(await ToolRunsRepo(db).for_turns(session.id, ["t1"])) == 1

        await sessions.delete(session.id)

        assert await ToolRunsRepo(db).for_turns(session.id, ["t1"]) == []


def test_a_config_that_is_not_text_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_config(path)


def test_deleting_a_session_stops_its_turn_first() -> None:
    from tests.test_e2e_agents import send, service

    with service(FakeProvider(infinite=True)) as (state, ws):
        send(ws, "talk forever")
        session_id = ""
        while not session_id:
            event = ws.receive_json()
            if event["type"] == "turn.started":
                session_id = event["session_id"]
        ws.send_json({"type": "session.delete", "id": "d", "session_id": session_id})

        seen = []
        while not seen or seen[-1]["type"] != "session.deleted":
            seen.append(ws.receive_json())

        finished = [e for e in seen if e["type"] == "turn.finished"]
        assert [e["status"] for e in finished] == ["cancelled"]
        assert seen[-1]["reply_to"] == "d"
        assert state.turns.active == []


async def test_files_of_a_session_are_found_and_removed(harness: Harness) -> None:
    kept = await images_to_files([ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")])
    outside = files_dir().parent / "notes.txt"
    outside.write_text("mine", encoding="utf-8")
    parts = [*kept, FilePart(name="notes.txt", path=str(outside))]
    await harness.messages.append(harness.session_id, "tool", parts, tool_call_id="c1")

    paths = await harness.messages.file_paths(harness.session_id)

    assert sorted(paths) == sorted(part.path for part in parts)  # type: ignore[union-attr]
    assert await remove_files(paths) == 1
    assert not Path(kept[0].path).parent.exists()  # type: ignore[union-attr]
    assert outside.exists()


def test_only_the_latest_tool_images_stay_in_the_prompt() -> None:
    image = ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")
    prompt = [
        Message(role="tool", content=[image, TextPart(text=str(n))], tool_call_id="c")
        for n in range(4)
    ]

    _keep_latest_images(prompt, keep=2)

    assert [type(m.content[0]) for m in prompt] == [TextPart, TextPart, ImagePart, ImagePart]
    assert [m.content[1] for m in prompt] == [TextPart(text=str(n)) for n in range(4)]


async def test_messages_before_the_window_are_summarized(harness: Harness) -> None:
    for number in range(HISTORY_FETCH_LIMIT // 2 + 2):
        await harness.messages.append(harness.session_id, "user", [TextPart(text=f"q{number}")])
        await harness.messages.append(
            harness.session_id, "assistant", [TextPart(text=f"a{number}")]
        )
    async with Database(db_path()) as db:
        summaries = SummariesRepo(db)
        builder = PromptBuilder(
            session_id=harness.session_id,
            client=None,
            source="desktop",
            config=Config(),
            provider=FakeProvider(),
            messages=harness.messages,
            events=harness.bus,
            summaries=summaries,
        )

        prompt = await builder.build()
        summary = await summaries.get(harness.session_id)

    assert prompt[1].role == "user"
    assert summary is not None and "User: q0" in summary.text
    assert "# Earlier in this conversation" in text_of(prompt[0])


async def test_a_late_client_can_be_given_the_open_approvals(harness: Harness) -> None:
    manager = ApprovalManager(harness.bus)
    ask = ApprovalAsk(session_id="s", turn_id="t", tool="run_command", label="$ ls")
    waiting = asyncio.create_task(manager.request(ask, CancelToken()))
    await until(lambda: bool(manager.waiting))

    (request,) = manager.open_requests
    assert request.tool == "run_command" and request.label == "$ ls"
    assert await manager.respond(request.approval_id, "allow", by_client_id="late")
    assert (await waiting).by_client_id == "late"
    assert manager.open_requests == []


async def test_the_scheduler_wakes_up_for_the_next_schedule(frozen_clock: object) -> None:
    async with Database(db_path()) as db:
        await db.migrate()
        schedules = SchedulesRepo(db)

        async def start(_: object) -> None:
            return None

        runner = SchedulerRunner(schedules, start, tick_s=30.0)
        assert await runner._pause() == 30.0

        zone = zone_named("UTC")
        await schedules.add("ping", parse_rule("in 1m", clock.now(), zone), "in 1m", "UTC")
        frozen_clock.advance(50)  # type: ignore[attr-defined]
        assert await runner._pause() == 10.0

        frozen_clock.advance(20)  # type: ignore[attr-defined]
        assert await runner._pause() == MIN_PAUSE_S


def test_the_fake_provider_says_it_cannot_write_titles() -> None:
    assert writes_prose(FakeProvider()) is False
    assert writes_prose(object()) is True
