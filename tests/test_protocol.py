"""Tests for the protocol models."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from zero.protocol.errors import PROTOCOL_VERSION, ErrorCode
from zero.protocol.messages import (
    ActionCall,
    ActionDecl,
    ActionResult,
    ApprovalRequest,
    ApprovalResolved,
    ApprovalRespond,
    Attachment,
    ClientMessage,
    ContentPart,
    Envelope,
    Error,
    Expression,
    ExpressionState,
    FilePart,
    Hello,
    HelloOk,
    HistoryGet,
    HistoryMessage,
    HistoryOk,
    ImagePart,
    MessageSend,
    Ping,
    Pong,
    ServerEvent,
    SessionCreate,
    SessionCreated,
    SessionDelete,
    SessionDeleted,
    SessionInfo,
    SessionList,
    SessionListOk,
    SessionRename,
    SessionUpdated,
    TextDelta,
    TextPart,
    ToolFinished,
    ToolRun,
    ToolStarted,
    TurnCancel,
    TurnFinished,
    TurnStarted,
    Usage,
)

content_part = TypeAdapter(ContentPart)
client_message = TypeAdapter(ClientMessage)
server_event = TypeAdapter(ServerEvent)

SPEC_HELLO = {
    "type": "hello",
    "id": "7f9c",
    "protocol": 1,
    "client": {
        "app": "desktop",
        "client_id": "c-1",
        "display_name": "Desktop on MacBook",
        "character": {
            "name": "Zero",
            "persona": "Markdown persona text",
            "poses": ["idle", "sit", "wave"],
            "moods": ["neutral", "happy", "sad", "angry", "surprised"],
        },
        "environment": "macOS desktop app; replies show in a chat window that renders Markdown",
        "actions": [
            {
                "name": "screenshot",
                "description": "Capture the user's screen",
                "input_schema": {"type": "object", "properties": {"display": {"type": "integer"}}},
                "policy": "allow",
                "timeout_s": 15,
            }
        ],
    },
}


def test_text_part_is_parsed() -> None:
    part = content_part.validate_json('{ "type": "text", "text": "hello" }')

    assert part == TextPart(text="hello")


def test_image_part_is_parsed() -> None:
    part = content_part.validate_json(
        '{ "type": "image", "mime": "image/png", "data_b64": "aGVsbG8=" }'
    )

    assert part == ImagePart(mime="image/png", data_b64="aGVsbG8=")


def test_file_part_is_parsed() -> None:
    part = content_part.validate_json(
        '{ "type": "file", "name": "report.pdf", "path": "/tmp/report.pdf" }'
    )

    assert part == FilePart(name="report.pdf", path="/tmp/report.pdf")


@pytest.mark.parametrize(
    "part",
    [
        TextPart(text="hello"),
        ImagePart(mime="image/png", data_b64="aGVsbG8="),
        FilePart(name="report.pdf", path="/tmp/report.pdf"),
    ],
)
def test_content_part_survives_a_json_round_trip(part: BaseModel) -> None:
    dumped = part.model_dump_json()

    assert f'"type":"{part.type}"' in dumped
    assert content_part.validate_json(dumped) == part


@pytest.mark.parametrize(
    "data",
    [
        {"type": "video", "url": "http://example.com"},
        {"text": "no type"},
        {"type": "text"},
        {"type": "image", "mime": "image/png"},
        {"type": "file", "name": "report.pdf"},
        {"type": "text", "text": 42},
        "just a string",
    ],
    ids=[
        "unknown-type",
        "missing-type",
        "text-without-text",
        "image-without-data",
        "file-without-path",
        "text-not-a-string",
        "not-an-object",
    ],
)
def test_invalid_content_part_is_rejected(data: Any) -> None:
    with pytest.raises(ValidationError):
        content_part.validate_python(data)


def test_content_parts_work_as_a_list_field() -> None:
    class Holder(BaseModel):
        content: list[ContentPart]

    holder = Holder.model_validate(
        {
            "content": [
                {"type": "text", "text": "see the picture"},
                {"type": "image", "mime": "image/jpeg", "data_b64": "AAAA"},
                {"type": "file", "name": "a.txt", "path": "/tmp/a.txt"},
            ]
        }
    )

    assert [type(part) for part in holder.content] == [TextPart, ImagePart, FilePart]


def test_envelope_requires_only_type() -> None:
    envelope = Envelope.model_validate({"type": "ping"})

    assert envelope.type == "ping"
    assert envelope.id is None
    assert envelope.reply_to is None


def test_envelope_carries_id_and_reply_to() -> None:
    envelope = Envelope.model_validate({"type": "pong", "id": "a1", "reply_to": "b2"})

    assert envelope.id == "a1"
    assert envelope.reply_to == "b2"


def test_envelope_without_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate({"id": "a1"})


def test_error_codes_match_the_spec() -> None:
    assert {code.value for code in ErrorCode} == {
        "unauthorized",
        "protocol_mismatch",
        "unknown_type",
        "invalid_message",
        "session_not_found",
        "turn_busy",
        "provider_error",
        "provider_not_configured",
        "tool_error",
        "action_timeout",
        "client_gone",
        "internal",
    }


def test_error_code_is_a_plain_string_in_json() -> None:
    class Holder(BaseModel):
        code: ErrorCode

    assert ErrorCode.TURN_BUSY == "turn_busy"
    assert Holder(code=ErrorCode.TURN_BUSY).model_dump_json() == '{"code":"turn_busy"}'
    assert Holder.model_validate({"code": "turn_busy"}).code is ErrorCode.TURN_BUSY


def test_protocol_version_is_1() -> None:
    assert PROTOCOL_VERSION == 1


def test_hello_example_from_the_spec_is_parsed() -> None:
    message = client_message.validate_python(SPEC_HELLO)

    assert isinstance(message, Hello)
    assert message.id == "7f9c"
    assert message.protocol == 1
    assert message.client.app == "desktop"
    assert message.client.client_id == "c-1"
    assert message.client.character is not None
    assert message.client.character.poses == ["idle", "sit", "wave"]
    assert message.client.character.moods[0] == "neutral"
    assert message.client.actions == [
        ActionDecl(
            name="screenshot",
            description="Capture the user's screen",
            input_schema={"type": "object", "properties": {"display": {"type": "integer"}}},
            policy="allow",
            timeout_s=15,
        )
    ]


def test_minimal_hello_gets_defaults() -> None:
    message = client_message.validate_python(
        {"type": "hello", "protocol": 1, "client": {"app": "cli", "client_id": "c-1"}}
    )

    assert isinstance(message, Hello)
    assert message.client.display_name is None
    assert message.client.character is None
    assert message.client.environment == ""
    assert message.client.actions == []


def test_action_declaration_defaults_to_asking_the_user() -> None:
    action = ActionDecl(name="open_url", description="Open a URL")

    assert action.policy == "ask"
    assert action.timeout_s == 15
    assert action.input_schema == {}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {"type": "message.send", "text": "hi", "source": "desktop"},
            MessageSend(text="hi", source="desktop"),
        ),
        (
            {"type": "message.send", "session_id": "s1", "text": "hi", "source": "phone"},
            MessageSend(session_id="s1", text="hi", source="phone"),
        ),
        ({"type": "turn.cancel", "turn_id": "t1"}, TurnCancel(turn_id="t1")),
        (
            {"type": "approval.respond", "approval_id": "a1", "decision": "always", "note": "ok"},
            ApprovalRespond(approval_id="a1", decision="always", note="ok"),
        ),
        (
            {
                "type": "action.result",
                "call_id": "c1",
                "ok": True,
                "content": [{"type": "image", "mime": "image/png", "data_b64": "AAAA"}],
            },
            ActionResult(
                call_id="c1", ok=True, content=[ImagePart(mime="image/png", data_b64="AAAA")]
            ),
        ),
        (
            {"type": "action.result", "call_id": "c1", "ok": False, "error": "no display"},
            ActionResult(call_id="c1", ok=False, error="no display"),
        ),
        ({"type": "session.list"}, SessionList()),
        ({"type": "session.create"}, SessionCreate()),
        ({"type": "session.create", "title": "Plans"}, SessionCreate(title="Plans")),
        (
            {"type": "session.rename", "session_id": "s1", "title": "New"},
            SessionRename(session_id="s1", title="New"),
        ),
        ({"type": "session.delete", "session_id": "s1"}, SessionDelete(session_id="s1")),
        ({"type": "history.get", "session_id": "s1"}, HistoryGet(session_id="s1")),
        ({"type": "ping", "id": "p1"}, Ping(id="p1")),
    ],
    ids=lambda value: value.get("type") if isinstance(value, dict) else None,
)
def test_client_message_is_parsed_into_its_model(data: dict[str, Any], expected: Envelope) -> None:
    message = client_message.validate_python(data)

    assert type(message) is type(expected)
    assert message == expected


def test_history_get_defaults_match_the_spec() -> None:
    message = HistoryGet(session_id="s1")

    assert message.before is None
    assert message.limit == 50
    assert message.include_tools is True


def test_message_send_defaults() -> None:
    message = MessageSend(text="hi", source="desktop")

    assert message.session_id is None
    assert message.attachments == []


def test_client_message_types_match_the_spec_table() -> None:
    types = {
        client_message.validate_python(data).type
        for data in (
            SPEC_HELLO,
            {"type": "message.send", "text": "hi", "source": "desktop"},
            {"type": "turn.cancel", "turn_id": "t1"},
            {"type": "approval.respond", "approval_id": "a1", "decision": "allow"},
            {"type": "action.result", "call_id": "c1", "ok": True},
            {"type": "session.list"},
            {"type": "session.create"},
            {"type": "session.rename", "session_id": "s1", "title": "New"},
            {"type": "session.delete", "session_id": "s1"},
            {"type": "history.get", "session_id": "s1"},
            {"type": "ping"},
        )
    }

    assert types == {
        "hello",
        "message.send",
        "turn.cancel",
        "approval.respond",
        "action.result",
        "session.list",
        "session.create",
        "session.rename",
        "session.delete",
        "history.get",
        "ping",
    }


def test_unknown_message_type_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        client_message.validate_python({"type": "session.explode"})

    assert excinfo.value.errors()[0]["type"] == "union_tag_invalid"


def test_message_without_type_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        client_message.validate_python({"text": "hi"})

    assert excinfo.value.errors()[0]["type"] == "union_tag_not_found"


@pytest.mark.parametrize(
    "data",
    [
        {"type": "hello", "client": {"app": "cli", "client_id": "c-1"}},
        {"type": "hello", "protocol": 1},
        {"type": "hello", "protocol": 1, "client": {"app": "cli", "client_id": ""}},
        {"type": "message.send", "source": "desktop"},
        {"type": "message.send", "text": "hi"},
        {"type": "message.send", "text": "hi", "source": "scheduler"},
        {"type": "turn.cancel"},
        {"type": "approval.respond", "approval_id": "a1", "decision": "maybe"},
        {"type": "action.result", "call_id": "c1"},
        {"type": "session.rename", "session_id": "s1"},
        {"type": "session.delete"},
        {"type": "history.get"},
        {"type": "history.get", "session_id": "s1", "limit": 0},
        {"type": "history.get", "session_id": "s1", "limit": 201},
    ],
    ids=[
        "hello-without-protocol",
        "hello-without-client",
        "hello-empty-client-id",
        "send-without-text",
        "send-without-source",
        "send-unknown-source",
        "cancel-without-turn",
        "approval-unknown-decision",
        "result-without-ok",
        "rename-without-title",
        "delete-without-session",
        "history-without-session",
        "history-limit-too-small",
        "history-limit-too-big",
    ],
)
def test_invalid_client_message_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        client_message.validate_python(data)


@pytest.mark.parametrize(
    "character",
    [
        {"name": "Zero", "poses": [], "moods": ["neutral"]},
        {"name": "Zero", "poses": ["idle"], "moods": []},
        {"poses": ["idle"], "moods": ["neutral"]},
    ],
    ids=["no-poses", "no-moods", "no-name"],
)
def test_character_needs_a_name_and_poses_with_moods(character: dict[str, Any]) -> None:
    hello = {
        "type": "hello",
        "protocol": 1,
        "client": {"app": "cli", "client_id": "c-1", "character": character},
    }

    with pytest.raises(ValidationError):
        client_message.validate_python(hello)


def test_character_may_show_no_expressions() -> None:
    hello = {
        "type": "hello",
        "protocol": 1,
        "client": {"app": "cli", "client_id": "c-1", "character": {"name": "Zero"}},
    }

    parsed = client_message.validate_python(hello)

    assert parsed.client.character is not None
    assert not parsed.client.character.expressive


@pytest.mark.parametrize(
    "action",
    [
        {"name": "", "description": "empty name"},
        {"name": "screenshot", "description": "bad policy", "policy": "sometimes"},
        {"name": "screenshot", "description": "bad timeout", "timeout_s": 0},
    ],
    ids=["empty-name", "unknown-policy", "zero-timeout"],
)
def test_invalid_action_declaration_is_rejected(action: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ActionDecl.model_validate(action)


def test_attachment_takes_inline_data_or_a_path() -> None:
    inline = Attachment(name="a.png", mime="image/png", data_b64="AAAA")
    by_path = Attachment(name="a.png", mime="image/png", path="/tmp/a.png")

    assert inline.path is None
    assert by_path.data_b64 is None


@pytest.mark.parametrize(
    "extra",
    [{}, {"data_b64": "AAAA", "path": "/tmp/a.png"}],
    ids=["neither", "both"],
)
def test_attachment_needs_exactly_one_source(extra: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        Attachment.model_validate({"name": "a.png", "mime": "image/png", **extra})


def test_message_send_carries_attachments() -> None:
    message = client_message.validate_python(
        {
            "type": "message.send",
            "text": "look",
            "source": "desktop",
            "attachments": [{"name": "a.png", "mime": "image/png", "data_b64": "AAAA"}],
        }
    )

    assert isinstance(message, MessageSend)
    assert message.attachments == [Attachment(name="a.png", mime="image/png", data_b64="AAAA")]


def test_unknown_fields_are_ignored() -> None:
    message = client_message.validate_python({"type": "ping", "sent_at": "now"})

    assert message == Ping()


WHEN = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TURN = {"session_id": "s1", "turn_id": "t1"}
SESSION = SessionInfo(id="s1", title="Plans", updated_at=WHEN)

SERVER_EVENTS: list[Envelope] = [
    HelloOk(reply_to="h1", zero_version="0.0.1", protocol=1, sessions=[SESSION]),
    Pong(reply_to="p1"),
    Error(code=ErrorCode.TURN_BUSY, message="a turn is already running", reply_to="m1"),
    TurnStarted(**TURN, source="desktop"),
    TextDelta(**TURN, delta="Hel"),
    ToolStarted(
        **TURN, call_id="c1", tool="run_command", label="$ ls -la", input={"command": "ls"}
    ),
    ToolFinished(
        **TURN, call_id="c1", status="ok", summary="exit 0", output_preview="a.txt", duration_ms=12
    ),
    ActionCall(**TURN, call_id="c2", action="screenshot", input={"display": 1}),
    ApprovalRequest(
        **TURN,
        approval_id="a1",
        tool="write_file",
        label="~/notes.md",
        input={"path": "~/notes.md", "content": "hi"},
        reason="save the notes",
        diff="+hi",
    ),
    ApprovalResolved(**TURN, approval_id="a1", decision="always", by_client_id="c-1"),
    Expression(**TURN, pose="wave", mood="happy"),
    TurnFinished(
        **TURN,
        status="ok",
        text="Hello",
        expression=ExpressionState(pose="idle", mood="neutral"),
        usage=Usage(input_tokens=10, output_tokens=3),
    ),
    SessionListOk(reply_to="l1", sessions=[SESSION]),
    SessionCreated(session=SESSION),
    SessionUpdated(session=SESSION),
    SessionDeleted(session_id="s1"),
    HistoryOk(
        reply_to="g1",
        session_id="s1",
        messages=[
            HistoryMessage(
                id="m1",
                turn_id="t1",
                role="user",
                content=[TextPart(text="hi")],
                status="ok",
                source="desktop",
                created_at=WHEN,
            )
        ],
        tool_runs=[
            ToolRun(
                call_id="c1",
                turn_id="t1",
                tool="run_command",
                label="$ ls",
                input={"command": "ls"},
                status="ok",
                summary="exit 0",
                output_preview="a.txt",
                duration_ms=12,
                created_at=WHEN,
            )
        ],
    ),
]


@pytest.mark.parametrize("event", SERVER_EVENTS, ids=lambda event: event.type)
def test_server_event_survives_a_json_round_trip(event: Envelope) -> None:
    restored = server_event.validate_json(event.model_dump_json())

    assert type(restored) is type(event)
    assert restored == event


@pytest.mark.parametrize("event", SERVER_EVENTS, ids=lambda event: event.type)
def test_server_event_survives_a_round_trip_without_null_fields(event: Envelope) -> None:
    dumped = event.model_dump_json(exclude_none=True)

    assert "null" not in dumped
    assert server_event.validate_json(dumped) == event


def test_server_event_types_match_the_spec() -> None:
    assert sorted(event.type for event in SERVER_EVENTS) == sorted(
        [
            "hello.ok",
            "pong",
            "error",
            "turn.started",
            "text.delta",
            "tool.started",
            "tool.finished",
            "action.call",
            "approval.request",
            "approval.resolved",
            "expression",
            "turn.finished",
            "session.list.ok",
            "session.created",
            "session.updated",
            "session.deleted",
            "history.ok",
        ]
    )


def test_hello_ok_example_from_the_spec_is_parsed() -> None:
    event = server_event.validate_python(
        {
            "type": "hello.ok",
            "reply_to": "7f9c",
            "zero_version": "0.1.0",
            "protocol": 1,
            "sessions": [{"id": "s1", "title": "Plans", "updated_at": "2026-01-01T12:00:00Z"}],
        }
    )

    assert isinstance(event, HelloOk)
    assert event.reply_to == "7f9c"
    assert event.sessions == [SESSION]


def test_timestamps_are_sent_as_iso_8601_utc() -> None:
    dumped = SessionCreated(session=SESSION).model_dump_json()

    assert '"updated_at":"2026-01-01T12:00:00Z"' in dumped


def test_tool_started_is_visible_unless_marked_hidden() -> None:
    visible = ToolStarted(**TURN, call_id="c1", tool="read_file")
    hidden = ToolStarted(**TURN, call_id="c2", tool="set_expression", hidden=True)

    assert visible.hidden is False
    assert hidden.hidden is True
    assert '"hidden":true' in hidden.model_dump_json()


def test_turn_finished_defaults() -> None:
    event = TurnFinished(**TURN, status="cancelled")

    assert event.text == ""
    assert event.expression is None
    assert event.usage == Usage(input_tokens=0, output_tokens=0)
    assert event.error is None


@pytest.mark.parametrize(
    "data",
    [
        {"type": "turn.started", "source": "desktop"},
        {"type": "text.delta", "delta": "x"},
        {"type": "tool.started", "call_id": "c1", "tool": "read_file"},
        {"type": "tool.finished", "call_id": "c1", "status": "ok"},
        {"type": "action.call", "call_id": "c1", "action": "screenshot"},
        {"type": "approval.request", "approval_id": "a1", "tool": "write_file"},
        {
            "type": "approval.resolved",
            "approval_id": "a1",
            "decision": "allow",
            "by_client_id": "c",
        },
        {"type": "expression", "pose": "idle", "mood": "neutral"},
        {"type": "turn.finished", "status": "ok"},
    ],
    ids=lambda data: data["type"],
)
def test_turn_events_require_session_and_turn(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        server_event.validate_python(data)

    assert server_event.validate_python({**data, **TURN}).type == data["type"]


@pytest.mark.parametrize(
    "data",
    [
        {"type": "tool.finished", **TURN, "call_id": "c1", "status": "exploded"},
        {"type": "turn.finished", **TURN, "status": "done"},
        {"type": "turn.started", **TURN, "source": "toaster"},
        {
            "type": "approval.resolved",
            **TURN,
            "approval_id": "a1",
            "decision": "maybe",
            "by_client_id": "c-1",
        },
        {"type": "error", "code": "made_up", "message": "x"},
        {"type": "session.created"},
        {"type": "session.deleted"},
        {"type": "history.ok"},
        {"type": "hello.ok", "protocol": 1},
        {"type": "ping"},
    ],
    ids=[
        "tool-status",
        "turn-status",
        "turn-source",
        "approval-decision",
        "error-code",
        "created-without-session",
        "deleted-without-id",
        "history-without-session",
        "hello-ok-without-version",
        "client-type",
    ],
)
def test_invalid_server_event_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        server_event.validate_python(data)


def test_client_and_server_message_types_do_not_overlap() -> None:
    for event in SERVER_EVENTS:
        with pytest.raises(ValidationError):
            client_message.validate_json(event.model_dump_json())
