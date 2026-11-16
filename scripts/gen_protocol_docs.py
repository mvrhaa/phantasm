"""Write docs/protocol.md, the protocol reference, from the pydantic models of the protocol.

    uv run python scripts/gen_protocol_docs.py          # rewrite docs/protocol.md
    uv run python scripts/gen_protocol_docs.py --check  # exit 1 if it is out of date

Every message gets its docstring, a table of its fields and an example. The
examples are built from the models themselves, so they always validate; a
message without an example stops the script.
"""

import enum
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from zero import __version__
from zero.config import ServerConfig
from zero.protocol import messages as m
from zero.protocol.errors import PROTOCOL_VERSION, ErrorCode

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "protocol.md"

T0 = datetime(2026, 11, 12, 9, 30, tzinfo=UTC)
T1 = datetime(2026, 11, 12, 9, 31, 5, tzinfo=UTC)
SESSION = m.SessionInfo(id="s-1", title="Nginx reverse proxy with TLS", updated_at=T1)
AGENT = m.AgentRef(id="a-1", name="log reader")
TURN = {"session_id": "s-1", "turn_id": "t-1"}

CLIENT_EXAMPLES: list[BaseModel] = [
    m.Hello(
        id="h-1",
        protocol=PROTOCOL_VERSION,
        client=m.ClientInfo(
            app="desktop",
            client_id="desktop-macbook",
            display_name="Desktop on MacBook",
            character=m.Character(
                name="Zero",
                persona="A calm, curious companion.",
                poses=["idle", "sit", "wave"],
                moods=["neutral", "happy", "sad"],
            ),
            environment="macOS desktop app; replies show in a chat window that renders Markdown",
            actions=[
                m.ActionDecl(
                    name="screenshot",
                    description="Capture the user's screen",
                    input_schema={"type": "object", "properties": {"display": {"type": "integer"}}},
                    policy="allow",
                    timeout_s=15,
                )
            ],
        ),
    ),
    m.MessageSend(
        id="m-1",
        session_id="s-1",
        text="What is in this log?",
        attachments=[m.Attachment(name="error.log", mime="text/plain", path="/tmp/error.log")],
        source="desktop",
    ),
    m.TurnCancel(id="c-1", turn_id="t-1"),
    m.TurnSteer(id="st-1", turn_id="t-1", text="Only the last hour, please"),
    m.ApprovalRespond(id="r-1", approval_id="ap-1", decision="allow"),
    m.ActionResult(
        id="ar-1",
        call_id="call-7",
        ok=True,
        content=[m.ImagePart(mime="image/png", data_b64="iVBORw0KGgo=")],
    ),
    m.SessionList(id="l-1"),
    m.SessionCreate(id="sc-1", title="Plans"),
    m.SessionRename(id="sr-1", session_id="s-1", title="Server setup"),
    m.SessionDelete(id="sd-1", session_id="s-1"),
    m.SessionRewind(id="rw-1", session_id="s-1", message_id="msg-12"),
    m.ProviderSet(id="pv-1", claude_code=True),
    m.HistoryGet(id="hg-1", session_id="s-1", limit=50),
    m.Ping(id="p-1"),
]

SERVER_EXAMPLES: list[BaseModel] = [
    m.HelloOk(
        reply_to="h-1",
        zero_version=__version__,
        protocol=PROTOCOL_VERSION,
        sessions=[SESSION],
        thinker=m.Thinker(name="Gemini"),
    ),
    m.Pong(reply_to="p-1"),
    m.Error(
        reply_to="m-1",
        code=ErrorCode.TURN_BUSY,
        message="a turn is already running in this session",
    ),
    m.TurnStarted(**TURN, source="desktop"),
    m.TurnSteered(**TURN, reply_to="st-1", text="Only the last hour, please"),
    m.TextDelta(**TURN, delta="The log shows "),
    m.ToolStarted(
        **TURN,
        call_id="call-1",
        tool="read_file",
        label="/tmp/error.log",
        input={"path": "/tmp/error.log"},
    ),
    m.ToolFinished(
        **TURN,
        call_id="call-1",
        status="ok",
        summary="120 lines",
        output_preview="2026-11-12 09:30:01 nginx: [emerg] bind() to 0.0.0.0:443 failed",
        duration_ms=4,
    ),
    m.ActionCall(**TURN, call_id="call-7", action="screenshot", input={"display": 1}),
    m.ApprovalRequest(
        **TURN,
        agent=AGENT,
        approval_id="ap-1",
        tool="run_command",
        label="$ sudo nginx -t",
        input={"command": "sudo nginx -t"},
        reason="Check the nginx config for errors",
    ),
    m.ApprovalResolved(
        **TURN, agent=AGENT, approval_id="ap-1", decision="allow", by_client_id="desktop-macbook"
    ),
    m.FileSent(
        **TURN,
        call_id="call-9",
        file=m.FilePart(
            name="report.pdf", path="/Users/me/Downloads/report.pdf", mime="application/pdf"
        ),
    ),
    m.TodoUpdated(
        session_id="s-1",
        items=[
            m.TodoItem(number=1, text="Read the error log", status="done"),
            m.TodoItem(number=2, text="Fix the port clash", status="in_progress"),
        ],
    ),
    m.Expression(**TURN, pose="sit", mood="happy"),
    m.TurnFinished(
        **TURN,
        status="ok",
        text="The log shows that port 443 is already taken.",
        expression=m.ExpressionState(pose="idle", mood="neutral"),
        usage=m.Usage(input_tokens=7285, output_tokens=450),
    ),
    m.SessionListOk(reply_to="l-1", sessions=[SESSION]),
    m.SessionCreated(reply_to="sc-1", session=SESSION),
    m.SessionUpdated(session=SESSION),
    m.SessionDeleted(reply_to="sd-1", session_id="s-1"),
    m.SessionRewound(
        reply_to="rw-1",
        session_id="s-1",
        message_id="msg-12",
        text="Now write it to notes.md",
        removed_messages=6,
        files=[m.FileUndo(path="/Users/me/notes.md", change="restored")],
        kept={"run_command": 1},
    ),
    m.SessionCompacted(
        session_id="s-1", summary="The user set up nginx …", until_message_id="msg-40"
    ),
    m.ProviderChanged(reply_to="pv-1", thinker=m.Thinker(claude_code=True, name="Claude Sonnet")),
    m.AgentUpdated(
        agent=m.AgentInfo(
            id="a-1",
            name="log reader",
            parent_session_id="s-1",
            session_id="s-2",
            task="Read /tmp/error.log and find the cause",
            status="done",
            report="Port 443 is used by another process.",
            created_at=T0,
            updated_at=T1,
        )
    ),
    m.HistoryOk(
        reply_to="hg-1",
        session_id="s-1",
        messages=[
            m.HistoryMessage(
                id="msg-1",
                turn_id="t-1",
                role="user",
                content=[m.TextPart(text="What is in this log?")],
                status="ok",
                source="desktop",
                created_at=T0,
            )
        ],
        tool_runs=[
            m.ToolRun(
                call_id="call-1",
                turn_id="t-1",
                tool="read_file",
                label="/tmp/error.log",
                input={"path": "/tmp/error.log"},
                status="ok",
                summary="120 lines",
                created_at=T0,
            )
        ],
    ),
]

SHARED: list[type[BaseModel]] = [
    m.TextPart,
    m.ImagePart,
    m.FilePart,
    m.ClientInfo,
    m.Character,
    m.ActionDecl,
    m.Attachment,
    m.MessageCommand,
    m.SessionInfo,
    m.AgentRef,
    m.AgentInfo,
    m.HistoryMessage,
    m.ToolRun,
    m.TodoItem,
    m.ExpressionState,
    m.Usage,
]

SCALARS = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    datetime: "ISO 8601 time",
}


def members(union: Any) -> list[type[BaseModel]]:
    """The models of a discriminated union, in the order they are declared."""
    return list(get_args(get_args(union)[0]))


def type_of(model: type[BaseModel]) -> str:
    return str(model.model_fields["type"].default)


def describe(annotation: Any) -> str:
    """A field's type in words, with links to the shared objects."""
    origin = get_origin(annotation)
    if annotation is m.ContentPart:
        return "[ContentPart](#contentpart)"
    if origin is Annotated:
        return describe(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if set(options) == set(get_args(get_args(m.ContentPart)[0])):
            return "[ContentPart](#contentpart)"
        return " or ".join(describe(o) for o in options)
    if origin is Literal:
        return " \\| ".join(f"`{json.dumps(v)}`" for v in get_args(annotation))
    if origin is list:
        return f"array of {describe(get_args(annotation)[0])}"
    if origin is dict:
        return "object"
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return "[error code](#error-codes)"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return f"[{annotation.__name__}](#{annotation.__name__.lower()})"
    return SCALARS.get(annotation, str(annotation))


def default_of(model: type[BaseModel], name: str) -> str:
    field = model.model_fields[name]
    if field.is_required():
        return "required"
    if field.default_factory is not None:
        value = field.default_factory()  # type: ignore[call-arg]
        value = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    else:
        value = field.default
    if value is PydanticUndefined:
        return "required"
    return f"`{json.dumps(value, ensure_ascii=False)}`"


def fields_table(model: type[BaseModel]) -> list[str]:
    rows = ["| Field | Type | Default |", "|---|---|---|"]
    for name, field in model.model_fields.items():
        if name == "type":
            continue
        rows.append(f"| `{name}` | {describe(field.annotation)} | {default_of(model, name)} |")
    return rows if len(rows) > 2 else ["No fields besides `type`."]


def docstring(model: type[BaseModel]) -> str:
    text = (model.__doc__ or "").strip().replace("``", "`")
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def example_json(example: BaseModel) -> list[str]:
    data = example.model_dump(mode="json", exclude_none=True)
    return ["```json", json.dumps(data, ensure_ascii=False, indent=2), "```"]


def message_sections(union: Any, examples: list[BaseModel]) -> list[str]:
    by_type = {type_of(type(example)): example for example in examples}
    lines: list[str] = []
    for model in members(union):
        name = type_of(model)
        if name not in by_type:
            raise SystemExit(f"no example for {name!r}: add one to {Path(__file__).name}")
        lines += [f"### `{name}`", "", docstring(model), "", *fields_table(model), ""]
        lines += [*example_json(by_type[name]), ""]
    return lines


def render() -> str:
    """The whole reference as Markdown."""
    port = ServerConfig().port
    lines = [
        "# Zero protocol reference",
        "",
        f"Protocol version **{PROTOCOL_VERSION}**, Zero {__version__}. Generated by "
        "`scripts/gen_protocol_docs.py` from `src/zero/protocol/messages.py`; do not edit by "
        "hand. The design behind it is in section 6 of `ZERO_SPEC.md`.",
        "",
        "## Connecting",
        "",
        f"- WebSocket at `ws://127.0.0.1:{port}/ws`, on the loopback interface only.",
        "- The local token goes in the header `Authorization: Bearer <token>` (or `?token=` "
        "while debugging); `zero token` prints it. A wrong token closes the socket with `4401`.",
        "- Every message is a JSON object with a `type`. A message that expects an answer "
        "carries an `id`; the answer points back at it with `reply_to`.",
        "- The first message is `hello`. An unknown `type` gets an `error` with "
        "`unknown_type`; the connection stays open.",
        "- Events of a turn go to every connected client, so all of them show the same.",
        "",
        "## Client → Zero",
        "",
        *message_sections(m.ClientMessage, CLIENT_EXAMPLES),
        "## Zero → client",
        "",
        *message_sections(m.ServerEvent, SERVER_EXAMPLES),
        "## Shared objects",
        "",
        "### ContentPart",
        "",
        "One item of content: a [TextPart](#textpart), an [ImagePart](#imagepart) or a "
        "[FilePart](#filepart), told apart by `type`.",
        "",
    ]
    for model in SHARED:
        lines += [f"### {model.__name__}", "", docstring(model), "", *fields_table(model), ""]
    lines += ["## Error codes", ""]
    lines += [f"- `{code.value}`" for code in ErrorCode]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT} is out of date: run scripts/gen_protocol_docs.py")
            return 1
        return 0
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
