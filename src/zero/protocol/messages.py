"""Pydantic models of the protocol messages.

Every message is a JSON object with a ``type`` field. This module holds the
shared envelope, the content parts, the messages a client sends to Zero
(``ClientMessage``) and the events Zero sends back (``ServerEvent``).
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from zero.config import Policy
from zero.protocol.errors import ErrorCode


class Envelope(BaseModel):
    """Fields every protocol message has.

    A message that expects an answer carries ``id`` (chosen by the sender);
    the answer points back at it with ``reply_to``.
    """

    type: str
    id: str | None = None
    reply_to: str | None = None


# --- Content parts -----------------------------------------------------------


class TextPart(BaseModel):
    """A piece of plain text."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image passed inline as base64."""

    type: Literal["image"] = "image"
    mime: str
    data_b64: str


class FilePart(BaseModel):
    """A file referred to by its name and path; ``mime`` tells an image from other files."""

    type: Literal["file"] = "file"
    name: str
    path: str
    mime: str | None = None


ContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="type")]
"""One item of message content: tool results and attachments are lists of these."""


# --- Client -> Zero: handshake ----------------------------------------------


class Character(BaseModel):
    """The character a client shows: its persona and what it can express."""

    name: str
    persona: str = ""
    poses: list[str] = Field(min_length=1)
    moods: list[str] = Field(min_length=1)


class ActionDecl(BaseModel):
    """An action the client can perform; Zero offers it to the model as a tool."""

    name: str = Field(min_length=1)
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    policy: Policy = "ask"
    timeout_s: float = Field(default=15, gt=0)


class ClientInfo(BaseModel):
    """How a client introduces itself in ``hello``."""

    app: str
    client_id: str = Field(min_length=1)
    display_name: str | None = None
    character: Character | None = None
    environment: str = ""
    actions: list[ActionDecl] = Field(default_factory=list)


class Hello(Envelope):
    """First message of a connection; sending it again updates the declaration."""

    type: Literal["hello"] = "hello"
    protocol: int
    client: ClientInfo


# --- Client -> Zero: turns ---------------------------------------------------


class Attachment(BaseModel):
    """A file sent along with a user message, either inline or as a local path."""

    name: str
    mime: str
    data_b64: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.data_b64 is None) == (self.path is None):
            raise ValueError("an attachment needs exactly one of data_b64 or path")
        return self


class MessageSend(Envelope):
    """A user message; starts a turn. Without ``session_id`` a session is created."""

    type: Literal["message.send"] = "message.send"
    session_id: str | None = None
    text: str
    attachments: list[Attachment] = Field(default_factory=list)
    source: Literal["desktop", "phone"]


class TurnCancel(Envelope):
    """Ask to cancel a running turn."""

    type: Literal["turn.cancel"] = "turn.cancel"
    turn_id: str


class ApprovalRespond(Envelope):
    """The user's answer to an approval request."""

    type: Literal["approval.respond"] = "approval.respond"
    approval_id: str
    decision: Literal["allow", "deny", "always"]
    note: str | None = None


class ActionResult(Envelope):
    """The outcome of an action Zero asked this client to perform."""

    type: Literal["action.result"] = "action.result"
    call_id: str
    ok: bool
    content: list[ContentPart] = Field(default_factory=list)
    error: str | None = None


# --- Client -> Zero: sessions and history ------------------------------------


class SessionList(Envelope):
    """Ask for the list of sessions."""

    type: Literal["session.list"] = "session.list"


class SessionCreate(Envelope):
    """Create a session, optionally with a title."""

    type: Literal["session.create"] = "session.create"
    title: str | None = None


class SessionRename(Envelope):
    """Change the title of a session."""

    type: Literal["session.rename"] = "session.rename"
    session_id: str
    title: str


class SessionDelete(Envelope):
    """Delete a session with its history."""

    type: Literal["session.delete"] = "session.delete"
    session_id: str


class HistoryGet(Envelope):
    """Ask for a page of a session's history, newest first when paging back."""

    type: Literal["history.get"] = "history.get"
    session_id: str
    before: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    include_tools: bool = True


class Ping(Envelope):
    """Liveness check; answered with ``pong``."""

    type: Literal["ping"] = "ping"


ClientMessage = Annotated[
    Hello
    | MessageSend
    | TurnCancel
    | ApprovalRespond
    | ActionResult
    | SessionList
    | SessionCreate
    | SessionRename
    | SessionDelete
    | HistoryGet
    | Ping,
    Field(discriminator="type"),
]
"""Any message a client may send. An unknown ``type`` fails validation."""


# --- Zero -> client: shared pieces -------------------------------------------


SessionKind = Literal["chat", "agent", "scheduled"]


class SessionInfo(BaseModel):
    """What clients are told about a session.

    ``kind`` keeps the user's chats apart from the sessions of agents and schedules.
    """

    id: str
    title: str | None = None
    updated_at: datetime
    kind: SessionKind = "chat"


class ExpressionState(BaseModel):
    """A pose and a mood of the character."""

    pose: str
    mood: str


class Usage(BaseModel):
    """Tokens spent by a turn."""

    input_tokens: int = 0
    output_tokens: int = 0


class HistoryMessage(BaseModel):
    """A stored message as returned by ``history.ok``."""

    id: str
    turn_id: str | None = None
    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]
    status: Literal["ok", "cancelled", "error"] | None = None
    source: Literal["desktop", "phone", "scheduler", "agent"] | None = None
    created_at: datetime


TodoStatus = Literal["pending", "in_progress", "done"]
AgentStatus = Literal["working", "asking", "waiting_user", "done", "failed", "stopped"]


class TodoItem(BaseModel):
    """One item of a session's todo list."""

    number: int
    text: str
    status: TodoStatus = "pending"


class ToolRun(BaseModel):
    """A finished tool call as returned by ``history.ok``."""

    call_id: str
    turn_id: str
    tool: str
    label: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "denied", "cancelled"]
    summary: str = ""
    output_preview: str = ""
    duration_ms: int = 0
    created_at: datetime


class AgentRef(BaseModel):
    """Which agent something belongs to."""

    id: str
    name: str


class TurnEvent(Envelope):
    """Base of the events of a turn: all of them name the session and the turn.

    ``agent`` is set on the events of an agent's turn, so clients can show who acts.
    """

    session_id: str
    turn_id: str
    agent: AgentRef | None = None


# --- Zero -> client: handshake and service -----------------------------------


class HelloOk(Envelope):
    """Answer to ``hello``."""

    type: Literal["hello.ok"] = "hello.ok"
    zero_version: str
    protocol: int
    sessions: list[SessionInfo] = Field(default_factory=list)


class Pong(Envelope):
    """Answer to ``ping``."""

    type: Literal["pong"] = "pong"


class Error(Envelope):
    """Something went wrong; ``reply_to`` is set when a message caused it."""

    type: Literal["error"] = "error"
    code: ErrorCode
    message: str


# --- Zero -> client: turn events ---------------------------------------------


class TurnStarted(TurnEvent):
    """A turn began."""

    type: Literal["turn.started"] = "turn.started"
    source: Literal["desktop", "phone", "scheduler", "agent"]


class TextDelta(TurnEvent):
    """The next chunk of the model's answer."""

    type: Literal["text.delta"] = "text.delta"
    delta: str


class ToolStarted(TurnEvent):
    """A tool call began. ``hidden`` tools are not shown as a separate line."""

    type: Literal["tool.started"] = "tool.started"
    call_id: str
    tool: str
    label: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    hidden: bool = False


class ToolFinished(TurnEvent):
    """A tool call ended."""

    type: Literal["tool.finished"] = "tool.finished"
    call_id: str
    status: Literal["ok", "error", "denied", "cancelled"]
    summary: str = ""
    output_preview: str = ""
    duration_ms: int = 0


class ActionCall(TurnEvent):
    """Ask one client to perform an action it declared in ``hello``."""

    type: Literal["action.call"] = "action.call"
    call_id: str
    action: str
    input: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(TurnEvent):
    """Ask the user to allow or deny a tool call."""

    type: Literal["approval.request"] = "approval.request"
    approval_id: str
    tool: str
    label: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    diff: str | None = None


class ApprovalResolved(TurnEvent):
    """An approval was answered, so other clients can drop its card."""

    type: Literal["approval.resolved"] = "approval.resolved"
    approval_id: str
    decision: Literal["allow", "deny", "always"]
    by_client_id: str


class TodoUpdated(Envelope):
    """The todo list of a session changed; carries the whole list."""

    type: Literal["todo.updated"] = "todo.updated"
    session_id: str
    items: list[TodoItem] = Field(default_factory=list)


class FileSent(TurnEvent):
    """A file was sent to the user, e.g. by ``send_file``; clients show it in the chat."""

    type: Literal["file"] = "file"
    call_id: str | None = None  # the tool call that sent it
    file: FilePart


class Expression(TurnEvent):
    """The character changed its pose and mood."""

    type: Literal["expression"] = "expression"
    pose: str
    mood: str


class TurnFinished(TurnEvent):
    """A turn ended; carries the full final text."""

    type: Literal["turn.finished"] = "turn.finished"
    status: Literal["ok", "error", "cancelled", "max_steps"]
    text: str = ""
    expression: ExpressionState | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


# --- Zero -> client: sessions and history ------------------------------------


class SessionListOk(Envelope):
    """Answer to ``session.list``."""

    type: Literal["session.list.ok"] = "session.list.ok"
    sessions: list[SessionInfo] = Field(default_factory=list)


class SessionCreated(Envelope):
    """A session was created."""

    type: Literal["session.created"] = "session.created"
    session: SessionInfo


class SessionUpdated(Envelope):
    """A session changed, for example it was renamed."""

    type: Literal["session.updated"] = "session.updated"
    session: SessionInfo


class SessionCompacted(Envelope):
    """The start of a long session was summarized so the rest fits the model's context."""

    type: Literal["session.compacted"] = "session.compacted"
    session_id: str
    summary: str
    until_message_id: str  # the last message the summary covers


class AgentInfo(AgentRef):
    """What clients are told about an agent."""

    parent_session_id: str
    session_id: str | None = None  # None once the agent's session is deleted
    task: str
    status: AgentStatus
    report: str | None = None
    created_at: datetime
    updated_at: datetime


class AgentUpdated(Envelope):
    """An agent was started or its status changed; carries the whole agent."""

    type: Literal["agent.updated"] = "agent.updated"
    agent: AgentInfo


class SessionDeleted(Envelope):
    """A session was deleted."""

    type: Literal["session.deleted"] = "session.deleted"
    session_id: str


class HistoryOk(Envelope):
    """Answer to ``history.get``: messages in ascending order plus tool runs."""

    type: Literal["history.ok"] = "history.ok"
    session_id: str
    messages: list[HistoryMessage] = Field(default_factory=list)
    tool_runs: list[ToolRun] = Field(default_factory=list)


ServerEvent = Annotated[
    HelloOk
    | Pong
    | Error
    | TurnStarted
    | TextDelta
    | ToolStarted
    | ToolFinished
    | ActionCall
    | ApprovalRequest
    | ApprovalResolved
    | FileSent
    | TodoUpdated
    | Expression
    | TurnFinished
    | SessionListOk
    | SessionCreated
    | SessionUpdated
    | SessionDeleted
    | SessionCompacted
    | AgentUpdated
    | HistoryOk,
    Field(discriminator="type"),
]
"""Any message Zero may send to a client."""
