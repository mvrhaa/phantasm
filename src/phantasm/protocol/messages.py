"""Pydantic models of the protocol messages.

Every message is a JSON object with a ``type`` field. This module holds the
shared envelope, the content parts, the messages a client sends to Phantasm
(``ClientMessage``) and the events Phantasm sends back (``ServerEvent``).
"""

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from phantasm.config import Policy
from phantasm.protocol.errors import ErrorCode


class Envelope(BaseModel):
    """Fields every protocol message has.

    A message that expects an answer carries ``id`` (chosen by the sender);
    the answer points back at it with ``reply_to``.
    """

    type: str
    id: str | None = None
    reply_to: str | None = None


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


class Character(BaseModel):
    """The character a client shows: its persona and what it can express.

    Without poses and moods the character shows no expressions: the model gets no
    ``set_expression`` and answers in one step.
    """

    name: str
    persona: str = ""
    poses: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _both_or_neither(self) -> "Character":
        if bool(self.poses) != bool(self.moods):
            raise ValueError("poses and moods go together: give both or neither")
        return self

    @property
    def expressive(self) -> bool:
        """Whether the client can show an expression at all."""
        return bool(self.poses)


class ActionDecl(BaseModel):
    """An action the client can perform; Phantasm offers it to the model as a tool."""

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


class MessageCommand(BaseModel):
    """How to run one turn: extra instructions, whether the model gets tools, which model.

    The instructions go into the system prompt of this turn only; ``name`` is kept with
    the user's message so a client can label it in the history. With ``shell`` the turn
    asks no model: it runs that command with ``run_command``, under the usual approval,
    and the command's output is the answer.
    """

    name: str
    instructions: str = ""
    tools: Literal["all", "none"] = "all"
    model: Literal["main", "small"] = "main"
    shell: str | None = Field(default=None, min_length=1)


class MessageSend(Envelope):
    """A user message; starts a turn. Without ``session_id`` a session is created."""

    type: Literal["message.send"] = "message.send"
    session_id: str | None = None
    text: str
    attachments: list[Attachment] = Field(default_factory=list)
    source: Literal["desktop", "phone"]
    command: MessageCommand | None = None


class TurnCancel(Envelope):
    """Ask to cancel a running turn."""

    type: Literal["turn.cancel"] = "turn.cancel"
    turn_id: str


class TurnSteer(Envelope):
    """Add the user's words to a running turn; they reach the model before its next step.

    A turn that is already ending takes no more: the answer is ``error``
    ``turn_not_running``, and the words should go as a new message.
    """

    type: Literal["turn.steer"] = "turn.steer"
    turn_id: str
    text: str = Field(min_length=1, max_length=20_000)


class ApprovalRespond(Envelope):
    """The user's answer to an approval request."""

    type: Literal["approval.respond"] = "approval.respond"
    approval_id: str
    decision: Literal["allow", "deny", "always"]
    note: str | None = None


class ActionResult(Envelope):
    """The outcome of an action Phantasm asked this client to perform."""

    type: Literal["action.result"] = "action.result"
    call_id: str
    ok: bool
    content: list[ContentPart] = Field(default_factory=list)
    error: str | None = None


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


class SessionRewind(Envelope):
    """Go back to just before a user message of the session.

    That message and everything after it are undone where they can be (see
    ``session.rewound``). With ``dry_run`` nothing changes and only the sender
    hears what would. Refused with ``turn_busy`` while a turn of the session runs.
    """

    type: Literal["session.rewind"] = "session.rewind"
    session_id: str
    message_id: str
    dry_run: bool = False


class ProviderSet(Envelope):
    """Choose who thinks in the user's turns: the configured provider, or with
    ``claude_code`` the user's own Claude Code.

    Everyone hears ``provider.changed``. Refused with ``provider_not_configured``
    when the ``claude`` command is not found.
    """

    type: Literal["provider.set"] = "provider.set"
    claude_code: bool


class SkillList(Envelope):
    """Ask for the skills the user may start with ``/name``, to offer them as the user types."""

    type: Literal["skill.list"] = "skill.list"


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
    | TurnSteer
    | ApprovalRespond
    | ActionResult
    | SessionList
    | SessionCreate
    | SessionRename
    | SessionDelete
    | SessionRewind
    | ProviderSet
    | SkillList
    | HistoryGet
    | Ping,
    Field(discriminator="type"),
]
"""Any message a client may send. An unknown ``type`` fails validation."""


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
    source: Literal["desktop", "phone", "scheduler", "agent", "recovery"] | None = None
    command: str | None = None
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


class Thinker(BaseModel):
    """Who thinks in the user's turns; ``name`` is for people ("Claude Sonnet", "Gemini")."""

    claude_code: bool = False
    name: str


class HelloOk(Envelope):
    """Answer to ``hello``; ``thinker`` tells who thinks in the user's turns."""

    type: Literal["hello.ok"] = "hello.ok"
    phantasm_version: str
    protocol: int
    sessions: list[SessionInfo] = Field(default_factory=list)
    thinker: Thinker | None = None


class ProviderChanged(Envelope):
    """Someone chose who thinks in the user's turns (``provider.set``)."""

    type: Literal["provider.changed"] = "provider.changed"
    thinker: Thinker


class Pong(Envelope):
    """Answer to ``ping``."""

    type: Literal["pong"] = "pong"


class Error(Envelope):
    """Something went wrong; ``reply_to`` is set when a message caused it."""

    type: Literal["error"] = "error"
    code: ErrorCode
    message: str


class TurnStarted(TurnEvent):
    """A turn began."""

    type: Literal["turn.started"] = "turn.started"
    source: Literal["desktop", "phone", "scheduler", "agent", "recovery"]


class TurnSteered(TurnEvent):
    """The user's words went into a running turn, as a user message before the model's next step.

    The sender of ``turn.steer`` gets it with ``reply_to``.
    """

    type: Literal["turn.steered"] = "turn.steered"
    text: str


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


class FileUndo(BaseModel):
    """What going back did to one file a tool had changed."""

    path: str
    change: Literal["restored", "deleted", "changed elsewhere"]


class SessionRewound(Envelope):
    """A session went back to just before a user message (or, on a dry run, would).

    ``text`` is that message, for the input. Gone: ``removed_messages``
    messages, ``removed_todos`` todo items and ``removed_session_memories``
    entries of the conversation's memory added since, the agents it started
    since. ``files`` were put back as they were; a file changed outside Phantasm
    since is left alone. ``kept`` counts the calls whose effects stay, by tool.
    Every client gets it after a real rewind and should load the history again.
    """

    type: Literal["session.rewound"] = "session.rewound"
    session_id: str
    message_id: str
    text: str
    dry_run: bool = False
    removed_messages: int = 0
    files: list[FileUndo] = Field(default_factory=list)
    removed_todos: int = 0
    removed_session_memories: int = 0
    stopped_agents: list[str] = Field(default_factory=list)
    kept: dict[str, int] = Field(default_factory=dict)


class TodoUpdated(Envelope):
    """The todo list of a session changed; carries the whole list."""

    type: Literal["todo.updated"] = "todo.updated"
    session_id: str
    items: list[TodoItem] = Field(default_factory=list)


class FileSent(TurnEvent):
    """A file was sent to the user, e.g. by ``send_file``; clients show it in the chat."""

    type: Literal["file"] = "file"
    call_id: str | None = None
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


class SessionListOk(Envelope):
    """Answer to ``session.list``."""

    type: Literal["session.list.ok"] = "session.list.ok"
    sessions: list[SessionInfo] = Field(default_factory=list)


class SkillInfo(BaseModel):
    """A skill as a client offers it after ``/``."""

    name: str
    description: str = ""
    argument_hint: str = ""


class SkillListOk(Envelope):
    """Answer to ``skill.list``: the skills the user may start, by name."""

    type: Literal["skill.list.ok"] = "skill.list.ok"
    skills: list[SkillInfo] = Field(default_factory=list)


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
    until_message_id: str


class AgentInfo(AgentRef):
    """What clients are told about an agent."""

    parent_session_id: str
    session_id: str | None = None
    task: str
    status: AgentStatus
    report: str | None = None
    created_at: datetime
    updated_at: datetime
    progress: str | None = None
    review_of: str | None = None
    claude: bool = False


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
    | TurnSteered
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
    | SkillListOk
    | SessionCreated
    | SessionUpdated
    | SessionDeleted
    | SessionCompacted
    | SessionRewound
    | AgentUpdated
    | ProviderChanged
    | HistoryOk,
    Field(discriminator="type"),
]
"""Any message Phantasm may send to a client."""
