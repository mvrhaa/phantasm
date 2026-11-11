"""Pydantic models of the protocol messages.

Every message is a JSON object with a ``type`` field. This module holds the
shared envelope, the content parts and the messages a client sends to Zero.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from zero.config import Policy


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
    """A file referred to by its name and path."""

    type: Literal["file"] = "file"
    name: str
    path: str


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
