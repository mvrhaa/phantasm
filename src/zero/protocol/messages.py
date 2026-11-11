"""Pydantic models of the protocol messages.

Every message is a JSON object with a ``type`` field. This module holds the
shared envelope and the content parts; the concrete messages build on them.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


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
    """A file referred to by its name and path."""

    type: Literal["file"] = "file"
    name: str
    path: str


ContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="type")]
"""One item of message content: tool results and attachments are lists of these."""
