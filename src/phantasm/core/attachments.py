"""Attachments of user messages: kept in ``files/``, shown to the model as images or notes.

A message stores its attachments as file parts that point into ``files_dir()``,
never as base64. When a prompt is built, every file part becomes something a
provider can take: an image of the current message becomes an image part
(if the model can see images and it is small enough), everything else a short
text note with the path, so the model can read the file with a tool if needed.
"""

import asyncio
import base64
import binascii
import contextlib
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

from phantasm.paths import files_dir
from phantasm.protocol.messages import Attachment, ContentPart, FilePart, ImagePart, TextPart
from phantasm.providers.base import Message
from phantasm.tools.builtin.files import MAX_IMAGE_BYTES, SNIFF_BYTES, human_size, image_mime
from phantasm.tools.permissions import PathDenied, PathGuard

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024


class AttachmentError(Exception):
    """An attachment cannot be taken; the message is for the client that sent it."""


def _safe_name(name: str) -> str:
    """The last part of a name, so it cannot point anywhere outside its folder."""
    cleaned = Path(name.replace("\\", "/")).name.strip()
    return cleaned if cleaned not in ("", ".", "..") else "attachment"


def _read(attachment: Attachment, guard: PathGuard) -> bytes:
    shown = attachment.name
    if attachment.data_b64 is not None:
        try:
            data = base64.b64decode(attachment.data_b64, validate=True)
        except (binascii.Error, ValueError):
            raise AttachmentError(f"attachment {shown!r} is not valid base64") from None
    else:
        assert attachment.path is not None
        try:
            path = guard.check(attachment.path)
        except PathDenied as exc:
            raise AttachmentError(f"attachment {shown!r}: {exc}") from None
        if not path.is_file():
            raise AttachmentError(f"attachment {shown!r}: {attachment.path} is not a file")
        if path.stat().st_size > MAX_ATTACHMENT_BYTES:
            raise AttachmentError(_too_big(shown, path.stat().st_size))
        data = path.read_bytes()
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise AttachmentError(_too_big(shown, len(data)))
    return data


def _too_big(name: str, size: int) -> str:
    limit = human_size(MAX_ATTACHMENT_BYTES)
    return f"attachment {name!r} is {human_size(size)}; the limit is {limit}"


def _mime(attachment: Attachment, data: bytes) -> str | None:
    """An image type only when the bytes show one; a claimed image that is not one is just data."""
    detected = image_mime(data[:SNIFF_BYTES])
    if detected is not None:
        return detected
    claimed = attachment.mime.strip().lower()
    if not claimed or claimed.startswith("image/"):
        return "application/octet-stream"
    return claimed


def save_attachments(attachments: Sequence[Attachment], guard: PathGuard) -> list[FilePart]:
    """Copy every attachment into its own folder in ``files_dir()``.

    Inline data is decoded; a path must pass the sandbox. ``mime`` is the
    image type found in the bytes, or what the client said for other files.
    Blocking; raises ``AttachmentError`` before anything is saved if one is bad.
    """
    loaded = [(attachment, _read(attachment, guard)) for attachment in attachments]
    parts = []
    for attachment, data in loaded:
        folder = files_dir() / str(uuid.uuid4())
        folder.mkdir(parents=True)
        path = folder / _safe_name(attachment.name)
        path.write_bytes(data)
        parts.append(FilePart(name=path.name, path=str(path), mime=_mime(attachment, data)))
    return parts


_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}


def _image_to_file(part: ImagePart) -> ContentPart:
    try:
        data = base64.b64decode(part.data_b64, validate=True)
    except (binascii.Error, ValueError):
        return TextPart(text="[an image that could not be read]")
    folder = files_dir() / str(uuid.uuid4())
    folder.mkdir(parents=True)
    path = folder / f"image.{_EXTENSIONS.get(part.mime, 'bin')}"
    path.write_bytes(data)
    return FilePart(name=path.name, path=str(path), mime=part.mime)


def _images_to_files(content: Sequence[ContentPart]) -> list[ContentPart]:
    return [_image_to_file(part) if isinstance(part, ImagePart) else part for part in content]


async def images_to_files(content: Sequence[ContentPart]) -> list[ContentPart]:
    """Content as it is kept in the history: every inline image saved to ``files_dir()``.

    The image becomes a file part that points at the saved copy, so the
    database never holds base64. Content without images comes back as it is.
    """
    if not any(isinstance(part, ImagePart) for part in content):
        return list(content)
    return await asyncio.to_thread(_images_to_files, content)


def _remove_folders(paths: Sequence[str]) -> int:
    root = files_dir().resolve()
    removed = 0
    for raw in paths:
        with contextlib.suppress(OSError):
            folder = Path(raw).resolve().parent
            if folder.parent == root and folder.is_dir():
                shutil.rmtree(folder)
                removed += 1
    return removed


async def remove_files(paths: Sequence[str]) -> int:
    """Delete the folders in ``files_dir()`` that hold these files; other paths are left alone.

    Returns how many folders were removed.
    """
    if not paths:
        return 0
    return await asyncio.to_thread(_remove_folders, paths)


def _is_image(part: FilePart) -> bool:
    return part.mime is not None and part.mime.startswith("image/")


def _note(part: FilePart, why: str = "") -> TextPart:
    kind = "image" if _is_image(part) else "file"
    return TextPart(text=f"[{kind} {part.name} at {part.path}{why}]")


def _as_image(part: FilePart, can_see: bool) -> ContentPart:
    if not can_see:
        return _note(part, "; this model cannot see images")
    try:
        data = Path(part.path).read_bytes()
    except OSError:
        return _note(part, "; the file is gone")
    mime = image_mime(data[:SNIFF_BYTES])
    if mime is None:
        return _note(part)
    if len(data) > MAX_IMAGE_BYTES:
        return _note(part, f"; too big to show ({human_size(len(data))})")
    return ImagePart(mime=mime, data_b64=base64.b64encode(data).decode("ascii"))


def _expand(messages: list[Message], can_see_images: bool) -> list[Message]:
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
    expanded = []
    for index, message in enumerate(messages):
        if not any(isinstance(part, FilePart) for part in message.content):
            expanded.append(message)
            continue
        content: list[ContentPart] = []
        for part in message.content:
            if not isinstance(part, FilePart):
                content.append(part)
            elif index == last_user and _is_image(part):
                content.append(_as_image(part, can_see_images))
            else:
                content.append(_note(part))
        expanded.append(message.model_copy(update={"content": content}))
    return expanded


async def prepare_for_model(messages: list[Message], can_see_images: bool) -> list[Message]:
    """Replace file parts with what a provider can take; reads images in a worker thread."""
    return await asyncio.to_thread(_expand, messages, can_see_images)
