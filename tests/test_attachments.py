"""Tests of attachments: saving them, storing paths, and what the model gets."""

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from zero.config import Config
from zero.core import attachments
from zero.core.attachments import AttachmentError, prepare_for_model, save_attachments
from zero.paths import db_path, files_dir
from zero.protocol.messages import Attachment, FilePart, ImagePart, TextPart
from zero.providers.base import Message, TextDelta
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools.permissions import PathGuard

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
PNG_B64 = base64.b64encode(PNG).decode()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("KEY")
    monkeypatch.setenv("HOME", str(home))
    return home


def guard() -> PathGuard:
    return PathGuard.from_config(Config().permissions)


# --- Saving ------------------------------------------------------------------


def test_inline_image_is_saved_with_its_detected_type(home: Path) -> None:
    (part,) = save_attachments(
        [Attachment(name="shot.png", mime="image/png", data_b64=PNG_B64)], guard()
    )

    path = Path(part.path)
    assert path.parent.parent == files_dir()
    assert (part.name, part.mime, path.read_bytes()) == ("shot.png", "image/png", PNG)


def test_type_comes_from_the_bytes_not_from_the_client(home: Path) -> None:
    lying = Attachment(name="x.png", mime="image/png", data_b64=base64.b64encode(b"plain").decode())

    (part,) = save_attachments([lying], guard())

    assert part.mime == "application/octet-stream"


def test_other_files_keep_the_clients_type(home: Path) -> None:
    pdf = Attachment(
        name="r.pdf", mime="application/pdf", data_b64=base64.b64encode(b"%PDF").decode()
    )

    (part,) = save_attachments([pdf], guard())

    assert part.mime == "application/pdf"


def test_file_by_path_is_copied(home: Path) -> None:
    (home / "notes.txt").write_text("hello")

    (part,) = save_attachments(
        [Attachment(name="notes.txt", mime="text/plain", path="~/notes.txt")], guard()
    )

    assert Path(part.path).read_text() == "hello"
    assert Path(part.path) != home / "notes.txt"


@pytest.mark.parametrize("name", ["../../evil.png", "/etc/passwd", "..", "a\\..\\b.png", ""])
def test_names_cannot_escape_their_folder(home: Path, name: str) -> None:
    (part,) = save_attachments([Attachment(name=name, mime="image/png", data_b64=PNG_B64)], guard())

    assert Path(part.path).parent.parent == files_dir()
    assert "/" not in part.name and part.name not in ("", ".", "..")


@pytest.mark.parametrize(
    ("attachment", "message"),
    [
        (
            Attachment(name="x", mime="image/png", data_b64="not base64!"),
            "attachment 'x' is not valid base64",
        ),
        (
            Attachment(name="k", mime="text/plain", path="~/.ssh/id_rsa"),
            "attachment 'k': ~/.ssh/id_rsa is in a protected location.",
        ),
        (
            Attachment(name="m", mime="text/plain", path="~/missing.txt"),
            "attachment 'm': ~/missing.txt is not a file",
        ),
        (
            Attachment(name="r", mime="text/plain", path="relative.txt"),
            "attachment 'r': 'relative.txt' is a relative path",
        ),
    ],
)
def test_bad_attachments_are_refused(home: Path, attachment: Attachment, message: str) -> None:
    with pytest.raises(AttachmentError) as error:
        save_attachments([attachment], guard())

    assert str(error.value).startswith(message)


def test_nothing_is_saved_when_one_attachment_is_bad(home: Path) -> None:
    good = Attachment(name="ok.png", mime="image/png", data_b64=PNG_B64)
    bad = Attachment(name="bad", mime="image/png", data_b64="@@@")

    with pytest.raises(AttachmentError):
        save_attachments([good, bad], guard())

    assert not files_dir().exists() or not any(files_dir().iterdir())


def test_too_big_attachment_is_refused(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attachments, "MAX_ATTACHMENT_BYTES", 10)

    with pytest.raises(AttachmentError, match="is 32 B; the limit is 10 B"):
        save_attachments([Attachment(name="big.png", mime="image/png", data_b64=PNG_B64)], guard())


# --- For the model -----------------------------------------------------------


def image_file(tmp_path: Path, data: bytes = PNG, name: str = "shot.png") -> FilePart:
    path = tmp_path / name
    path.write_bytes(data)
    return FilePart(name=name, path=str(path), mime="image/png")


def user(*parts: Any) -> Message:
    return Message(role="user", content=list(parts))


async def test_image_of_the_current_message_becomes_an_image(tmp_path: Path) -> None:
    shot = image_file(tmp_path)

    (message,) = await prepare_for_model([user(TextPart(text="look"), shot)], can_see_images=True)

    assert message.content == [TextPart(text="look"), ImagePart(mime="image/png", data_b64=PNG_B64)]


async def test_model_without_eyes_gets_a_note(tmp_path: Path) -> None:
    shot = image_file(tmp_path)

    (message,) = await prepare_for_model([user(shot)], can_see_images=False)

    assert message.content == [
        TextPart(text=f"[image shot.png at {shot.path}; this model cannot see images]")
    ]


async def test_older_images_and_all_files_become_notes(tmp_path: Path) -> None:
    old = image_file(tmp_path, name="old.png")
    new = image_file(tmp_path, name="new.png")
    report = FilePart(name="r.pdf", path="/x/r.pdf", mime="application/pdf")
    sent = FilePart(name="plan.md", path="/f/plan.md")
    messages = [
        Message(role="system", content=[TextPart(text="rules")]),
        user(TextPart(text="first"), old),
        Message(role="assistant", content=[TextPart(text="ok")]),
        Message(role="tool", content=[TextPart(text="Sent."), sent], tool_call_id="c1"),
        user(TextPart(text="second"), new, report),
    ]

    prepared = await prepare_for_model(messages, can_see_images=True)

    assert prepared[0] == messages[0]
    assert prepared[1].content[1] == TextPart(text=f"[image old.png at {old.path}]")
    assert prepared[3].content[1] == TextPart(text="[file plan.md at /f/plan.md]")
    assert isinstance(prepared[4].content[1], ImagePart)
    assert prepared[4].content[2] == TextPart(text="[file r.pdf at /x/r.pdf]")


async def test_huge_or_missing_or_fake_images_become_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attachments, "MAX_IMAGE_BYTES", 10)
    big = image_file(tmp_path, name="big.png")
    gone = FilePart(name="gone.png", path=str(tmp_path / "gone.png"), mime="image/png")
    fake = image_file(tmp_path, data=b"not an image", name="fake.png")

    (message,) = await prepare_for_model([user(big, gone, fake)], can_see_images=True)

    assert [part.text for part in message.content] == [  # type: ignore[union-attr]
        f"[image big.png at {big.path}; too big to show (32 B)]",
        f"[image gone.png at {gone.path}; the file is gone]",
        f"[image fake.png at {fake.path}]",
    ]


# --- Over the WebSocket ------------------------------------------------------


def connect(client: TestClient) -> Any:
    token = get_state(client.app).token  # type: ignore[arg-type]
    return client.websocket_connect(f"/ws?token={token}")


def test_attached_image_reaches_the_model_and_is_stored_as_a_path(home: Path) -> None:
    provider = FakeProvider([[TextDelta(text="A tiny picture.")]])
    with TestClient(create_app(Config(), provider)) as client, connect(client) as ws:
        ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
        ws.receive_json()
        ws.send_json(
            {
                "type": "message.send",
                "text": "what is this?",
                "source": "desktop",
                "attachments": [{"name": "shot.png", "mime": "image/png", "data_b64": PNG_B64}],
            }
        )
        session_id = None
        while (event := ws.receive_json())["type"] != "turn.finished":
            session_id = session_id or event.get("session", {}).get("id")
        while (event := ws.receive_json())["type"] != "session.updated":
            pass
        ws.send_json({"type": "history.get", "session_id": session_id})
        history = ws.receive_json()

    sent = provider.calls[0].messages[-1]
    assert sent.content == [
        TextPart(text="what is this?"),
        ImagePart(mime="image/png", data_b64=PNG_B64),
    ]
    stored = history["messages"][0]["content"]
    assert stored[0] == {"type": "text", "text": "what is this?"}
    assert (stored[1]["type"], stored[1]["name"], stored[1]["mime"]) == (
        "file",
        "shot.png",
        "image/png",
    )
    assert Path(stored[1]["path"]).read_bytes() == PNG
    with sqlite3.connect(db_path()) as raw:
        (content_json,) = raw.execute(
            "SELECT content_json FROM messages WHERE role = 'user'"
        ).fetchone()
    assert PNG_B64 not in content_json
    assert json.loads(content_json)[1]["type"] == "file"


def test_bad_attachment_is_refused_and_starts_nothing(home: Path) -> None:
    provider = FakeProvider([[TextDelta(text="never")]])
    with TestClient(create_app(Config(), provider)) as client, connect(client) as ws:
        ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
        ws.receive_json()
        ws.send_json(
            {
                "type": "message.send",
                "id": "m1",
                "text": "see",
                "source": "desktop",
                "attachments": [{"name": "k", "mime": "text/plain", "path": "~/.ssh/id_rsa"}],
            }
        )
        error = ws.receive_json()
        ws.send_json({"type": "session.list"})
        sessions = ws.receive_json()["sessions"]

    assert (error["type"], error["code"], error["reply_to"]) == ("error", "invalid_message", "m1")
    assert error["message"] == "attachment 'k': ~/.ssh/id_rsa is in a protected location."
    assert sessions == []
    assert provider.calls == []
