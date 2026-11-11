"""Tests for the built-in file tools: read_file and list_dir."""

import base64
import socket
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.turns import text_of
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.protocol.messages import ImagePart
from zero.providers.base import TextDelta, ToolCall, ToolCallEvent
from zero.providers.fake import FakeProvider
from zero.server.app import create_app
from zero.server.state import get_state
from zero.tools import builtin
from zero.tools.base import NEEDS_APPROVAL, ToolContext, ToolResult
from zero.tools.builtin import builtin_registry
from zero.tools.builtin.files import (
    MAX_LINE_CHARS,
    ListDirInput,
    ReadFileInput,
    human_size,
    looks_secret,
)
from zero.tools.permissions import PathGuard

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder that ``~`` points at; the sandbox is the default one."""
    home = tmp_path.resolve() / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
    monkeypatch.setenv("HOME", str(home))
    return home


def context() -> ToolContext:
    config = Config()
    return ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )


async def call(tool: str, **tool_input: Any) -> ToolResult:
    return await builtin_registry().execute(
        ToolCall(id="c", name=tool, input=tool_input), context()
    )


async def read(path: str | Path, **options: Any) -> ToolResult:
    return await call("read_file", path=str(path), **options)


async def listing(path: str | Path, **options: Any) -> ToolResult:
    return await call("list_dir", path=str(path), **options)


def lines_of(count: int) -> str:
    return "".join(f"line {n}\n" for n in range(1, count + 1))


# --- Helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "shown"),
    [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (1536, "1.5 KB"), (5 * 1024**2, "5.0 MB")],
)
def test_human_size(size: int, shown: str) -> None:
    assert human_size(size) == shown


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        (".env", True),
        (".env.local", True),
        (".ENV", True),
        ("id_rsa", True),
        ("id_ed25519.pub", True),
        ("server.pem", True),
        ("Cert.PEM", True),
        ("environment.txt", False),
        ("notes.md", False),
        ("my_id_rsa_notes.txt", False),
        ("pem.txt", False),
    ],
)
def test_looks_secret(name: str, secret: bool) -> None:
    assert looks_secret(name) is secret


def test_builtin_tools_are_registered_and_allowed() -> None:
    registry = builtin_registry()

    assert [spec.name for spec in registry.specs()] == [
        "read_file",
        "list_dir",
        "search_files",
        "write_file",
        "edit_file",
        "run_command",
        "web_fetch",
        "send_file",
    ]
    assert {registry.get(name).default_policy for name in ("read_file", "list_dir")} == {"allow"}  # type: ignore[union-attr]
    assert builtin.files.ReadFile().label(ReadFileInput(path="~/a.md")) == "~/a.md"
    assert builtin.files.ListDir().label(ListDirInput(path="~/src")) == "~/src"


# --- read_file: text ---------------------------------------------------------


async def test_text_comes_with_line_numbers(home: Path) -> None:
    (home / "notes.md").write_text("first\nsecond\n\nfourth")

    result = await read("~/notes.md")

    assert text_of(result) == "     1\tfirst\n     2\tsecond\n     3\t\n     4\tfourth"
    assert (result.summary, result.is_error) == ("4 lines", False)


async def test_offset_and_limit_pick_a_page(home: Path) -> None:
    (home / "log.txt").write_text(lines_of(10))

    result = await read("~/log.txt", offset=3, limit=2)

    assert text_of(result) == (
        "     3\tline 3\n     4\tline 4\n[… more lines follow; read on with offset=5]"
    )


async def test_last_page_has_no_note(home: Path) -> None:
    (home / "log.txt").write_text(lines_of(10))

    result = await read("~/log.txt", offset=9)

    assert text_of(result) == "     9\tline 9\n    10\tline 10"


async def test_offset_past_the_end_is_an_error(home: Path) -> None:
    (home / "log.txt").write_text(lines_of(3))

    result = await read("~/log.txt", offset=7)

    assert result.is_error
    assert text_of(result) == "~/log.txt has only 3 lines; offset 7 is past the end."


async def test_empty_file(home: Path) -> None:
    (home / "empty.txt").write_text("")

    result = await read("~/empty.txt")

    assert (text_of(result), result.summary, result.is_error) == ("(empty file)", "empty", False)


async def test_windows_line_endings_are_dropped(home: Path) -> None:
    (home / "crlf.txt").write_bytes(b"a\r\nb\r\n")

    assert text_of(await read("~/crlf.txt")) == "     1\ta\n     2\tb"


async def test_bytes_that_are_not_utf8_are_replaced(home: Path) -> None:
    (home / "latin1.txt").write_bytes("café".encode("latin-1"))

    result = await read("~/latin1.txt")

    assert text_of(result) == "     1\tcaf�"


async def test_very_long_line_is_cut(home: Path) -> None:
    (home / "min.js").write_text("x" * (MAX_LINE_CHARS + 500))

    text = text_of(await read("~/min.js"))

    assert text == f"     1\t{'x' * MAX_LINE_CHARS} [… line cut]"


async def test_big_file_stops_before_the_output_limit(home: Path) -> None:
    (home / "big.txt").write_text("".join(f"{'y' * 60} {n}\n" for n in range(1, 1501)))

    text = text_of(await read("~/big.txt"))

    assert len(text) < 30_000
    assert "truncated" not in text
    assert text.endswith("]") and "read on with offset=" in text


# --- read_file: images and binaries ------------------------------------------


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 16, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8, "image/webp"),
    ],
)
async def test_images_come_back_as_images(home: Path, data: bytes, mime: str) -> None:
    (home / "picture").write_bytes(data)

    result = await read("~/picture")

    assert result.content == [ImagePart(mime=mime, data_b64=base64.b64encode(data).decode())]
    assert result.summary == f"image, {len(data)} B"


async def test_too_big_image_is_an_error(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtin.files, "MAX_IMAGE_BYTES", 16)
    (home / "photo.png").write_bytes(PNG)

    result = await read("~/photo.png")

    assert result.is_error
    assert text_of(result) == "~/photo.png is an image of 32 B; the limit is 16 B."


async def test_binary_file_is_an_error(home: Path) -> None:
    (home / "app.bin").write_bytes(b"\x7fELF\x00\x01\x02")

    result = await read("~/app.bin")

    assert result.is_error
    assert text_of(result) == "~/app.bin is a binary file and cannot be read as text."


# --- read_file: what cannot be read ------------------------------------------


async def test_missing_file(home: Path) -> None:
    assert text_of(await read("~/nope.txt")) == "~/nope.txt does not exist."


async def test_folder_points_to_list_dir(home: Path) -> None:
    assert text_of(await read("~")) == "~ is a folder; use list_dir to see what is in it."


async def test_special_file_is_refused(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A socket stands in for every file that is not a regular one. A FIFO is the
    # real danger (opening it waits for a writer forever), but a test with one
    # would hang instead of failing if the check were ever lost.
    monkeypatch.chdir(home)  # a short relative name keeps under the socket path limit
    with socket.socket(socket.AF_UNIX) as server:
        server.bind("agent.sock")

        result = await read("~/agent.sock")

    assert text_of(result) == "~/agent.sock is not a regular file."


async def test_unreadable_file(home: Path) -> None:
    locked = home / "locked.txt"
    locked.write_text("x")
    locked.chmod(0)
    try:
        result = await read("~/locked.txt")
    finally:
        locked.chmod(0o600)

    assert result.is_error
    assert text_of(result) == "Cannot read ~/locked.txt: Permission denied."


async def test_sandbox_refuses_paths_outside_and_relative_ones(home: Path) -> None:
    (home.parent / "outside.txt").write_text("secret")

    outside = await read(home.parent / "outside.txt")
    relative = await read("notes.md")

    assert outside.is_error and "outside the folders tools may use" in text_of(outside)
    assert relative.is_error and "relative path" in text_of(relative)


async def test_protected_folder_is_refused_without_asking(home: Path) -> None:
    result = await read("~/.ssh/id_rsa")

    assert text_of(result) == "~/.ssh/id_rsa is in a protected location."


# --- read_file: files that may hold secrets ----------------------------------


@pytest.mark.parametrize("name", [".env", ".env.production", "id_ed25519", "server.pem", ".ENV"])
async def test_secret_looking_file_needs_approval(home: Path, name: str) -> None:
    (home / "app").mkdir()
    (home / "app" / name).write_text("API_KEY=sk-123")

    result = await read(f"~/app/{name}")

    assert result.is_error
    assert text_of(result) == f"~/app/{name} may hold secrets, {NEEDS_APPROVAL}"
    assert "sk-123" not in text_of(result)


async def test_innocent_name_linking_to_a_secret_needs_approval(home: Path) -> None:
    (home / ".env").write_text("API_KEY=sk-123")
    (home / "notes.txt").symlink_to(home / ".env")

    result = await read("~/notes.txt")

    assert text_of(result) == f"~/notes.txt may hold secrets, {NEEDS_APPROVAL}"


async def test_secret_name_linking_to_an_ordinary_file_needs_approval(home: Path) -> None:
    (home / "plain.txt").write_text("hello")
    (home / ".env").symlink_to(home / "plain.txt")

    assert (await read("~/.env")).is_error


# --- list_dir ----------------------------------------------------------------


@pytest.fixture
def project(home: Path) -> Path:
    root = home / "project"
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "main.py").write_text("print()")
    (root / "docs").mkdir()
    (root / "README.md").write_text("x" * 2048)
    (root / "a.txt").write_text("abc")
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.pyc\n")
    return root


async def test_list_dir_shows_one_level_with_sizes(project: Path) -> None:
    result = await listing("~/project")

    assert text_of(result) == "\n".join(
        ["~/project/", "  docs/", "  src/", "  a.txt  3 B", "  README.md  2.0 KB"]
    )
    assert result.summary == "4 entries"


async def test_list_dir_goes_deeper_with_depth(project: Path) -> None:
    result = await listing("~/project/", depth=3)

    assert text_of(result) == "\n".join(
        [
            "~/project/",
            "  docs/",
            "  src/",
            "    app/",
            "      main.py  7 B",
            "  a.txt  3 B",
            "  README.md  2.0 KB",
        ]
    )


async def test_list_dir_shows_hidden_names_on_request(project: Path) -> None:
    text = text_of(await listing("~/project", show_hidden=True))

    assert "  .git/" in text.splitlines()
    assert "  .gitignore  6 B" in text.splitlines()


async def test_list_dir_does_not_open_protected_folders(home: Path) -> None:
    text = text_of(await listing("~", depth=2, show_hidden=True))

    assert "  .ssh/  (protected)" in text.splitlines()
    assert "id_rsa" not in text


async def test_list_dir_shows_symlinks_without_following_them(home: Path) -> None:
    (home / "elsewhere").symlink_to("/usr")

    text = text_of(await listing("~", depth=3))

    assert "  elsewhere -> /usr" in text.splitlines()
    assert "bin" not in text


async def test_list_dir_of_an_empty_folder(home: Path) -> None:
    (home / "empty").mkdir()

    assert text_of(await listing("~/empty")) == "~/empty/\n  (empty)"


async def test_list_dir_stops_after_the_entry_limit(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin.files, "MAX_LIST_ENTRIES", 3)

    result = await listing("~/project", depth=3)

    assert text_of(result).splitlines() == [
        "~/project/",
        "  docs/",
        "  src/",
        "    app/",
        "[… stopped after 3 entries; list a subfolder]",
    ]
    assert result.summary == "3 entries"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("~/nope", "~/nope does not exist."),
        ("~/project/a.txt", "~/project/a.txt is a file; use read_file to read it."),
        ("~/.ssh", "~/.ssh is in a protected location."),
    ],
)
async def test_list_dir_errors(project: Path, path: str, message: str) -> None:
    result = await listing(path)

    assert result.is_error
    assert text_of(result) == message


async def test_list_dir_depth_is_limited(project: Path) -> None:
    result = await listing("~/project", depth=6)

    assert result.is_error
    assert "depth: Input should be less than or equal to 5" in text_of(result)


# --- In a turn ---------------------------------------------------------------


def test_model_reads_a_file_through_a_turn(home: Path) -> None:
    (home / "notes.md").write_text("buy milk\n")
    provider = FakeProvider(
        [
            [ToolCallEvent(call=ToolCall(id="r1", name="read_file", input={"path": "~/notes.md"}))],
            [TextDelta(text="You need milk.")],
        ]
    )
    with TestClient(create_app(Config(), provider)) as client:
        token = get_state(client.app).token  # type: ignore[arg-type]
        with client.websocket_connect(f"/ws?token={token}") as ws:
            ws.send_json({"type": "hello", "protocol": 1, "client": {"app": "t", "client_id": "c"}})
            ws.receive_json()
            ws.send_json({"type": "message.send", "text": "what to buy?", "source": "desktop"})
            events = [ws.receive_json()]
            while events[-1]["type"] != "turn.finished":
                events.append(ws.receive_json())

    result = provider.calls[1].messages[-1]
    assert (result.role, result.is_error, text_of(result)) == ("tool", False, "     1\tbuy milk")
    finished = [e for e in events if e["type"] == "tool.finished"]
    assert [e["status"] for e in finished] == ["ok"]
    assert events[-1]["text"] == "You need milk."
