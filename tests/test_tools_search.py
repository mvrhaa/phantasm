"""Tests for the built-in search_files tool: the ripgrep branch and the Python one."""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.providers.base import ToolCall
from phantasm.tools.base import CANCELLED, ToolContext, ToolResult
from phantasm.tools.builtin import builtin_registry, search
from phantasm.tools.permissions import PathGuard
from tests.turns import text_of


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def project(home: Path) -> Path:
    root = home / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "import os\n# TODO: tests\ndef main():\n    pass  # todo\n"
    )
    (root / "src" / "util.py").write_text("x = 1  # TODO later\n")
    (root / "README.md").write_text("Nothing to do here.\nTODO: write docs\n")
    (root / ".secret.txt").write_text("TODO hidden\n")
    (root / "image.bin").write_bytes(b"TODO\x00\x01")
    return root


def context(cancel: CancelToken | None = None) -> ToolContext:
    config = Config()
    return ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id="c1",
        cancel=cancel or CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )


async def search_for(
    pattern: str, path: str | Path, cancel: CancelToken | None = None, **extra: Any
) -> ToolResult:
    tool_input = {"pattern": pattern, "path": str(path), **extra}
    call = ToolCall(id="c", name="search_files", input=tool_input)
    return await builtin_registry().execute(call, context(cancel))


@pytest.fixture
def python_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ripgrep is not installed."""
    monkeypatch.setattr(search, "find_rg", lambda: None)


pytestmark = pytest.mark.usefixtures("home")


@pytest.mark.usefixtures("python_only")
async def test_matches_come_as_path_line_text_in_path_order(project: Path) -> None:
    result = await search_for("TODO", "~/project")

    assert text_of(result) == "\n".join(
        [
            "README.md:2: TODO: write docs",
            "src/app.py:2: # TODO: tests",
            "src/util.py:1: x = 1  # TODO later",
        ]
    )
    assert (result.summary, result.is_error) == ("3 matches", False)


@pytest.mark.usefixtures("python_only")
async def test_one_match_is_singular(project: Path) -> None:
    assert (await search_for("def main", "~/project")).summary == "1 match"


@pytest.mark.usefixtures("python_only")
async def test_no_matches(project: Path) -> None:
    result = await search_for("nowhere to be found", "~/project")

    assert (text_of(result), result.summary, result.is_error) == ("No matches.", "0 matches", False)


@pytest.mark.usefixtures("python_only")
async def test_glob_narrows_the_files(project: Path) -> None:
    result = await search_for("TODO", "~/project", glob="*.py")

    assert [line.split(":")[0] for line in text_of(result).splitlines()] == [
        "src/app.py",
        "src/util.py",
    ]


@pytest.mark.usefixtures("python_only")
async def test_pattern_is_a_regex(project: Path) -> None:
    result = await search_for(r"(?i)\btodo\b", "~/project/src")

    assert text_of(result).splitlines() == [
        "app.py:2: # TODO: tests",
        "app.py:4:     pass  # todo",
        "util.py:1: x = 1  # TODO later",
    ]


@pytest.mark.usefixtures("python_only")
async def test_one_file_can_be_searched(project: Path) -> None:
    result = await search_for("TODO", "~/project/README.md")

    assert text_of(result) == "README.md:2: TODO: write docs"


@pytest.mark.usefixtures("python_only")
async def test_symlinks_are_not_followed(project: Path, home: Path) -> None:
    outside = home.parent / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("TODO outside\n")
    (project / "linked").symlink_to(outside)
    (project / "leak.txt").symlink_to(outside / "leak.txt")

    assert "outside" not in text_of(await search_for("TODO", "~/project"))


@pytest.mark.usefixtures("python_only")
async def test_protected_folders_are_not_searched(home: Path) -> None:
    keychains = home / "Library" / "Keychains"
    keychains.mkdir(parents=True)
    (keychains / "notes.txt").write_text("password TODO\n")
    (home / "plan.txt").write_text("TODO plan\n")

    assert text_of(await search_for("TODO", "~")) == "plan.txt:1: TODO plan"


@pytest.mark.usefixtures("python_only")
async def test_long_matching_line_is_cut(project: Path) -> None:
    (project / "min.js").write_text("TODO" + "x" * 1000 + "\n")

    line = text_of(await search_for("TODO", "~/project/min.js"))

    assert line == "min.js:1: TODO" + "x" * (search.MAX_MATCH_CHARS - 4) + "…"


@pytest.mark.usefixtures("python_only")
async def test_search_stops_at_the_match_limit(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "MAX_MATCHES", 2)

    result = await search_for("TODO", "~/project")

    assert text_of(result).splitlines()[-1] == (
        "[… stopped after 2 matches; narrow the pattern, path or glob]"
    )
    assert result.summary == "2 matches"


@pytest.mark.usefixtures("python_only")
async def test_search_stops_when_time_is_up(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "SEARCH_TIMEOUT_S", 0)

    result = await search_for("TODO", "~/project")

    assert text_of(result) == "No matches.\n[… search stopped after 0 s; narrow the path or glob]"


@pytest.mark.usefixtures("python_only")
async def test_invalid_regex_is_an_error(project: Path) -> None:
    result = await search_for("(unclosed", "~/project")

    assert result.is_error
    assert text_of(result).startswith("Invalid pattern: missing ), unterminated subpattern")


@pytest.mark.usefixtures("python_only")
async def test_cancelled_turn_stops_the_search(project: Path) -> None:
    cancel = CancelToken()
    cancel.cancel()

    assert text_of(await search_for("TODO", "~/project", cancel)) == CANCELLED


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("~/missing", "~/missing does not exist."),
        ("relative", "'relative' is a relative path"),
        ("/etc", "/private/etc is outside the folders tools may use"),
    ],
)
async def test_bad_paths_are_errors(path: str, message: str) -> None:
    result = await search_for("x", path)

    assert result.is_error
    assert text_of(result).startswith(message)


async def test_empty_pattern_is_invalid_input() -> None:
    result = await search_for("", "~")

    assert result.is_error
    assert "pattern: String should have at least 1 character" in text_of(result)


class FakeRg:
    """A stand-in for ``rg``: prints the given JSON lines and remembers its arguments."""

    def __init__(self, folder: Path) -> None:
        self.script = folder / "fake-rg"
        self.args_file = folder / "rg-args.json"

    def write(
        self,
        events: list[dict[str, Any]],
        *,
        stderr: str = "",
        code: int = 0,
        sleep_s: float = 0,
    ) -> str:
        lines = [json.dumps(event) for event in events]
        self.script.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, time\n"
            f"json.dump(sys.argv[1:], open({str(self.args_file)!r}, 'w'))\n"
            f"for line in {lines!r}:\n"
            "    print(line, flush=True)\n"
            f"sys.stderr.write({stderr!r})\n"
            f"time.sleep({sleep_s})\n"
            f"sys.exit({code})\n"
        )
        self.script.chmod(0o755)
        return str(self.script)

    @property
    def args(self) -> list[str]:
        return json.loads(self.args_file.read_text())


def match(path: str | Path, line: int, text: str) -> dict[str, Any]:
    return {
        "type": "match",
        "data": {"path": {"text": str(path)}, "lines": {"text": text}, "line_number": line},
    }


@pytest.fixture
def fake_rg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRg:
    fake = FakeRg(tmp_path.resolve())
    monkeypatch.setattr(search, "find_rg", lambda: str(fake.script))
    return fake


async def test_rg_is_called_with_safe_arguments(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write([], code=1)

    await search_for("-TODO", "~/project", glob="*.py")

    assert fake_rg.args == [
        "--json",
        "--no-config",
        "--no-messages",
        "--sort",
        "path",
        "--max-filesize",
        str(search.MAX_SCAN_BYTES),
        "-e",
        "-TODO",
        "--glob",
        "*.py",
        "--",
        str(project),
    ]


async def test_rg_matches_are_shown_relative_to_the_folder(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write(
        [
            {"type": "begin", "data": {"path": {"text": str(project / "README.md")}}},
            match(project / "README.md", 2, "TODO: write docs\n"),
            match(project / "src" / "app.py", 2, "# TODO: tests\n"),
            {"type": "summary", "data": {}},
        ]
    )

    result = await search_for("TODO", "~/project")

    assert text_of(result) == "README.md:2: TODO: write docs\nsrc/app.py:2: # TODO: tests"
    assert result.summary == "2 matches"


async def test_rg_no_matches(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write([], code=1)

    assert text_of(await search_for("TODO", "~/project")) == "No matches."


async def test_rg_matches_in_protected_paths_are_dropped(fake_rg: FakeRg, home: Path) -> None:
    fake_rg.write(
        [
            match(home / "Library" / "Keychains" / "notes.txt", 1, "password TODO\n"),
            match(home / "plan.txt", 1, "TODO plan\n"),
        ]
    )

    assert text_of(await search_for("TODO", "~")) == "plan.txt:1: TODO plan"


async def test_rg_names_that_are_not_utf8_are_skipped(fake_rg: FakeRg, project: Path) -> None:
    raw = {
        "type": "match",
        "data": {"path": {"bytes": "/w=="}, "lines": {"text": "TODO\n"}, "line_number": 1},
    }
    fake_rg.write([raw, match(project / "a.txt", 1, "TODO\n")])

    assert text_of(await search_for("TODO", "~/project")) == "a.txt:1: TODO"


async def test_rg_error_is_reported(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write([], stderr="rg: regex parse error:\n    (\nerror: unclosed group\n", code=2)

    result = await search_for("(", "~/project")

    assert result.is_error
    assert text_of(result) == "Search failed: rg: regex parse error:\n    (\nerror: unclosed group"


async def test_rg_matches_survive_a_partial_error(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write([match(project / "a.txt", 1, "TODO\n")], stderr="some file failed", code=2)

    result = await search_for("TODO", "~/project")

    assert (text_of(result), result.is_error) == ("a.txt:1: TODO", False)


async def test_rg_is_stopped_at_the_match_limit(
    fake_rg: FakeRg, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "MAX_MATCHES", 2)
    fake_rg.write([match(project / "a.txt", n, "TODO\n") for n in range(1, 6)], sleep_s=30)

    async with asyncio.timeout(10):
        result = await search_for("TODO", "~/project")

    assert text_of(result).splitlines() == [
        "a.txt:1: TODO",
        "a.txt:2: TODO",
        "[… stopped after 2 matches; narrow the pattern, path or glob]",
    ]


async def test_rg_is_stopped_when_time_is_up(
    fake_rg: FakeRg, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "SEARCH_TIMEOUT_S", 2)
    fake_rg.write([match(project / "a.txt", 1, "TODO\n")], sleep_s=30)

    async with asyncio.timeout(10):
        result = await search_for("TODO", "~/project")

    assert text_of(result) == (
        "a.txt:1: TODO\n[… search stopped after 2 s; narrow the path or glob]"
    )


async def test_cancel_stops_rg_at_once(fake_rg: FakeRg, project: Path) -> None:
    fake_rg.write([], sleep_s=30)
    cancel = CancelToken()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.3)
        cancel.cancel()

    async with asyncio.timeout(10):
        result, _ = await asyncio.gather(search_for("TODO", "~/project", cancel), cancel_soon())

    assert text_of(result) == CANCELLED


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
async def test_real_rg_finds_what_python_finds(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_rg = await search_for("TODO", "~/project")
    monkeypatch.setattr(search, "find_rg", lambda: None)
    with_python = await search_for("TODO", "~/project")

    assert text_of(with_rg) == text_of(with_python)
    assert with_rg.summary == "3 matches"
