"""Project memory: files in projects/, the tools that keep them and the prompt block."""

from datetime import datetime
from pathlib import Path

from phantasm.config import Config
from phantasm.core.cancellation import CancelToken
from phantasm.paths import home_dir
from phantasm.projects import MEMORY_PROMPT_CHARS, ProjectBook
from phantasm.providers.base import TextDelta, ToolCall
from phantasm.providers.fake import FakeProvider
from phantasm.tools.base import ToolContext, ToolRegistry, ToolResult
from phantasm.tools.builtin.projects import project_tools
from phantasm.tools.permissions import PathGuard
from tests.test_turn_text import connected, running, send, until_finished
from tests.turns import text_of

MEMORY = "Swift app. Run scripts/test.sh before every commit."


def book(tmp_path: Path) -> ProjectBook:
    return ProjectBook(tmp_path / "projects")


def saved(book: ProjectBook) -> ProjectBook:
    book.write("phantasm-app", path="~/phantasm-app", summary="The menu bar app", body=MEMORY)
    return book


async def call(book: ProjectBook, tool: str, **params: object) -> ToolResult:
    registry = ToolRegistry()
    for each in project_tools(book):
        registry.register(each)
    config = Config()
    ctx = ToolContext(
        session_id="s1",
        turn_id="t1",
        client_id=None,
        cancel=CancelToken(),
        config=config,
        paths=PathGuard.from_config(config.permissions),
    )
    return await registry.execute(ToolCall(id="c", name=tool, input=params), ctx)


def test_a_project_is_kept_as_a_markdown_file(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    text = (tmp_path / "projects" / "phantasm-app.md").read_text()
    assert text.startswith("---\nname: phantasm-app\npath: ~/phantasm-app\n")
    assert text.endswith(MEMORY + "\n")
    project = projects.get("phantasm-app")
    assert project is not None
    assert (project.summary, project.body) == ("The menu bar app", MEMORY)


def test_a_note_goes_under_notes_with_its_date(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    projects.note("phantasm-app", "No commit prefixes.", datetime(2026, 11, 20))
    projects.note("phantasm-app", "Answer in Russian.", datetime(2026, 11, 21))
    project = projects.get("phantasm-app")
    assert project is not None
    assert project.body.endswith(
        "## Notes\n- 2026-11-20: No commit prefixes.\n- 2026-11-21: Answer in Russian."
    )
    assert projects.note("unknown", "x", datetime(2026, 11, 20)) is None


def test_a_project_is_found_by_its_name_folder_or_spaced_name(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    for text in (
        "fix the phantasm-app build",
        "look at ~/phantasm-app/Sources",
        "what about Phantasm App?",
    ):
        assert [p.name for p in projects.mentioned([text])] == ["phantasm-app"], text
    for text in ("phantasm only", "my-phantasm-apps", "/tmp/x/phantasm-app2"):
        assert projects.mentioned([text]) == [], text


def test_the_most_recently_named_project_comes_first(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    projects.write("chess", path="~/chess_proj", summary="Pygame chess", body="Run Main.py")
    found = projects.mentioned(["about chess_proj", "now phantasm-app"])
    assert [p.name for p in found] == ["phantasm-app", "chess"]


def test_the_prompt_lists_projects_and_shows_the_named_one(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    projects.write("chess", path="~/chess_proj", summary="Pygame chess", body="Run Main.py")
    block = projects.prompt_block(["the phantasm-app tests fail"])
    assert block.startswith("# Projects\n")
    assert "- chess (~/chess_proj): Pygame chess" in block
    assert "- phantasm-app (~/phantasm-app): The menu bar app" in block
    assert f"# Project phantasm-app (~/phantasm-app)\n{MEMORY}" in block
    assert "Run Main.py" not in block


def test_a_long_memory_is_cut_in_the_prompt(tmp_path: Path) -> None:
    projects = book(tmp_path)
    projects.write("big", path="~/big", summary="Big", body="x" * (MEMORY_PROMPT_CHARS + 100))
    block = projects.prompt_block(["the big project"])
    assert "(read the rest with project_read)" in block
    assert "x" * (MEMORY_PROMPT_CHARS + 1) not in block


def test_files_with_bad_names_are_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "projects"
    folder.mkdir()
    (folder / "Bad Name.md").write_text("no frontmatter")
    (folder / "ok.md").write_text("plain memory")
    assert [p.name for p in book(tmp_path).all()] == ["ok"]


async def test_the_tools_write_note_and_read(tmp_path: Path) -> None:
    projects = book(tmp_path)
    written = await call(
        projects, "project_write", name="chess", path="~/chess_proj", summary="Chess", memory="M"
    )
    assert text_of(written) == "Saved the memory of chess."
    noted = await call(projects, "project_note", name="chess", note="Keep cattch_errors.")
    assert text_of(noted) == "Noted in chess."
    read = await call(projects, "project_read", name="chess")
    assert text_of(read).startswith("# chess (~/chess_proj)\nChess\n\nM\n\n## Notes\n- ")
    assert text_of(read).endswith(": Keep cattch_errors.")


async def test_the_tools_refuse_unknown_projects_and_bad_names(tmp_path: Path) -> None:
    projects = saved(book(tmp_path))
    missing = await call(projects, "project_read", name="nope")
    assert missing.is_error
    assert "Known projects: phantasm-app." in text_of(missing)
    assert (await call(projects, "project_note", name="nope", note="x")).is_error
    bad = await call(projects, "project_write", name="Bad Name", summary="s", memory="m")
    assert bad.is_error


def test_a_turn_sees_the_memory_of_the_project_the_user_names() -> None:
    ProjectBook(home_dir() / "projects").write(
        "phantasm-app", path="~/phantasm-app", summary="The menu bar app", body=MEMORY
    )
    provider = FakeProvider([[TextDelta(text="ok")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "why do the phantasm-app tests fail?")
        until_finished(ws)
    (model_call,) = provider.calls
    system = model_call.messages[0].content[0].text  # type: ignore[union-attr]
    assert "# Projects" in system
    assert f"# Project phantasm-app (~/phantasm-app)\n{MEMORY}" in system
