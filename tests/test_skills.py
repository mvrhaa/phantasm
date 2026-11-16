"""Skills: SKILL.md folders, the prompt block, skill_use and /name from the user."""

from pathlib import Path

from tests.test_turn_text import connected, running, send, until_finished
from zero.config import Config
from zero.core.cancellation import CancelToken
from zero.paths import home_dir
from zero.providers.base import TextDelta
from zero.providers.fake import FakeProvider
from zero.skills import SkillBook, expand, parse, user_start
from zero.tools.base import ToolContext
from zero.tools.builtin.skills import SkillUse, SkillUseInput
from zero.tools.permissions import PathGuard


def write_skill(folder: Path, name: str, text: str, *files: str) -> Path:
    skill = folder / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    for file in files:
        (skill / file).write_text("x", encoding="utf-8")
    return skill


REVIEW = """---
name: code-review
description: >
  Review a diff for bugs.
  Use when asked to review.
argument-hint: "[file]"
---
Review $ARGUMENTS carefully, then $0 first. Notes in ${SKILL_DIR}/notes.md.
"""


def test_parse_reads_plain_quoted_and_folded_fields() -> None:
    fields, body = parse(REVIEW)
    assert fields["name"] == "code-review"
    assert fields["description"] == "Review a diff for bugs. Use when asked to review."
    assert fields["argument-hint"] == "[file]"
    assert body.startswith("Review $ARGUMENTS")
    assert parse("Just text") == ({}, "Just text")


def test_book_lists_valid_skills_and_skips_bad_ones(tmp_path: Path) -> None:
    write_skill(tmp_path, "code-review", REVIEW, "notes.md")
    write_skill(tmp_path, "plain", "Say hello in three languages.\nMore lines.")
    write_skill(tmp_path, "Bad Name", "---\nname: Bad Name\n---\nx")
    write_skill(tmp_path, "hidden", "---\ndisable-model-invocation: true\n---\nOnly by hand.")
    book = SkillBook(tmp_path)
    assert [skill.name for skill in book.all()] == ["code-review", "hidden", "plain"]
    assert book.get("plain").description == "Say hello in three languages."  # type: ignore[union-attr]
    block = book.prompt_block()
    assert block is not None and block.startswith("# Skills")
    assert "- code-review: Review a diff for bugs." in block
    assert "hidden" not in block
    assert SkillBook(tmp_path / "missing").prompt_block() is None


def test_expand_fills_the_placeholders(tmp_path: Path) -> None:
    folder = write_skill(tmp_path, "code-review", REVIEW)
    skill = SkillBook(tmp_path).get("code-review")
    assert skill is not None
    text = expand(skill, "a.py b.py")
    assert text.startswith("Review a.py b.py carefully, then a.py first.")
    assert f"{folder}/notes.md" in text
    write_skill(tmp_path, "plain", "Do it.")
    plain = SkillBook(tmp_path).get("plain")
    assert plain is not None and expand(plain, "now") == "Do it.\n\nARGUMENTS: now"


def test_user_start_finds_the_skill_and_its_arguments(tmp_path: Path) -> None:
    write_skill(tmp_path, "code-review", REVIEW)
    write_skill(tmp_path, "auto", "---\nuser-invocable: false\n---\nx")
    book = SkillBook(tmp_path)
    found = user_start("/code-review src/a.py", book)
    assert found is not None and found[0].name == "code-review" and found[1] == "src/a.py"
    assert user_start("/auto", book) is None
    assert user_start("/missing x", book) is None
    assert user_start("say /code-review", book) is None


async def test_skill_use_gives_the_instructions_and_files(tmp_path: Path) -> None:
    write_skill(tmp_path, "code-review", REVIEW, "notes.md")
    tool = SkillUse(SkillBook(tmp_path))
    context = ToolContext(
        session_id="s",
        turn_id="t",
        client_id=None,
        cancel=CancelToken(),
        config=Config(),
        paths=PathGuard.from_config(Config().permissions),
    )
    result = await tool.run(SkillUseInput(name="code-review", arguments="a.py"), context)
    text = result.content[0].text  # type: ignore[union-attr]
    assert "Review a.py carefully" in text and "Files of this skill: notes.md" in text
    missing = await tool.run(SkillUseInput(name="nope"), context)
    assert missing.is_error and "Skills: code-review" in missing.content[0].text  # type: ignore[union-attr]
    assert str(tmp_path) in tool.description


def test_skills_reach_the_prompt_and_a_slash_message_runs_one() -> None:
    write_skill(home_dir() / "skills", "greet", "---\ndescription: Greet warmly\n---\nGreet in $0.")
    provider = FakeProvider([[TextDelta(text="Hola")], [TextDelta(text="title")]])
    with running(provider) as (client, state), connected(client, state) as ws:
        send(ws, "/greet Spanish")
        until_finished(ws)
    system = provider.calls[0].messages[0].content[0].text  # type: ignore[union-attr]
    assert "# Skills" in system and "- greet: Greet warmly" in system
    assert "# Command\nGreet in Spanish." in system
    assert "skill_use" in {tool.name for tool in provider.calls[0].tools}
