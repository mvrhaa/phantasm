"""Tests for zero.core.context."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from zero.core.context import PROMPTS_DIR, Computer, build_system_prompt
from zero.protocol.messages import Character, ClientInfo

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
PRAGUE = timezone(timedelta(hours=1), "CET")

CLIENT = ClientInfo(
    app="desktop",
    client_id="c-1",
    character=Character(
        name="Zero",
        persona="A curious fool who loves a good question.",
        poses=["idle", "sit", "wave"],
        moods=["neutral", "happy", "sad"],
    ),
    environment="macOS desktop app; replies show in a chat window that renders Markdown",
)

BARE = ClientInfo(app="cli", client_id="c-2")


def prompt(client: ClientInfo | None = CLIENT, source: str = "desktop", memories=()) -> str:
    return build_system_prompt(client, source, memories, now=NOW, tz=PRAGUE)


def test_base_rules_file_ships_with_the_package() -> None:
    text = (PROMPTS_DIR / "base.md").read_text(encoding="utf-8")

    assert "# Tools" in text
    assert "# Approvals" in text


def test_blocks_come_in_the_order_of_the_spec() -> None:
    text = prompt(memories=["writes Python", "lives in Prague"])

    markers = [
        "# Who you are",
        "# Character",
        "# Environment",
        "# Expression",
        "# What you remember about the user",
        "# Current time",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert text.startswith("# Who you are")


def test_persona_block_has_name_and_persona() -> None:
    text = prompt()

    assert "Your name is Zero." in text
    assert "A curious fool who loves a good question." in text


def test_environment_block_describes_the_client() -> None:
    assert "Client: macOS desktop app; replies show in a chat window" in prompt()


def test_phone_message_says_the_user_cannot_see_the_screen() -> None:
    from_phone = prompt(source="phone")
    from_desktop = prompt(source="desktop")

    assert "cannot see the computer screen" in from_phone
    assert "cannot see the computer screen" not in from_desktop


@pytest.mark.parametrize(
    ("source", "expected"), [("scheduler", "started by a schedule"), ("agent", "another agent")]
)
def test_other_sources_are_explained(source: str, expected: str) -> None:
    assert expected in prompt(source=source)


def test_expression_block_lists_poses_and_moods() -> None:
    text = prompt()

    assert "Poses: idle, sit, wave" in text
    assert "Moods: neutral, happy, sad" in text
    assert "set_expression" in text
    assert "instead of emoji" in text


def test_memories_are_listed_with_a_hint_to_search_the_rest() -> None:
    text = prompt(memories=["writes Python", "lives in Prague"])

    assert (
        "# What you remember about the user\n\n- writes Python\n- lives in Prague\n\n"
        "These are the memories used most and saved last; find the rest with memory_search."
    ) in text


def test_base_rules_say_what_may_be_remembered() -> None:
    text = prompt()

    assert "# Memory" in text
    assert "only when the user told it about themselves or asked you to remember it" in text


def test_memory_block_is_left_out_when_there_is_nothing_to_remember() -> None:
    assert "What you remember" not in prompt()


def test_time_block_is_last_and_in_the_users_zone() -> None:
    text = prompt(memories=["writes Python"])

    assert text.endswith("# Current time\n\nThursday, 2026-01-01 13:00, time zone UTC+01:00 (CET)")


def test_time_uses_the_computers_zone_by_default() -> None:
    text = build_system_prompt(CLIENT, "desktop", now=NOW)
    local = NOW.astimezone()

    assert local.strftime("%Y-%m-%d %H:%M") in text


def test_time_zone_without_a_name() -> None:
    text = build_system_prompt(None, "desktop", now=NOW, tz=timezone(timedelta(hours=-5)))

    assert text.endswith("Thursday, 2026-01-01 07:00, time zone UTC-05:00")


def test_client_without_character_gets_no_persona_or_expression() -> None:
    text = prompt(BARE)

    assert "# Character" not in text
    assert "# Expression" not in text
    assert "# Environment" not in text
    assert text.startswith("# Who you are")
    assert "# Current time" in text


def test_no_client_at_all_still_gives_rules_and_time() -> None:
    text = prompt(None, source="scheduler")

    assert text.startswith("# Who you are")
    assert "started by a schedule" in text
    assert "# Current time" in text


def test_empty_persona_keeps_only_the_name() -> None:
    client = ClientInfo(
        app="desktop",
        client_id="c-1",
        character=Character(name="Zero", poses=["idle"], moods=["neutral"]),
    )

    text = prompt(client)

    assert "Your name is Zero.\n\n# Expression" in text


def test_only_the_time_block_changes_between_turns() -> None:
    first = build_system_prompt(CLIENT, "desktop", ["writes Python"], now=NOW, tz=PRAGUE)
    later = build_system_prompt(
        CLIENT, "desktop", ["writes Python"], now=NOW + timedelta(hours=3), tz=PRAGUE
    )

    stable = first[: first.index("# Current time")]
    assert later.startswith(stable)
    assert first != later


def test_base_rules_prefer_the_file_tools_and_a_narrow_search() -> None:
    text = prompt()

    assert "# Looking at files" in text
    assert "list_dir, read_file and search_files" in text
    assert "Never scan the whole home folder" in text


def test_computer_block_names_home_own_code_and_access() -> None:
    computer = Computer(home=Path("/Users/u"), own_code=Path("/Users/u/zero"), allowed_roots=["~"])

    text = build_system_prompt(CLIENT, "desktop", now=NOW, tz=PRAGUE, computer=computer)

    assert "# Computer" in text
    assert "Home folder: /Users/u. Commands run there unless you give them another cwd." in text
    assert "Your own code (Zero, this brain): /Users/u/zero" in text
    assert "You may read and work with files in: ~" in text
    assert text.index("# Environment") < text.index("# Computer") < text.index("# Expression")


def test_computer_block_without_own_code() -> None:
    computer = Computer(home=Path("/Users/u"), own_code=None, allowed_roots=["~", "/Volumes/Data"])

    text = build_system_prompt(None, "desktop", now=NOW, tz=PRAGUE, computer=computer)

    assert "Your own code" not in text
    assert "You may read and work with files in: ~, /Volumes/Data" in text


def test_the_computer_is_found() -> None:
    computer = Computer.detect(allowed_roots=["~"])

    assert computer.home == Path.home()
    assert computer.own_code is not None
    assert (computer.own_code / "pyproject.toml").is_file()
    assert (computer.own_code / "src" / "zero").is_dir()
