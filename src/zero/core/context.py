"""Assembling what is sent to the model: the system prompt and the history."""

from collections.abc import Sequence
from datetime import datetime, tzinfo
from pathlib import Path

from zero.protocol.messages import Character, ClientInfo

PROMPTS_DIR = Path(__file__).parent / "prompts"

_SOURCE_NOTES = {
    "phone": (
        "The user is writing from their phone and cannot see the computer screen. "
        "Describe results in words instead of pointing at something on the screen."
    ),
    "scheduler": "This turn was started by a schedule, not by a message the user just sent.",
    "agent": "This turn was started by another agent, not directly by the user.",
}


def _base_rules() -> str:
    return (PROMPTS_DIR / "base.md").read_text(encoding="utf-8").strip()


def _persona(character: Character) -> str:
    lines = ["# Character", "", f"Your name is {character.name}."]
    if character.persona.strip():
        lines += ["", character.persona.strip()]
    return "\n".join(lines)


def _environment(client: ClientInfo | None, source: str) -> str | None:
    lines = []
    if client is not None and client.environment.strip():
        lines.append(f"Client: {client.environment.strip()}")
    if source in _SOURCE_NOTES:
        lines.append(_SOURCE_NOTES[source])
    if not lines:
        return None
    return "\n".join(["# Environment", "", *lines])


def _expression(character: Character) -> str:
    return "\n".join(
        [
            "# Expression",
            "",
            "Your character has a body. Show how you feel with the set_expression tool "
            "instead of emoji or actions written between asterisks.",
            "",
            f"Poses: {', '.join(character.poses)}",
            f"Moods: {', '.join(character.moods)}",
            "",
            "Pick the pose and mood that fit your reply. You may change them during a turn.",
        ]
    )


def _memories(memories: Sequence[str]) -> str:
    return "\n".join(
        ["# What you remember about the user", "", *(f"- {memory}" for memory in memories)]
    )


def _time(now: datetime, tz: tzinfo | None) -> str:
    local = now.astimezone(tz)
    offset = local.strftime("%z")
    zone = f"UTC{offset[:3]}:{offset[3:]}"
    name = local.tzname()
    if name and name != zone and not name.startswith(("+", "-")):
        zone = f"{zone} ({name})"
    return "\n".join(
        [
            "# Current time",
            "",
            f"{local.strftime('%A, %Y-%m-%d %H:%M')}, time zone {zone}",
        ]
    )


def build_system_prompt(
    client: ClientInfo | None,
    source: str,
    memories: Sequence[str] = (),
    *,
    now: datetime,
    tz: tzinfo | None = None,
) -> str:
    """Build the system prompt from blocks in a fixed order.

    Stable blocks come first so providers can cache the prefix: base rules,
    persona, environment, expression rules, memory, and the current time last
    because it always changes. Blocks with nothing to say are left out.
    ``tz`` is the user's time zone; the computer's own zone is used by default.
    """
    character = client.character if client is not None else None
    blocks = [
        _base_rules(),
        _persona(character) if character else None,
        _environment(client, source),
        _expression(character) if character else None,
        _memories(memories) if memories else None,
        _time(now, tz),
    ]
    return "\n\n".join(block for block in blocks if block)
