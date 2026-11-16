"""Assembling what is sent to the model: the system prompt and the history."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path

import zero
from zero.protocol.messages import (
    Character,
    ClientInfo,
    ContentPart,
    FilePart,
    ImagePart,
    TextPart,
)
from zero.providers.base import Message, ToolCall
from zero.store.messages import StoredMessage

PROMPTS_DIR = Path(__file__).parent / "prompts"

IMAGE_OMITTED = "[image omitted]"
NO_ANSWER = {
    "cancelled": "[no answer: the user cancelled this turn]",
    "error": "[no answer: this turn failed]",
}
CHARS_PER_TOKEN = 3.5
OTHER_CHARS_PER_TOKEN = 2.5
IMAGE_TOKENS = 1000
# Past this share of the history budget, old tool results give way to a short note.
CLEAR_AT = 0.5
CLEAR_MIN_CHARS = 400
CLEARED = "[old result of {tool} cleared to save space; call it again if you need it]"

CountFn = Callable[[Message], int]

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


@dataclass(frozen=True)
class Computer:
    """Facts about the computer the model needs to find its way around."""

    home: Path
    own_code: Path | None
    allowed_roots: Sequence[str]

    @classmethod
    def detect(cls, allowed_roots: Sequence[str]) -> "Computer":
        """This computer: the home folder and the checkout Zero runs from, if it runs from one."""
        root = Path(zero.__file__).resolve().parents[2]
        own_code = root if (root / "pyproject.toml").is_file() else None
        return cls(home=Path.home(), own_code=own_code, allowed_roots=list(allowed_roots))


def _computer(computer: Computer) -> str:
    lines = [
        "# Computer",
        "",
        f"- Home folder: {computer.home}. Commands run there unless you give them another cwd.",
    ]
    if computer.own_code is not None:
        lines.append(f"- Your own code (Zero, this brain): {computer.own_code}")
    roots = ", ".join(computer.allowed_roots)
    lines.append(f"- You may read and work with files in: {roots}")
    return "\n".join(lines)


def _agent_role(name: str) -> str:
    return "\n".join(
        [
            "# Your role",
            "",
            f"You are {name}, an agent: a copy of Zero that the main conversation started to "
            "do one task on its own. The task is the first message; you see nothing else of "
            "that conversation. Work on it with your tools until it is done. Nobody watches "
            "you work, but the user may be asked to approve risky steps.",
            "",
            "If you need something only that conversation or the user knows, ask with "
            "ask_parent and end your turn: the answer comes as your next message.",
            "",
            "When you are done, reply with a short report of what you did and found. That "
            "reply is sent back to the conversation that started you.",
        ]
    )


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


MEMORY_HINT = "These are the memories used most and saved last; find the rest with memory_search."


def _memories(memories: Sequence[str]) -> str:
    return "\n".join(
        [
            "# What you remember about the user",
            "",
            *(f"- {memory}" for memory in memories),
            "",
            MEMORY_HINT,
        ]
    )


def _summary(summary: str) -> str:
    return "\n".join(
        [
            "# Earlier in this conversation",
            "",
            "The start of this conversation no longer fits; this is a summary of it.",
            "",
            summary.strip(),
        ]
    )


NO_TOOLS = "You have no tools in this turn: answer in text only and do not call any function."


def _command(instructions: str | None, tools: bool) -> str | None:
    lines = [line for line in [(instructions or "").strip(), "" if tools else NO_TOOLS] if line]
    return "\n".join(["# Command", *lines]) if lines else None


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


def _session_memories(entries: Sequence[str]) -> str:
    return "\n".join(
        [
            "# This conversation's memory",
            "",
            "Kept for this conversation only; where it differs from the memory above, it wins.",
            "",
            *(f"- {entry}" for entry in entries),
        ]
    )


def _journal(notes: Sequence[str]) -> str:
    return "\n".join(
        [
            "# Lately in your other conversations",
            "",
            "Notes about what happened in your other conversations, oldest first; find "
            "older ones with journal_search.",
            "",
            *(f"- {note}" for note in notes),
        ]
    )


def build_system_prompt(
    client: ClientInfo | None,
    source: str,
    memories: Sequence[str] = (),
    *,
    session_memories: Sequence[str] = (),
    journal: Sequence[str] = (),
    now: datetime,
    tz: tzinfo | None = None,
    summary: str | None = None,
    agent_name: str | None = None,
    computer: Computer | None = None,
    command: str | None = None,
    tools: bool = True,
    more_tools: str | None = None,
    skills: str | None = None,
) -> str:
    """Build the system prompt from blocks in a fixed order.

    Stable blocks come first so providers can cache the prefix: base rules,
    persona, environment, the computer, expression rules, memory, a summary of the
    start of a long conversation, the instructions of a command sent with this
    turn's message, and the current time last because it always changes.
    Without ``tools`` the model is told so and gets no expression rules.
    Blocks with nothing to say are left out.
    ``tz`` is the user's time zone; the computer's own zone is used by default.
    ``agent_name`` turns the prompt into an agent's: its role follows the environment.
    ``more_tools`` lists the tools the model loads on demand (``tools/loading.py``).
    """
    character = client.character if client is not None else None
    blocks = [
        _base_rules(),
        _persona(character) if character else None,
        _environment(client, source),
        _computer(computer) if computer else None,
        more_tools if tools else None,
        skills if tools else None,
        _agent_role(agent_name) if agent_name else None,
        _expression(character) if character and character.expressive and tools else None,
        _memories(memories) if memories else None,
        _session_memories(session_memories) if session_memories else None,
        _journal(journal) if journal else None,
        _summary(summary) if summary else None,
        _command(command, tools),
        _time(now, tz),
    ]
    return "\n\n".join(block for block in blocks if block)


def text_tokens(text: str) -> float:
    """Rough size of a text in tokens.

    Non-ASCII letters, Cyrillic among them, cost a tokenizer more than English
    ones, so they count for more. The estimate errs on the large side: a
    shorter history is a smaller harm than a prompt the model cannot take.
    """
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars / CHARS_PER_TOKEN + (len(text) - ascii_chars) / OTHER_CHARS_PER_TOKEN


def estimate_tokens(message: Message) -> int:
    """Rough size of a message in tokens, for providers that cannot count."""
    tokens = 0.0
    images = 0
    for part in message.content:
        if isinstance(part, TextPart):
            tokens += text_tokens(part.text)
        elif isinstance(part, ImagePart):
            images += 1
        elif isinstance(part, FilePart):
            tokens += text_tokens(part.name + part.path)
    for call in message.tool_calls:
        tokens += text_tokens(call.name + call.model_dump_json(exclude={"extra"}))
    return math.ceil(tokens) + images * IMAGE_TOKENS


def _to_message(stored: StoredMessage, keep_images: bool) -> Message:
    content: list[ContentPart] = [
        TextPart(text=IMAGE_OMITTED) if isinstance(part, ImagePart) and not keep_images else part
        for part in stored.content
    ]
    if stored.role == "assistant" and not content and not stored.tool_calls:
        content = [TextPart(text=NO_ANSWER.get(stored.status or "", NO_ANSWER["error"]))]
    return Message(
        role=stored.role,
        content=content,
        tool_calls=[ToolCall.model_validate(call) for call in stored.tool_calls],
        tool_call_id=stored.tool_call_id,
        is_error=stored.is_error,
    )


def _split_into_turns(messages: Sequence[StoredMessage]) -> list[list[StoredMessage]]:
    """Group messages so that each group starts with a user message.

    A group holds a whole turn: the user message, the assistant's tool calls
    and their results. Cutting history between groups never separates a tool
    call from its result.
    """
    turns: list[list[StoredMessage]] = []
    for message in messages:
        if message.role == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    return turns


def _text_length(message: StoredMessage) -> int:
    return sum(len(part.text) for part in message.content if isinstance(part, TextPart))


def clear_old_results(
    messages: Sequence[StoredMessage], budget: int, count_fn: CountFn = estimate_tokens
) -> list[StoredMessage]:
    """The history with old tool results replaced by a note while it is over half the budget.

    The lightest way to make room, before turns drop out into the summary:
    results are cleared oldest first until the history fits into half of
    ``budget``. The latest turn, errors (they keep the model from repeating a
    mistake) and short results stay. Ids and order do not change, so indexes
    into ``messages`` still hold.
    """
    turns = _split_into_turns(messages)
    if len(turns) < 2:
        return list(messages)
    latest_start = len(messages) - len(turns[-1])

    def size(index: int, message: StoredMessage) -> int:
        return count_fn(_to_message(message, keep_images=index >= latest_start))

    result = list(messages)
    total = sum(size(index, message) for index, message in enumerate(result))
    limit = budget * CLEAR_AT
    if total <= limit:
        return result
    names = {
        str(call.get("id")): str(call.get("name", "a tool"))
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    }
    for index in range(latest_start):
        if total <= limit:
            break
        message = result[index]
        if message.role != "tool" or message.is_error or _text_length(message) < CLEAR_MIN_CHARS:
            continue
        note = CLEARED.format(tool=names.get(message.tool_call_id or "", "a tool"))
        cleared = message.model_copy(update={"content": [TextPart(text=note)]})
        total -= size(index, message) - size(index, cleared)
        result[index] = cleared
    return result


def keep_from(
    messages: Sequence[StoredMessage], budget: int, count_fn: CountFn = estimate_tokens
) -> int:
    """Index of the first message of the latest turns that fit into ``budget`` tokens.

    History is cut only between turns, and the latest turn is always kept,
    even if it alone is over budget. Everything before the index is dropped.
    """
    turns = _split_into_turns(messages)
    start = len(messages)
    used = 0
    for index in range(len(turns) - 1, -1, -1):
        is_latest = index == len(turns) - 1
        converted = [_to_message(stored, keep_images=is_latest) for stored in turns[index]]
        size = sum(count_fn(message) for message in converted)
        if not is_latest and used + size > budget:
            break
        used += size
        start -= len(turns[index])
    return start


def to_history(messages: Sequence[StoredMessage]) -> list[Message]:
    """Messages for the model; images survive only in the latest turn, others become a note."""
    turns = _split_into_turns(messages)
    return [
        _to_message(stored, keep_images=index == len(turns) - 1)
        for index, turn in enumerate(turns)
        for stored in turn
    ]


def build_history(
    messages: Sequence[StoredMessage],
    budget: int,
    count_fn: CountFn = estimate_tokens,
) -> list[Message]:
    """Pick the latest messages that fit into ``budget`` tokens, oldest first.

    History is cut only between turns. The latest turn is always included,
    even if it alone is over budget. Images survive only in the latest turn;
    in earlier ones they are replaced with a short note.
    """
    return to_history(messages[keep_from(messages, budget, count_fn) :])
