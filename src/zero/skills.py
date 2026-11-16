"""Skills, as Claude has them: instructions for a kind of task, loaded only when a task needs them.

A skill is a folder in ``skills/`` of Zero's data folder with a ``SKILL.md``:
YAML frontmatter, then the instructions, plus any files they point to. The
prompt lists each skill's name and description; when a task matches one, the
model loads its instructions with ``skill_use`` and reads its other files only
when they say to. The user starts one with ``/name arguments``.

Frontmatter fields, as Claude Code reads them: ``name`` (lowercase letters,
digits and hyphens, at most 64; the folder's name without it), ``description``
(at most 1024 characters; without one, the first line of the instructions),
``argument-hint``, ``disable-model-invocation`` (true: only the user starts it),
``user-invocable`` (false: only the model does). Other fields are ignored. The
folder is read again whenever the prompt is built, so a skill the model writes
counts from the next turn.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from zero.paths import home_dir

NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
NAME_CHARS, DESCRIPTION_CHARS = 64, 1024
MAX_SKILLS = 100
TRUE = {"true", "yes", "on", "1"}
PLACEHOLDER = re.compile(r"\$ARGUMENTS\[(\d+)\]|\$ARGUMENTS\b|\$(\d+)|\$\{(?:CLAUDE_)?SKILL_DIR\}")
USER_START = re.compile(r"^/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+(.*))?$", re.S)


@dataclass(frozen=True)
class Skill:
    name: str
    folder: Path
    description: str
    body: str
    argument_hint: str = ""
    model: bool = True  # the model may load it
    user: bool = True  # the user may start it with /name


def parse(text: str) -> tuple[dict[str, str], str]:
    """Frontmatter fields and the body of a SKILL.md. Keys with plain, quoted or
    folded (``>``, ``|``) values are read; anything else is ignored."""
    if not text.startswith("---"):
        return {}, text.strip()
    lines = text.splitlines()
    end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}, text.strip()
    fields: dict[str, str] = {}
    index = 1
    while index < end:
        match = re.match(r"([A-Za-z][\w-]*)\s*:\s*(.*)$", lines[index])
        index += 1
        if match is None:
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        if value in (">", "|", ">-", "|-"):
            block = []
            while index < end and (not lines[index].strip() or lines[index][:1].isspace()):
                block.append(lines[index].strip())
                index += 1
            joiner = " " if value.startswith(">") else "\n"
            value = joiner.join(block).strip()
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key] = value
    return fields, "\n".join(lines[end + 1 :]).strip()


def read_skill(folder: Path) -> Skill | None:
    """The skill in ``folder``, or None when it has no readable SKILL.md or a bad name."""
    try:
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    fields, body = parse(text)
    name = fields.get("name") or folder.name
    if len(name) > NAME_CHARS or not NAME.fullmatch(name):
        return None
    description = fields.get("description") or (body.splitlines()[0] if body else "")
    return Skill(
        name=name,
        folder=folder,
        description=description[:DESCRIPTION_CHARS],
        body=body,
        argument_hint=fields.get("argument-hint", ""),
        model=fields.get("disable-model-invocation", "").lower() not in TRUE,
        user=fields.get("user-invocable", "true").lower() in TRUE,
    )


class SkillBook:
    """The skills in a folder, read anew on every call."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or home_dir() / "skills"

    def all(self) -> list[Skill]:
        if not self.folder.is_dir():
            return []
        found: dict[str, Skill] = {}
        for folder in sorted(self.folder.iterdir())[: MAX_SKILLS * 2]:
            skill = read_skill(folder) if folder.is_dir() else None
            if skill is not None and skill.name not in found:
                found[skill.name] = skill
        return list(found.values())[:MAX_SKILLS]

    def get(self, name: str) -> Skill | None:
        return next((skill for skill in self.all() if skill.name == name), None)

    def prompt_block(self) -> str | None:
        """The ``# Skills`` block: what they are and the ones the model may load; None
        when there are none."""
        skills = [skill for skill in self.all() if skill.model]
        lines = [
            "# Skills",
            "Skills are instructions for kinds of tasks. When a task matches a skill below, "
            "load it with skill_use before you start, and follow it. A skill is a folder with "
            "a SKILL.md (frontmatter with name and description, then the instructions); read "
            "its other files with read_file when it says to. To make a skill the user asks "
            f"for, write {self.folder}/<name>/SKILL.md.",
        ]
        if not skills:
            return None
        lines += [f"- {skill.name}: {skill.description}" for skill in skills]
        return "\n".join(lines)


def expand(skill: Skill, arguments: str = "") -> str:
    """The skill's instructions with ``$ARGUMENTS``, ``$0``… and ``${SKILL_DIR}`` filled in;
    without a placeholder the arguments follow the instructions."""
    words = arguments.split()
    used = False

    def fill(match: re.Match[str]) -> str:
        nonlocal used
        if match.group(0).startswith("${"):
            return str(skill.folder)
        used = True
        index = match.group(1) or match.group(2)
        if index is None:
            return arguments
        position = int(index)
        return words[position] if position < len(words) else ""

    body = PLACEHOLDER.sub(fill, skill.body)
    if arguments and not used:
        body += f"\n\nARGUMENTS: {arguments}"
    return body


def user_start(text: str, book: SkillBook) -> tuple[Skill, str] | None:
    """The skill a message starts with ``/name``, and its arguments, if there is one."""
    match = USER_START.match(text.strip())
    if match is None:
        return None
    skill = book.get(match.group(1))
    if skill is None or not skill.user:
        return None
    return skill, (match.group(2) or "").strip()
