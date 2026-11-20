"""Project memory: what Phantasm keeps about each of the user's projects between conversations.

A project is a Markdown file ``projects/<name>.md`` in Phantasm's data folder: frontmatter
with ``name``, ``path`` and ``summary``, then the memory itself (structure, how to run and
test it, conventions, decisions and agreements). The user can read and edit these files.

The prompt lists every project in one line each. When the user's recent messages name a
project (its name, or its folder's name or path), its whole memory is put in the prompt
too, so the model knows it without being told again. The model keeps the files up to date
with ``project_write`` and ``project_note``.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from phantasm.paths import home_dir
from phantasm.skills import parse

NAME = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
NAME_CHARS = 64
SUMMARY_CHARS = 300
MAX_PROJECTS = 50
MEMORY_PROMPT_CHARS = 6000
MAX_SHOWN = 2
NOTES = "## Notes"


@dataclass(frozen=True)
class Project:
    name: str
    path: str
    summary: str
    body: str
    file: Path

    def words(self) -> set[str]:
        """How the user may name it: the name, with spaces for its dashes, the folder's name
        and its path."""
        folder = Path(self.path).expanduser()
        found = {self.name, re.sub(r"[-_.]", " ", self.name), folder.name, self.path}
        if self.path:
            found.add(str(folder))
        return {word.lower() for word in found if len(word) >= 3}


def valid_name(name: str) -> bool:
    return len(name) <= NAME_CHARS and NAME.fullmatch(name) is not None


def _render(name: str, path: str, summary: str, body: str) -> str:
    header = ["---", f"name: {name}", f"path: {path}", f"summary: {summary}", "---", ""]
    return "\n".join(header) + body.strip() + "\n"


class ProjectBook:
    """The project files in a folder, read anew on every call."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or home_dir() / "projects"

    def all(self) -> list[Project]:
        if not self.folder.is_dir():
            return []
        projects = []
        for file in sorted(self.folder.glob("*.md"))[:MAX_PROJECTS]:
            project = self._read(file)
            if project is not None:
                projects.append(project)
        return projects

    def get(self, name: str) -> Project | None:
        return next((project for project in self.all() if project.name == name), None)

    def write(self, name: str, *, path: str, summary: str, body: str) -> Project:
        """Creates the project or replaces its memory; notes not in ``body`` are lost, so
        write the whole memory."""
        self.folder.mkdir(parents=True, exist_ok=True)
        file = self.folder / f"{name}.md"
        file.write_text(
            _render(name, path, " ".join(summary.split())[:SUMMARY_CHARS], body),
            encoding="utf-8",
        )
        project = self._read(file)
        assert project is not None
        return project

    def note(self, name: str, text: str, now: datetime) -> Project | None:
        """Adds one dated line under ``## Notes``; None when there is no such project."""
        project = self.get(name)
        if project is None:
            return None
        line = f"- {now.date().isoformat()}: {' '.join(text.split())}"
        body = project.body.rstrip()
        body = f"{body}\n{line}" if NOTES in body else f"{body}\n\n{NOTES}\n{line}"
        return self.write(name, path=project.path, summary=project.summary, body=body)

    def mentioned(self, texts: list[str]) -> list[Project]:
        """The projects the texts name, the most recently named first."""
        found: list[Project] = []
        projects = self.all()
        for text in reversed(texts):
            lowered = text.lower()
            for project in projects:
                if project not in found and any(
                    re.search(rf"(?<![\w/]){re.escape(word)}(?![\w])", lowered)
                    for word in project.words()
                ):
                    found.append(project)
        return found

    def prompt_block(self, texts: list[str]) -> str:
        """``# Projects``: how project memory works, every project in one line, and the
        whole memory of the projects the user's recent messages name."""
        projects = self.all()
        lines = [
            "# Projects",
            "You keep a memory of each of the user's projects, so they never have to explain "
            "a project twice. When you learn how a project is built (its structure, how to run "
            "and test it, its conventions), save it with project_write; when the user and you "
            "decide or agree on something, add it with project_note. Read a project's memory "
            "with project_read before working on it, unless it is shown below.",
        ]
        if projects:
            lines.append("Known projects:")
            lines += [
                f"- {project.name} ({project.path or 'no folder'}): {project.summary}"
                for project in projects
            ]
        for project in self.mentioned(texts)[:MAX_SHOWN]:
            memory = project.body
            if len(memory) > MEMORY_PROMPT_CHARS:
                memory = memory[:MEMORY_PROMPT_CHARS] + "\n… (read the rest with project_read)"
            lines += ["", f"# Project {project.name} ({project.path})", memory]
        return "\n".join(lines)

    def _read(self, file: Path) -> Project | None:
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        fields, body = parse(text)
        name = fields.get("name") or file.stem
        if not valid_name(name):
            return None
        return Project(
            name=name,
            path=fields.get("path", ""),
            summary=fields.get("summary", ""),
            body=body,
            file=file,
        )
