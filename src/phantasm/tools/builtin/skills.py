"""``skill_use``: the instructions of a skill (``phantasm/skills.py``) for the task at hand."""

from pydantic import BaseModel, Field

from phantasm.skills import SkillBook, expand
from phantasm.tools.base import Tool, ToolContext, ToolResult, error_result, text_result

MAX_FILES = 30


class SkillUseInput(BaseModel):
    name: str = Field(min_length=1, description="The skill's name from # Skills")
    arguments: str = Field(default="", description="What the skill works on, if it takes that")


class SkillUse(Tool[SkillUseInput]):
    """Loads a skill: its instructions and the list of its other files."""

    name = "skill_use"
    input_model = SkillUseInput
    default_policy = "allow"

    def __init__(self, book: SkillBook) -> None:
        self._book = book
        self.description = (
            "Load a skill from # Skills before a task it matches: you get its instructions "
            f"and its files. Skills are folders in {book.folder} with a SKILL.md."
        )

    def label(self, params: SkillUseInput) -> str:
        return params.name

    async def run(self, params: SkillUseInput, ctx: ToolContext) -> ToolResult:
        skill = self._book.get(params.name)
        if skill is None or not skill.model:
            names = ", ".join(s.name for s in self._book.all() if s.model) or "none"
            return error_result(f"No skill {params.name!r}. Skills: {names}.")
        files = sorted(
            str(path.relative_to(skill.folder))
            for path in skill.folder.rglob("*")
            if path.is_file() and path.name != "SKILL.md"
        )[:MAX_FILES]
        lines = [f"# Skill {skill.name} (folder {skill.folder})", expand(skill, params.arguments)]
        if files:
            lines.append("Files of this skill: " + ", ".join(files))
        return text_result("\n\n".join(lines), summary=skill.name)
