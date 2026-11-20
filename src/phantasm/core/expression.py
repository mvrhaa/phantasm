"""The character's expression: the ``set_expression`` tool and the pose and mood it picks."""

from collections.abc import Sequence
from typing import Any

from phantasm.protocol.messages import Character, ExpressionState
from phantasm.providers.base import ToolSpec

EXPRESSION_TOOL = "set_expression"
EXPRESSION_SET = "Expression set."
DEFAULT_MOOD = "neutral"

_DESCRIPTION = (
    "Show how you feel by setting your character's pose and mood. "
    "The user sees the character change, not a message. "
    "Call it whenever your feelings change, as often as needed."
)


def default_expression(character: Character) -> ExpressionState:
    """The first pose and a neutral mood, or the first mood if there is no neutral one."""
    mood = DEFAULT_MOOD if DEFAULT_MOOD in character.moods else character.moods[0]
    return ExpressionState(pose=character.poses[0], mood=mood)


def _problem(field: str, value: Any, allowed: Sequence[str]) -> str | None:
    if isinstance(value, str) and value in allowed:
        return None
    return f"{field} must be one of {', '.join(allowed)} (got {value!r})"


class ExpressionTool:
    """``set_expression`` for one character during one turn.

    The poses and moods the client declared in ``hello`` become enums in the
    tool's schema, so the model picks only from them; anything else is an
    error the model can read and fix. The tool remembers the last expression.
    """

    def __init__(self, character: Character) -> None:
        self._character = character
        self.current: ExpressionState | None = None

    @property
    def spec(self) -> ToolSpec:
        """The tool as the model sees it."""
        return ToolSpec(
            name=EXPRESSION_TOOL,
            description=_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "pose": {"type": "string", "enum": list(self._character.poses)},
                    "mood": {"type": "string", "enum": list(self._character.moods)},
                },
                "required": ["pose", "mood"],
                "additionalProperties": False,
            },
        )

    @property
    def final(self) -> ExpressionState:
        """The last expression set during the turn, or the character's default one."""
        return self.current or default_expression(self._character)

    def apply(self, tool_input: dict[str, Any]) -> ExpressionState:
        """Make the requested pose and mood the current expression.

        Raises ``ValueError`` with a message for the model if either of them
        is not one the character has; the current expression then stays.
        """
        problems = [
            problem
            for problem in (
                _problem("pose", tool_input.get("pose"), self._character.poses),
                _problem("mood", tool_input.get("mood"), self._character.moods),
            )
            if problem is not None
        ]
        if problems:
            raise ValueError(f"Invalid expression: {'; '.join(problems)}.")
        self.current = ExpressionState(pose=tool_input["pose"], mood=tool_input["mood"])
        return self.current
