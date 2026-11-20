"""``skill.list``: the skills a client offers after ``/``."""

import asyncio

from phantasm.protocol.messages import SkillInfo, SkillList, SkillListOk
from phantasm.server.connections import Connection
from phantasm.server.state import AppState


async def handle_skill_list(connection: Connection, message: SkillList, state: AppState) -> None:
    """Answer with the skills the user may start, sorted by name."""
    skills = await asyncio.to_thread(state.skills.all)
    infos = sorted(
        (
            SkillInfo(
                name=skill.name, description=skill.description, argument_hint=skill.argument_hint
            )
            for skill in skills
            if skill.user
        ),
        key=lambda info: info.name,
    )
    await connection.send(SkillListOk(reply_to=message.id, skills=infos))
