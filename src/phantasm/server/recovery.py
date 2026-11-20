"""After a crash: the turns and agents it cut off go on.

While a turn runs it is noted in ``open_turns`` (``TurnManager``); a turn that
ends in any way, a cancelled one too, is crossed out. So what is left there at
start is what a crash interrupted. Each such session gets a turn with source
``recovery`` that tells the model to go on, and agents left at work get the
same as a message. Turns older than ``RESUME_WITHIN`` are let go.
"""

import logging
from datetime import timedelta

from phantasm import clock
from phantasm.server.state import AppState
from phantasm.server.turns import make_turn, start_agent_turn
from phantasm.store.agents import Agent
from phantasm.store.open_turns import OpenTurn

logger = logging.getLogger(__name__)

RESUME_WITHIN = timedelta(hours=2)
CONTINUE = (
    "[Phantasm was restarted while working on this. Continue where you stopped: check what is "
    "already done before you repeat anything.]"
)


async def resume_after_restart(
    state: AppState, agents: list[Agent], open_turns: list[OpenTurn]
) -> int:
    """Start again what a crash cut off; returns how many turns were started."""
    now = clock.now()
    started = 0
    for agent in agents:
        if state.provider is None or agent.session_id is None:
            await state.agent_runner.give_up(agent)
            continue
        if now - agent.updated_at > RESUME_WITHIN:
            await state.agent_runner.give_up(agent)
            continue
        await start_agent_turn(state, agent, CONTINUE)
        started += 1
    done: set[str] = set()
    for turn in open_turns:
        if turn.agent_id is not None or turn.session_id in done:
            continue
        if state.provider is None or now - turn.started_at > RESUME_WITHIN:
            continue
        if await state.sessions.get(turn.session_id) is None or state.turns.busy(turn.session_id):
            continue
        done.add(turn.session_id)
        state.turns.start(
            make_turn(
                state, state.provider, session_id=turn.session_id, text=CONTINUE, source="recovery"
            )
        )
        started += 1
    if started:
        logger.info("%d turns cut off by a restart go on", started)
    return started
