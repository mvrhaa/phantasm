"""Running agents: starting them, talking to them, stopping them and hearing back.

An agent works in a session of its own. What it has to tell the session that
started it (a question, its report, its failure) arrives there as a message
with source ``agent`` that starts a turn, as soon as that session is free.
"""

import asyncio
import logging
from asyncio import FIRST_COMPLETED
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from zero.agents.types import AgentError, AgentProgress, AgentWait
from zero.approvals.manager import Answer, ApprovalAsk, Approver
from zero.core.attachments import remove_files
from zero.core.cancellation import CancelToken
from zero.core.events import EventBus
from zero.core.turn_manager import TurnManager
from zero.protocol.messages import (
    AgentInfo,
    AgentStatus,
    AgentUpdated,
    SessionCreated,
    SessionDeleted,
    TextPart,
    TurnFinished,
)
from zero.store.agents import Agent, AgentsRepo
from zero.store.messages import MessagesRepo
from zero.store.sessions import SessionsRepo

NO_REPORT = "(the agent finished without a report)"
VERIFY_ROUNDS = 2
REVIEWER = "reviewer"
VERIFY_PROMPT = (
    "You review work another assistant says it finished. Be skeptical: the author tends to "
    "praise its own work. Check the work yourself with your tools (read the files, run the "
    "tests or commands); do not trust the report. Do not fix anything. Start your answer with "
    "PASS or FAIL on the first line, then explain briefly. FAIL if any part of the task is "
    "missing, broken or unverified.\n\nThe task:\n{task}\n\nThe author's report:\n{report}"
)
FIX = "A reviewer checked your work and it did not pass:\n{verdict}\nFix this, then report again."
RESTARTED = "Zero was restarted before the agent finished."
LAST_OUTPUT_CHARS = 1000
STATUS_MESSAGES = 200

StartAgentTurn = Callable[[Agent, str], Awaitable[None]]
"""Starts a turn with this text in the agent's session; it must call ``finished`` after."""
StartNoticeTurn = Callable[[str, str], Awaitable[bool]]
"""Starts a turn in a session with this text from an agent; False if the session is busy."""

logger = logging.getLogger(__name__)


def agent_info(agent: Agent) -> AgentInfo:
    """What clients are told about an agent."""
    return AgentInfo.model_validate(agent.model_dump())


def _who(agent: Agent) -> str:
    return f"Agent '{agent.name}' (id {agent.id})"


@dataclass(eq=False)
class Mail:
    """What an agent has to tell a session, until a turn or ``wait`` takes it."""

    session_id: str
    agent_id: str
    text: str


class AgentRunner:
    """Starts agents in sessions of their own and carries messages between them and parents.

    ``start_turn`` and ``start_notice`` come from the server, which knows how to
    build turns. ``start_turn`` must call ``finished`` once the agent's turn is over.
    """

    def __init__(
        self,
        *,
        agents: AgentsRepo,
        sessions: SessionsRepo,
        messages: MessagesRepo,
        events: EventBus,
        turns: TurnManager,
        start_turn: StartAgentTurn,
        start_notice: StartNoticeTurn,
        max_agents: int = 3,
        claude_ready: Callable[[], bool] = lambda: False,
    ) -> None:
        self._agents = agents
        self._sessions = sessions
        self._messages = messages
        self._events = events
        self._turns = turns
        self._start_turn = start_turn
        self._start_notice = start_notice
        self._max_agents = max_agents
        self._claude_ready = claude_ready
        self._deliveries: set[asyncio.Task[None]] = set()
        self._questions: dict[str, list[str]] = {}
        self._mail: list[Mail] = []
        self._bell = asyncio.Event()
        self._closed = False

    async def start(
        self,
        name: str,
        task: str,
        parent_session_id: str,
        verify: bool = False,
        claude: bool = False,
    ) -> Agent:
        """Create the agent's session, store the agent and start its first turn on ``task``.

        With ``verify`` a reviewer checks its work before its report goes to the parent;
        with ``claude`` it thinks through the user's Claude Code.
        """
        if await self._agents.by_session(parent_session_id) is not None:
            raise AgentError("an agent cannot start agents")
        if claude and not self._claude_ready():
            raise AgentError("Claude Code is not installed, so there is no Claude agent")
        at_work = len([a for a in await self._agents.active() if a.review_of is None])
        if at_work >= self._max_agents:
            raise AgentError(
                f"{at_work} agents are already at work and the limit is {self._max_agents}; "
                "wait for one to finish or stop one"
            )
        session = await self._sessions.create(f"Agent: {' '.join(name.split())}", kind="agent")
        try:
            agent = await self._agents.add(
                name, parent_session_id, session.id, task, verify=verify, claude=claude
            )
        except ValueError as exc:
            await self._sessions.delete(session.id)
            raise AgentError(str(exc)) from exc
        await self._events.publish(SessionCreated(session=session.info()))
        await self._announce(agent)
        logger.info("Agent %s (%s) started by session %s", agent.id, agent.name, parent_session_id)
        await self._start_turn(agent, agent.task)
        return agent

    async def message(self, agent_id: str, text: str, parent_session_id: str) -> Agent:
        """Send the agent a message: an answer, a correction or more work. It gets back to work."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            raise AgentError(f"{_who(agent)} was stopped and cannot get messages")
        if self._turns.busy(agent.session_id):
            raise AgentError(f"{_who(agent)} is still working; wait for its report or stop it")
        updated = await self._set(agent, "working")
        await self._start_turn(updated, text)
        return updated

    async def progress(self, agent_id: str, parent_session_id: str) -> AgentProgress:
        """The agent's status, how many steps it took and the last thing it wrote."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            return AgentProgress(agent, 0, None)
        stored = await self._messages.list(agent.session_id, limit=STATUS_MESSAGES)
        replies = [m for m in stored if m.role == "assistant"]
        texts = ["".join(p.text for p in m.content if isinstance(p, TextPart)) for m in replies]
        last = next((text.strip() for text in reversed(texts) if text.strip()), None)
        if last is not None and len(last) > LAST_OUTPUT_CHARS:
            last = "…" + last[-(LAST_OUTPUT_CHARS - 1) :]
        return AgentProgress(agent, len(replies), last)

    async def stop(self, agent_id: str, parent_session_id: str) -> Agent:
        """Cancel the agent's work and delete its session; its record stays."""
        agent = await self._own(agent_id, parent_session_id)
        if agent.session_id is None:
            return agent
        if agent.active:
            agent = await self._set(agent, "stopped")
        await self._turns.stop_session(agent.session_id)
        files = await self._messages.file_paths(agent.session_id)
        await self._sessions.delete(agent.session_id)
        await remove_files(files)
        await self._events.publish(SessionDeleted(session_id=agent.session_id))
        stopped = await self._agents.get(agent.id)
        assert stopped is not None
        await self._announce(stopped)
        logger.info("Agent %s stopped", agent.id)
        return stopped

    async def session_closing(self, session_id: str) -> None:
        """A session is about to be deleted: stop what agents have to do with it.

        The agents it started are stopped and their sessions deleted; if it is
        an agent's own session, that agent is marked as stopped.
        """
        own = await self._agents.by_session(session_id)
        if own is not None and own.active:
            await self._set(own, "stopped")
        for agent in await self._agents.list(session_id):
            if agent.session_id is not None:
                await self.stop(agent.id, session_id)

    async def ask(self, session_id: str, question: str) -> Agent:
        """An agent asks the session that started it; the answer comes as an ``agent_message``.

        The question goes out once the agent's turn is over, so the agent is
        free to take the answer as soon as the parent sees the question.
        """
        agent = await self._agents.by_session(session_id)
        if agent is None:
            raise AgentError("only agents can ask the conversation that started them")
        self._questions.setdefault(agent.id, []).append(question.strip())
        return await self._set(agent, "asking")

    async def finished(self, agent_id: str, result: TurnFinished | None) -> None:
        """Record how an agent's turn ended and tell its parent what it has to say.

        That is its questions if it asked any (it keeps waiting for the answer),
        otherwise its report or its failure. A stopped agent stays stopped.
        """
        questions = self._questions.pop(agent_id, [])
        agent = await self._agents.get(agent_id)
        if agent is None or agent.status not in ("working", "asking"):
            return
        cancelled = result is not None and result.status == "cancelled"
        if agent.review_of is not None:
            await self._reviewed(agent, result)
        elif cancelled:
            await self._set(agent, "stopped")
        elif agent.status == "asking":
            if questions:
                self._deliver(
                    agent.parent_session_id,
                    f"{_who(agent)} asks:\n" + "\n\n".join(questions) + "\n\n"
                    "Answer it with agent_message, or ask the user first if only they know.",
                    agent.id,
                )
        elif result is None or result.status == "error":
            error = (result.error if result is not None else None) or "internal error"
            failed = await self._set(agent, "failed", error)
            self._deliver(agent.parent_session_id, f"{_who(failed)} failed: {error}", agent.id)
        elif agent.verify and agent.reviews < VERIFY_ROUNDS and agent.session_id is not None:
            report = result.text.strip() or NO_REPORT
            await self._agents.set_status(agent.id, "working", report)
            await self._review(agent, report)
        else:
            done = await self._set(agent, "done", result.text.strip() or NO_REPORT)
            self._deliver(
                agent.parent_session_id,
                f"{_who(done)} has finished. Its report:\n{done.report}",
                agent.id,
            )

    async def note_progress(self, session_id: str, note: str) -> Agent:
        """An agent records how far it got; its parent sees it in agent_status."""
        agent = await self._agents.by_session(session_id)
        if agent is None:
            raise AgentError("only an agent records progress")
        updated = await self._agents.set_progress(agent.id, note.strip())
        assert updated is not None
        await self._announce(updated)
        return updated

    async def _review(self, agent: Agent, report: str) -> None:
        """Start a reviewer on the agent's work; the agent stays at work meanwhile."""
        assert agent.session_id is not None
        prompt = VERIFY_PROMPT.format(task=agent.task, report=report)
        session = await self._sessions.create(f"Review: {agent.name}", kind="agent")
        reviewer = await self._agents.add(
            REVIEWER, agent.session_id, session.id, prompt, review_of=agent.id, claude=agent.claude
        )
        await self._events.publish(SessionCreated(session=session.info()))
        await self._announce(reviewer)
        await self._start_turn(reviewer, prompt)

    async def _reviewed(self, reviewer: Agent, result: TurnFinished | None) -> None:
        """A reviewer is done: a pass sends the agent's report on, a failure sends the agent
        back to work, until ``VERIFY_ROUNDS`` reviews were made."""
        ok = result is not None and result.status == "ok"
        verdict = result.text.strip() if ok and result is not None else ""
        await self._set(reviewer, "done" if ok else "failed", verdict or None)
        agent = await self._agents.get(reviewer.review_of or "")
        if agent is None or agent.status != "working":
            return
        agent = await self._agents.count_review(agent.id)
        assert agent is not None
        passed = verdict.lstrip().upper().startswith("PASS")
        if ok and not passed and agent.reviews < VERIFY_ROUNDS:
            await self._start_turn(agent, FIX.format(verdict=verdict))
            return
        if passed:
            note = f"A reviewer checked its work: {verdict}"
        elif ok:
            note = f"Its work did not pass {agent.reviews} reviews. The last one:\n{verdict}"
        else:
            note = "Its work could not be reviewed (the reviewer did not finish)."
        done = await self._set(agent, "done")
        self._deliver(
            agent.parent_session_id,
            f"{_who(done)} has finished. {note}\nIts report:\n{done.report}",
            agent.id,
        )

    async def waiting(self, agent_id: str, waiting: bool) -> None:
        """An agent's turn starts or stops waiting for the user's approval."""
        before, after = ("working", "waiting_user") if waiting else ("waiting_user", "working")
        agent = await self._agents.get(agent_id)
        if agent is not None and agent.status == before:
            await self._set(agent, after)

    async def recover(self, resume: bool = False) -> list[Agent]:
        """The agents an earlier run left at work. With ``resume`` they stay at work for the
        caller to start again (see ``give_up``); else they are marked failed."""
        if not resume:
            stale = await self._agents.active()
            for agent in stale:
                await self._agents.set_status(agent.id, "failed", RESTARTED)
            if stale:
                logger.info("%d agents were left unfinished by the last run", len(stale))
            return stale
        stale = await self._agents.active()
        if stale:
            logger.info("%d agents left unfinished by the last run go on", len(stale))
        return [await self._agents.set_status(agent.id, "working") or agent for agent in stale]

    async def give_up(self, agent: Agent) -> None:
        """An agent the restart could not take up again has failed."""
        await self._set(agent, "failed", RESTARTED)

    async def close(self) -> None:
        """Drop the messages still waiting for their session; nothing is delivered after this."""
        self._closed = True
        for task in self._deliveries:
            task.cancel()
        await asyncio.gather(*self._deliveries, return_exceptions=True)

    async def _own(self, agent_id: str, parent_session_id: str) -> Agent:
        agent = await self._agents.get(agent_id)
        if agent is None or agent.parent_session_id != parent_session_id:
            raise AgentError(f"this conversation has no agent with id {agent_id}")
        return agent

    async def _set(self, agent: Agent, status: AgentStatus, report: str | None = None) -> Agent:
        updated = await self._agents.set_status(agent.id, status, report)
        assert updated is not None
        await self._announce(updated)
        return updated

    async def _announce(self, agent: Agent) -> None:
        await self._events.publish(AgentUpdated(agent=agent_info(agent)))

    async def wait(
        self,
        parent_session_id: str,
        agent_ids: Sequence[str],
        timeout_s: float,
        cancel: CancelToken,
    ) -> AgentWait:
        """Wait until the parent's agents (those of ``agent_ids``, or all) have something to
        say, and take it: what this returns does not come again as a message."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            bell = self._bell
            mine = [
                mail
                for mail in self._mail
                if mail.session_id == parent_session_id
                and (not agent_ids or mail.agent_id in agent_ids)
            ]
            if mine:
                self._mail = [mail for mail in self._mail if mail not in mine]
                return AgentWait([mail.text for mail in mine], [])
            working = [
                agent
                for agent in await self._agents.active(parent_session_id)
                if agent.review_of is None and (not agent_ids or agent.id in agent_ids)
            ]
            left = deadline - asyncio.get_running_loop().time()
            if not working or left <= 0 or cancel.cancelled:
                return AgentWait([], [_who(agent) for agent in working])
            rung = asyncio.ensure_future(bell.wait())
            stopped = asyncio.ensure_future(cancel.wait())
            try:
                await asyncio.wait({rung, stopped}, timeout=left, return_when=FIRST_COMPLETED)
            finally:
                rung.cancel()
                stopped.cancel()

    def _deliver(self, session_id: str, text: str, agent_id: str = "") -> None:
        """Leave the message for the session: ``wait`` takes it, else a turn brings it when
        the session is free."""
        if self._closed:
            return
        mail = Mail(session_id, agent_id, text)
        self._mail.append(mail)
        self._bell.set()
        self._bell = asyncio.Event()
        task = asyncio.create_task(self._deliver_when_free(mail), name="agent-notice")
        self._deliveries.add(task)
        task.add_done_callback(self._deliveries.discard)

    async def _deliver_when_free(self, mail: "Mail") -> None:
        while True:
            await self._turns.wait_free(mail.session_id)
            if mail not in self._mail:
                return
            if await self._sessions.get(mail.session_id) is None:
                self._mail.remove(mail)
                return
            self._mail.remove(mail)
            if await self._start_notice(mail.session_id, mail.text):
                return
            self._mail.append(mail)


class AgentApprovals:
    """Approvals of an agent's turn: while one waits, the agent's status is ``waiting_user``."""

    def __init__(self, approvals: Approver, runner: AgentRunner, agent_id: str) -> None:
        self._approvals = approvals
        self._runner = runner
        self._agent_id = agent_id

    async def request(self, ask: ApprovalAsk, cancel: CancelToken) -> Answer:
        """Ask the user as usual, with the agent's status showing the wait."""
        await self._runner.waiting(self._agent_id, True)
        try:
            return await self._approvals.request(ask, cancel)
        finally:
            await self._runner.waiting(self._agent_id, False)
