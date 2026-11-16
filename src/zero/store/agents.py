"""Agents repository: who was started, by which session, and how they are doing."""

import uuid
from datetime import datetime

import aiosqlite
from pydantic import BaseModel

from zero import clock
from zero.protocol.messages import AgentStatus
from zero.store.db import Database

ACTIVE_STATUSES: tuple[AgentStatus, ...] = ("working", "asking", "waiting_user")
FINAL_STATUSES: tuple[AgentStatus, ...] = ("done", "failed", "stopped")


class Agent(BaseModel):
    """A stored agent."""

    id: str
    name: str
    parent_session_id: str
    session_id: str | None
    task: str
    status: AgentStatus
    report: str | None
    created_at: datetime
    updated_at: datetime
    # A skeptical reviewer checks its work before the report goes out (up to two rounds).
    verify: bool = False
    reviews: int = 0
    # Set for a reviewer: the agent whose work it checks.
    review_of: str | None = None
    # The agent's latest progress note (report_progress).
    progress: str | None = None

    @property
    def active(self) -> bool:
        """Still at work: not done, failed or stopped."""
        return self.status in ACTIVE_STATUSES


def _to_agent(row: aiosqlite.Row) -> Agent:
    return Agent(
        id=row["id"],
        name=row["name"],
        parent_session_id=row["parent_session_id"],
        session_id=row["session_id"],
        task=row["task"],
        status=row["status"],
        report=row["report"],
        created_at=clock.from_iso(row["created_at"]),
        updated_at=clock.from_iso(row["updated_at"]),
        verify=bool(row["verify"]),
        reviews=row["reviews"],
        review_of=row["review_of"],
        progress=row["progress"],
    )


class AgentsRepo:
    """Stores agents. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        name: str,
        parent_session_id: str,
        session_id: str,
        task: str,
        *,
        verify: bool = False,
        review_of: str | None = None,
    ) -> Agent:
        """Store a new agent at work. Raises ``ValueError`` if the name or task is empty."""
        name = " ".join(name.split())
        task = task.strip()
        if not name:
            raise ValueError("an agent needs a name")
        if not task:
            raise ValueError("an agent needs a task")
        agent_id = str(uuid.uuid4())
        now = clock.to_iso(clock.now())
        await self._db.conn.execute(
            "INSERT INTO agents (id, name, parent_session_id, session_id, task, created_at, "
            "updated_at, verify, review_of) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_id, name, parent_session_id, session_id, task, now, now, int(verify), review_of),
        )
        await self._db.conn.commit()
        agent = await self.get(agent_id)
        assert agent is not None
        return agent

    async def get(self, agent_id: str) -> Agent | None:
        """The agent with this id, or None."""
        async with self._db.conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)) as cur:
            row = await cur.fetchone()
        return _to_agent(row) if row else None

    async def by_session(self, session_id: str) -> Agent | None:
        """The agent that works in this session, or None for an ordinary session."""
        async with self._db.conn.execute(
            "SELECT * FROM agents WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return _to_agent(row) if row else None

    async def set_status(
        self, agent_id: str, status: AgentStatus, report: str | None = None
    ) -> Agent | None:
        """Change an agent's status; ``report`` replaces the report when given."""
        sql = "UPDATE agents SET status = ?, updated_at = ?"
        params: list[str] = [status, clock.to_iso(clock.now())]
        if report is not None:
            sql += ", report = ?"
            params.append(report)
        cursor = await self._db.conn.execute(f"{sql} WHERE id = ?", (*params, agent_id))
        await self._db.conn.commit()
        return await self.get(agent_id) if cursor.rowcount else None

    async def set_progress(self, agent_id: str, note: str) -> Agent | None:
        """Keep the agent's latest progress note."""
        await self._db.conn.execute(
            "UPDATE agents SET progress = ?, updated_at = ? WHERE id = ?",
            (note, clock.to_iso(clock.now()), agent_id),
        )
        await self._db.conn.commit()
        return await self.get(agent_id)

    async def count_review(self, agent_id: str) -> Agent | None:
        """One more review of the agent's work is done."""
        await self._db.conn.execute(
            "UPDATE agents SET reviews = reviews + 1 WHERE id = ?", (agent_id,)
        )
        await self._db.conn.commit()
        return await self.get(agent_id)

    async def active(self, parent_session_id: str | None = None) -> list[Agent]:
        """Agents still at work, the oldest first; all of them or those of one parent."""
        marks = ", ".join("?" * len(ACTIVE_STATUSES))
        sql = f"SELECT * FROM agents WHERE status IN ({marks})"
        params: list[str] = list(ACTIVE_STATUSES)
        if parent_session_id is not None:
            sql += " AND parent_session_id = ?"
            params.append(parent_session_id)
        async with self._db.conn.execute(f"{sql} ORDER BY created_at, rowid", params) as cur:
            rows = await cur.fetchall()
        return [_to_agent(row) for row in rows]

    async def list(self, parent_session_id: str) -> list[Agent]:
        """Every agent a session started, the oldest first."""
        async with self._db.conn.execute(
            "SELECT * FROM agents WHERE parent_session_id = ? ORDER BY created_at, rowid",
            (parent_session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [_to_agent(row) for row in rows]
