"""Approval rules repository: the tools the user chose to always allow."""

from datetime import datetime
from typing import Literal

import aiosqlite
from pydantic import BaseModel

from phantasm import clock
from phantasm.store.db import Database


class ApprovalRule(BaseModel):
    """A standing answer for every call of one tool."""

    tool: str
    decision: Literal["allow"]
    created_at: datetime


def _to_rule(row: aiosqlite.Row) -> ApprovalRule:
    return ApprovalRule(
        tool=row["tool"], decision=row["decision"], created_at=clock.from_iso(row["created_at"])
    )


class ApprovalRulesRepo:
    """Stores "always allow" answers. Every change is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def allow_always(self, tool: str) -> ApprovalRule:
        """Remember that ``tool`` may run without asking; the first rule is kept as it is."""
        await self._db.conn.execute(
            "INSERT OR IGNORE INTO approval_rules (tool, decision, created_at) VALUES (?, ?, ?)",
            (tool, "allow", clock.to_iso(clock.now())),
        )
        await self._db.conn.commit()
        async with self._db.conn.execute(
            "SELECT * FROM approval_rules WHERE tool = ?", (tool,)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return _to_rule(row)

    async def always_allowed(self) -> set[str]:
        """Names of the tools the user chose to always allow."""
        async with self._db.conn.execute(
            "SELECT tool FROM approval_rules WHERE decision = 'allow'"
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["tool"] for row in rows}
