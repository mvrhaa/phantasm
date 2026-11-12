"""Usage repository: the record of model calls, and totals over a period."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from zero import clock
from zero.store.db import Database

Purpose = Literal["turn", "agent", "title", "summary"]


@dataclass
class ModelCall:
    """One request to a model, as it is recorded."""

    provider: str
    model: str
    purpose: Purpose
    session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True  # False if it failed or was cancelled; it still counts as a request


class UsageTotal(BaseModel):
    """What one model of one provider was used for over a period."""

    provider: str
    model: str
    requests: int
    failed: int
    input_tokens: int
    output_tokens: int


class UsageRepo:
    """Stores model calls. Every call is committed immediately."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(self, call: ModelCall) -> None:
        """Write one model call down with the current time."""
        await self._db.conn.execute(
            "INSERT INTO model_calls (id, provider, model, purpose, session_id, input_tokens, "
            "output_tokens, ok, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                call.provider,
                call.model,
                call.purpose,
                call.session_id,
                call.input_tokens,
                call.output_tokens,
                int(call.ok),
                clock.to_iso(clock.now()),
            ),
        )
        await self._db.conn.commit()

    async def totals(self, since: datetime) -> list[UsageTotal]:
        """Requests and tokens of every provider and model since ``since``."""
        async with self._db.conn.execute(
            "SELECT provider, model, COUNT(*) AS requests, SUM(1 - ok) AS failed, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens "
            "FROM model_calls WHERE created_at >= ? GROUP BY provider, model "
            "ORDER BY provider, model",
            (clock.to_iso(since),),
        ) as cursor:
            rows = await cursor.fetchall()
        return [UsageTotal(**dict(row)) for row in rows]
